"""
Joint Training Script for Option 2 Watermarking Framework
Implements Adila's thesis design: LPCNet + DSS + DNN

Training phases:
1. LPCNet + DSS (no attack) - embedding, addition
2. LPCNet + DSS + DNN (no attack) - DSS: embedding/addition, DNN: extraction
3. LPCNet + DSS + DNN (with attack) - DSS: embedding/addition, DNN: extraction
4. LPCNet + DSS + DNN (with attack) - DSS: embedding, DNN: addition/extraction

Based on timeline from thesis slides (pages 22-24)
"""

import argparse
import os

import lpcnet
import numpy as np
import tensorflow as tf

# Import LPCNet components
from dataloader import LPCNetLoader

# Import watermark models
from extractor.model import create_discriminator, create_watermark_extractor
from lossfuncs import (
    adversarial_discriminator_loss,
    adversarial_generator_loss,
    bit_error_rate_metric,
    perceptual_embedding_loss,
    signal_to_noise_ratio_metric,
    watermark_embedding_loss,
    watermark_extraction_loss,
)
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def parse_args():
    parser = argparse.ArgumentParser(description="Joint Watermark Training")

    # Data arguments
    parser.add_argument("features", help="binary features file (float32)")
    parser.add_argument("data", help="binary audio data file (uint8)")
    parser.add_argument("periods", help="binary periods file")

    # Training arguments
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Training phase (1-4)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    # Model arguments
    parser.add_argument("--lpcnet-weights", help="pretrained LPCNet weights")
    parser.add_argument("--extractor-weights", help="pretrained extractor weights")
    parser.add_argument(
        "--discriminator-weights", help="pretrained discriminator weights"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        default="checkpoints/watermark",
        help="output directory for models",
    )
    parser.add_argument(
        "--log-dir", default="logs/watermark", help="tensorboard log directory"
    )

    # Loss weights
    parser.add_argument(
        "--alpha-extract", type=float, default=1.0, help="extraction loss weight"
    )
    parser.add_argument(
        "--alpha-embed", type=float, default=0.1, help="embedding loss weight"
    )
    parser.add_argument(
        "--alpha-adv", type=float, default=0.01, help="adversarial loss weight"
    )
    
    # Perceptual loss weights for Phase 1
    parser.add_argument(
        "--alpha-mse", type=float, default=1.0, help="MSE loss weight in perceptual loss"
    )
    parser.add_argument(
        "--alpha-spectral", type=float, default=0.5, help="spectral loss weight"
    )
    parser.add_argument(
        "--alpha-temporal", type=float, default=0.3, help="temporal loss weight"
    )

    return parser.parse_args()


def load_data(features_file, data_file, periods_file, batch_size):
    """Load and prepare training data"""

    # Load memmap data (similar to train_lpcnet.py)
    data = np.memmap(data_file, dtype="int16", mode="r")
    features = np.memmap(features_file, dtype="float32", mode="r")
    periods = np.memmap(periods_file, dtype="float32", mode="r")

    # Reshape for LPCNetLoader
    frame_size = 160
    feature_chunk_size = 15
    pcm_chunk_size = frame_size * feature_chunk_size  # 2400
    nb_features = 36  # 20 + 16 LPC coeffs

    nb_frames = (len(data) // (2 * pcm_chunk_size) - 1) // batch_size * batch_size

    # Reshape data
    data = data[: nb_frames * 2 * pcm_chunk_size]
    data = np.reshape(data, (nb_frames, pcm_chunk_size, 2))

    # Reshape features
    sizeof = features.strides[-1]
    features = np.lib.stride_tricks.as_strided(
        features,
        shape=(nb_frames, feature_chunk_size + 4, nb_features),
        strides=(
            feature_chunk_size * nb_features * sizeof,
            nb_features * sizeof,
            sizeof,
        ),
    )

    # Reshape periods
    periods = np.reshape(
        periods[: nb_frames * feature_chunk_size], (nb_frames, feature_chunk_size, 1)
    )

    print(
        f"Data shapes: data={data.shape}, features={features.shape}, periods={periods.shape}"
    )

    # Create data loader with watermark bits
    loader = LPCNetLoader(
        data=data,
        features=features,
        periods=periods,
        batch_size=batch_size,
        bits_in=None,  # Will generate random bits
        e2e=False,
    )

    return loader


def create_models(phase, args):
    """Create models based on training phase"""

    models = {}

    # Always create LPCNet
    lpcnet_model, encoder, decoder = lpcnet.new_lpcnet_model(
        rnn_units1=384,
        rnn_units2=16,
        nb_used_features=20,
        batch_size=args.batch_size,
        training=True,
        adaptation=False,
        quantize=False,
        flag_e2e=False,
    )

    if args.lpcnet_weights:
        lpcnet_model.load_weights(args.lpcnet_weights)
        print(f"Loaded LPCNet weights: {args.lpcnet_weights}")

    models["lpcnet"] = lpcnet_model

    # Add extractor for phases 2+
    if phase >= 2:
        extractor = create_watermark_extractor(
            input_length=2400,  # 15 frames * 160
            bits_per_frame=64,
            num_frames=15,
        )

        if args.extractor_weights:
            extractor.load_weights(args.extractor_weights)
            print(f"Loaded extractor weights: {args.extractor_weights}")

        models["extractor"] = extractor

    # Add discriminator for phases 3+
    if phase >= 3:
        discriminator = create_discriminator(input_length=2400)

        if args.discriminator_weights:
            discriminator.load_weights(args.discriminator_weights)
            print(f"Loaded discriminator weights: {args.discriminator_weights}")

        models["discriminator"] = discriminator

    return models


def compile_models(models, phase, args):
    """Compile models with appropriate loss functions"""

    # Phase 1: LPCNet + DSS (no attack)  
    if phase == 1:
        models["lpcnet"].compile(
            optimizer=Adam(args.lr),
            loss=[
                None,  # m_out: no loss (keep LPCNet synthesis frozen)
                None,  # residual_w: no loss for now
                perceptual_embedding_loss(
                    alpha_mse=args.alpha_mse,         # Time-domain similarity
                    alpha_spectral=args.alpha_spectral, # Spectral consistency  
                    alpha_temporal=args.alpha_temporal  # Temporal structure
                ),  # pcm_w: enhanced perceptual loss for addition network
            ],
            metrics=[signal_to_noise_ratio_metric],
        )

    # Phase 2: + DNN extraction (no attack)
    elif phase == 2:
        models["lpcnet"].compile(
            optimizer=Adam(args.lr),
            loss=[None, None, perceptual_embedding_loss(
                alpha_mse=args.alpha_mse,
                alpha_spectral=args.alpha_spectral, 
                alpha_temporal=args.alpha_temporal
            )],
            metrics=[signal_to_noise_ratio_metric],
        )
        models["extractor"].compile(
            optimizer=Adam(args.lr),
            loss=watermark_extraction_loss(),
            metrics=[bit_error_rate_metric],
        )

    # Phase 3: + adversarial training (with attack)
    elif phase == 3:
        models["lpcnet"].compile(
            optimizer=Adam(args.lr),
            loss=[None, None, watermark_embedding_loss()],
            metrics=[signal_to_noise_ratio_metric],
        )
        models["extractor"].compile(
            optimizer=Adam(args.lr),
            loss=watermark_extraction_loss(),
            metrics=[bit_error_rate_metric],
        )
        models["discriminator"].compile(
            optimizer=Adam(args.lr * 0.1),  # Slower learning for discriminator
            loss=adversarial_discriminator_loss(),
            metrics=["accuracy"],
        )

    # Phase 4: Full joint training
    elif phase == 4:
        # Create combined model for end-to-end training
        # This would require more complex model architecture changes
        pass


def train_phase_1(models, loader, args):
    """Phase 1: LPCNet + DSS embedding and addition (no attack)"""
    print("=== Phase 1: LPCNet + DSS (no attack) ===")

    lpcnet_model = models["lpcnet"]

    # Setup callbacks
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(
        f"{args.output_dir}/lpcnet_phase1_{{epoch:02d}}.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor="loss",
    )

    csv_logger = CSVLogger(f"{args.output_dir}/phase1_training.log")

    # Train LPCNet with watermark embedding
    history = lpcnet_model.fit(
        loader, epochs=args.epochs, callbacks=[checkpoint, csv_logger], verbose=1
    )

    # Save final model
    lpcnet_model.save_weights(f"{args.output_dir}/lpcnet_phase1_final.h5")
    return history


def train_phase_2(models, loader, args):
    """Phase 2: + DNN extraction (no attack)"""
    print("=== Phase 2: LPCNet + DSS + DNN extraction (no attack) ===")

    lpcnet_model = models["lpcnet"]
    extractor = models["extractor"]

    # Freeze LPCNet embedding layers (keep synthesis trainable)
    for layer in lpcnet_model.layers:
        if "wm_" in layer.name:  # Freeze watermark layers
            layer.trainable = False

    # Train extractor on watermarked speech
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        epoch_extract_loss = 0
        epoch_ber = 0
        num_batches = 0

        for batch_idx, (inputs, outputs) in enumerate(loader):
            pcm, feat, pitch, bits_in, lpc = inputs

            # Generate watermarked speech using LPCNet
            lpcnet_outputs = lpcnet_model.predict_on_batch(inputs)
            m_out, residual_w, pcm_w = lpcnet_outputs

            # Train extractor on watermarked speech
            # Reshape bits for extractor (flatten frame dimension)
            bits_flat = tf.reshape(bits_in, (args.batch_size, -1))  # (B, 15*64)

            extractor_loss = extractor.train_on_batch(pcm_w, bits_flat)

            epoch_extract_loss += (
                extractor_loss[0]
                if isinstance(extractor_loss, list)
                else extractor_loss
            )
            if isinstance(extractor_loss, list) and len(extractor_loss) > 1:
                epoch_ber += extractor_loss[1]

            num_batches += 1

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Extract Loss={extractor_loss}")

        print(
            f"  Epoch {epoch + 1} - Extract Loss: {epoch_extract_loss / num_batches:.4f}, BER: {epoch_ber / num_batches:.4f}"
        )

    # Save models
    extractor.save_weights(f"{args.output_dir}/extractor_phase2_final.h5")
    lpcnet_model.save_weights(f"{args.output_dir}/lpcnet_phase2_final.h5")


def train_phase_3(models, loader, args):
    """Phase 3: + adversarial training (with attack)"""
    print("=== Phase 3: LPCNet + DSS + DNN + Adversarial (with attack) ===")

    lpcnet_model = models["lpcnet"]
    extractor = models["extractor"]
    discriminator = models["discriminator"]

    # Add attack simulation (noise, compression, etc.)
    def apply_attacks(audio):
        # Gaussian noise attack
        noise = tf.random.normal(tf.shape(audio), stddev=0.01)
        return audio + noise

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, (inputs, outputs) in enumerate(loader):
            pcm, feat, pitch, bits_in, lpc = inputs

            # === Train Discriminator ===
            # Generate watermarked speech
            lpcnet_outputs = lpcnet_model.predict_on_batch(inputs)
            m_out, residual_w, pcm_w = lpcnet_outputs

            # Apply attacks to watermarked speech
            pcm_w_attacked = apply_attacks(pcm_w)

            # Create discriminator training data
            real_audio = pcm
            fake_audio = pcm_w_attacked

            # Labels: 0 = real, 1 = watermarked
            real_labels = tf.zeros((args.batch_size, 1))
            fake_labels = tf.ones((args.batch_size, 1))

            # Train discriminator
            d_loss_real = discriminator.train_on_batch(real_audio, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_audio, fake_labels)

            # === Train Extractor ===
            bits_flat = tf.reshape(bits_in, (args.batch_size, -1))
            extractor_loss = extractor.train_on_batch(pcm_w_attacked, bits_flat)

            if batch_idx % 10 == 0:
                print(
                    f"  Batch {batch_idx}: D_real={d_loss_real}, D_fake={d_loss_fake}, Extract={extractor_loss}"
                )

    # Save models
    discriminator.save_weights(f"{args.output_dir}/discriminator_phase3_final.h5")
    extractor.save_weights(f"{args.output_dir}/extractor_phase3_final.h5")
    lpcnet_model.save_weights(f"{args.output_dir}/lpcnet_phase3_final.h5")


def main():
    args = parse_args()

    print(f"Starting Phase {args.phase} training...")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}, LR: {args.lr}")

    # Load data
    loader = load_data(args.features, args.data, args.periods, args.batch_size)
    print(f"Data loaded: {len(loader)} batches")

    # Create models
    models = create_models(args.phase, args)
    compile_models(models, args.phase, args)

    # Print model summaries
    for name, model in models.items():
        print(f"\n{name.upper()} Model:")
        model.summary(line_length=100)

    # Run training based on phase
    if args.phase == 1:
        train_phase_1(models, loader, args)
    elif args.phase == 2:
        train_phase_2(models, loader, args)
    elif args.phase == 3:
        train_phase_3(models, loader, args)
    elif args.phase == 4:
        print("Phase 4 not yet implemented - requires model architecture changes")

    print(f"Phase {args.phase} training completed!")


if __name__ == "__main__":
    main()

