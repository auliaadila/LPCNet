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
    # parser.add_argument("periods", help="binary periods file")

    # Training arguments
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


def load_data(features_file, data_file, batch_size):
    """Load and prepare training data"""

    # Load memmap data (similar to train_lpcnet.py)
    data = np.memmap(data_file, dtype="int16", mode="r")
    features = np.memmap(features_file, dtype="float32", mode="r")
    # periods = np.memmap(periods_file, dtype="float32", mode="r")
    

    

    # Reshape for LPCNetLoader
    frame_size = 160
    feature_chunk_size = 15
    pcm_chunk_size = frame_size * feature_chunk_size  # 2400
    nb_features = 36  # 20 + 16 LPC coeffs
    nb_used_features = 20
    lookahead = 2

    nb_frames = (len(data) // (2 * pcm_chunk_size) - 1) // batch_size * batch_size
    
    # Reshape data
    data = data[(4 - lookahead) * 2 * frame_size:]
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
    periods = (.1 + 50*features[:,:,nb_used_features-2:nb_used_features-1]+100).astype('int16')

    # # Reshape periods
    # periods = np.reshape(
    #     periods[: nb_frames * feature_chunk_size], (nb_frames, feature_chunk_size, 1)
    # )

    print(
        f"Data shapes: data={data.shape}, features={features.shape}, periods={periods.shape}"
    )

    # Create data loader with watermark bits
    loader = LPCNetLoader(
        data=data,
        features=features,
        periods=periods,
        batch_size=batch_size,
        # bits_in=None,  # Will generate random bits
        e2e=False,
    )

    return loader


def create_models(args):
    """Create models based on training phase"""

    models = {}

    import lpcnet_wm_v1 as lpcnet_wm
    lpcnet_wm_model, encoder, decoder = lpcnet_wm.new_lpcnet_wm_model(
        rnn_units1=384,
        rnn_units2=16,
        nb_used_features=20,
        batch_size=args.batch_size,
        training=True,
        adaptation=False,
        quantize=False,
        flag_e2e=False,
        bits_in=None
    )

    if args.lpcnet_weights:
        lpcnet_wm_model.load_weights(args.lpcnet_weights)
        print(f"Loaded LPCNet weights: {args.lpcnet_weights}")
    
    models["lpcnet"] = lpcnet_wm_model

    return models


def compile_models(models, phase, args):
    """Compile models with appropriate loss functions"""
    # Create combined model for end-to-end training
    # This would require more complex model architecture changes
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
        # pass

def train(models, loader, args):
    print("=== Joint training ===")

    lpcnet_model = models["lpcnet"]

    # Setup callbacks
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(
        f"{args.output_dir}/lpcnet_v1{{epoch:02d}}.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor="loss",
    )
    csv_logger = CSVLogger(f"{args.output_dir}/v1_training.log")

    # Train LPCNet with watermark embedding
    history = lpcnet_model.fit(
        loader, epochs=args.epochs, callbacks=[checkpoint, csv_logger], verbose=1
    )

    # Save final model
    lpcnet_model.save_weights(f"{args.output_dir}/lpcnet_v1_final.h5")
    return history

def main():
    args = parse_args()

    print(f"Starting training...")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}, LR: {args.lr}")
    
    # Load data
    # loader = load_data(args.features, args.data, args.periods, args.batch_size)
    # import IPython
    # IPython.embed()

    loader = load_data(args.features, args.data, args.batch_size)
    print(f"Data loaded: {len(loader)} batches")

    # Create models
    models = create_models(args)
    compile_models(models, args)

    # Print model summaries
    for name, model in models.items():
        print(f"\n{name.upper()} Model:")
        model.summary(line_length=100)

    # Run training based on phase
    train()
    # print("Phase 4 not yet implemented - requires model architecture changes")

    print(f"Training completed!")


if __name__ == "__main__":
    main()

