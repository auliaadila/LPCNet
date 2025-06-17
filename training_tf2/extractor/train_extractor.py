#!/usr/bin/env python3
"""
Standalone Watermark Extractor Training
Trains the watermark extractor on pre-generated watermarked speech
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from model import create_watermark_extractor
import sys
sys.path.append('..')
from lossfuncs import watermark_extraction_loss, bit_error_rate_metric

def parse_args():
    parser = argparse.ArgumentParser(description='Train Watermark Extractor')
    
    parser.add_argument('--watermarked-audio', required=True,
                       help='Path to watermarked audio data (.npy)')
    parser.add_argument('--watermark-bits', required=True, 
                       help='Path to watermark bits (.npy)')
    
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--validation-split', type=float, default=0.2, 
                       help='validation split ratio')
    
    parser.add_argument('--output', default='extractor_model.h5', 
                       help='output model path')
    parser.add_argument('--log-dir', default='logs/extractor',
                       help='tensorboard log directory')
    
    return parser.parse_args()


def load_data(audio_path, bits_path):
    """Load watermarked audio and corresponding bits"""
    
    # Load watermarked audio data  
    audio_data = np.load(audio_path)  # Shape: (N, 2400, 1)
    bits_data = np.load(bits_path)    # Shape: (N, 15, 64) or (N, 960)
    
    print(f"Audio data shape: {audio_data.shape}")
    print(f"Bits data shape: {bits_data.shape}")
    
    # Ensure correct shapes
    if len(audio_data.shape) == 2:
        audio_data = np.expand_dims(audio_data, -1)  # Add channel dim
    
    if len(bits_data.shape) == 3:
        # Flatten frame dimension: (N, 15, 64) -> (N, 960)
        bits_data = bits_data.reshape(bits_data.shape[0], -1)
    
    print(f"Final shapes - Audio: {audio_data.shape}, Bits: {bits_data.shape}")
    
    return audio_data, bits_data


def create_dataset(audio_data, bits_data, batch_size, validation_split=0.2):
    """Create tensorflow datasets"""
    
    # Split data
    n_samples = len(audio_data)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_val
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Create train dataset
    train_audio = audio_data[train_indices]
    train_bits = bits_data[train_indices]
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_audio, train_bits))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create validation dataset
    val_audio = audio_data[val_indices] 
    val_bits = bits_data[val_indices]
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_audio, val_bits))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    
    return train_dataset, val_dataset


def main():
    args = parse_args()
    
    print("Loading data...")
    audio_data, bits_data = load_data(args.watermarked_audio, args.watermark_bits)
    
    print("Creating datasets...")
    train_dataset, val_dataset = create_dataset(
        audio_data, bits_data, args.batch_size, args.validation_split
    )
    
    print("Creating extractor model...")
    extractor = create_watermark_extractor(
        input_length=audio_data.shape[1],  # Should be 2400
        bits_per_frame=64,
        num_frames=15
    )
    
    # Compile model
    extractor.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss=watermark_extraction_loss(),
        metrics=[bit_error_rate_metric, 'mae']
    )
    
    extractor.summary()
    
    # Setup callbacks
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            args.output,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        CSVLogger(f'{args.log_dir}/training.log'),
        EarlyStopping(
            monitor='val_bit_error_rate_metric',
            patience=10,
            mode='min',
            restore_best_weights=True
        )
    ]
    
    print("Starting training...")
    history = extractor.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate final model
    print("\nFinal evaluation:")
    val_loss, val_ber, val_mae = extractor.evaluate(val_dataset, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation BER: {val_ber:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")
    
    # Save training history
    np.save(f'{args.log_dir}/history.npy', history.history)
    
    print(f"Training completed! Model saved to: {args.output}")


if __name__ == '__main__':
    main()