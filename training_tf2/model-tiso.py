# Watermark Extractor Model
# Extracts watermark bits from watermarked speech (pcm_w)
# Based on TimbreWatermarking architecture adapted for LPCNet
# """

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Reshape,
)
from tensorflow.keras.models import Model


def create_watermark_extractor(
    input_length=2400,  # 15 frames * 160 samples
    bits_per_frame=64,
    num_frames=15,
    filters=[32, 64, 128],
    kernel_size=9,
    dropout_rate=0.3,
):
    """
    Create watermark extractor model

    Args:
        input_length: Length of input audio chunk (watermarked speech)
        bits_per_frame: Number of bits per frame (64)
        num_frames: Number of frames in chunk (15)
        filters: CNN filter sizes
        kernel_size: Convolution kernel size
        dropout_rate: Dropout rate

    Returns:
        extractor_model: Keras model
    """

    # Input: watermarked speech pcm_w
    audio_input = Input(shape=(input_length, 1), name="watermarked_audio")

    x = audio_input

    # Multi-scale CNN feature extraction
    for i, num_filters in enumerate(filters):
        x = Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="same",
            name=f"conv1d_{i + 1}",
        )(x)
        x = BatchNormalization(name=f"bn_{i + 1}")(x)
        x = Activation("relu", name=f"relu_{i + 1}")(x)
        x = Dropout(dropout_rate, name=f"dropout_{i + 1}")(x)

    # Global feature extraction
    global_features = GlobalAveragePooling1D(name="global_pool")(x)

    # Frame-wise feature extraction for per-frame bit prediction
    # Reshape to frame-based processing
    frame_features = Reshape((num_frames, input_length // num_frames, filters[-1]))(x)
    frame_features = tf.reduce_mean(
        frame_features, axis=2
    )  # Average over frame samples

    # Per-frame bit prediction
    frame_dense = Dense(128, activation="relu", name="frame_dense1")(frame_features)
    frame_dense = Dropout(dropout_rate, name="frame_dropout")(frame_dense)
    frame_dense = Dense(bits_per_frame, activation="sigmoid", name="frame_bits")(
        frame_dense
    )

    # Flatten to match expected output shape (batch, total_bits)
    extracted_bits = Reshape((num_frames * bits_per_frame,), name="extracted_bits")(
        frame_dense
    )

    extractor = Model(
        inputs=audio_input, outputs=extracted_bits, name="watermark_extractor"
    )

    return extractor


def create_discriminator(
    input_length=2400, filters=[64, 128, 256], kernel_size=9, dropout_rate=0.5
):
    """
    Create discriminator to distinguish real vs watermarked speech

    Args:
        input_length: Length of input audio
        filters: CNN filter sizes
        kernel_size: Convolution kernel size
        dropout_rate: Dropout rate

    Returns:
        discriminator_model: Keras model
    """

    audio_input = Input(shape=(input_length, 1), name="audio_input")

    x = audio_input

    # CNN feature extraction
    for i, num_filters in enumerate(filters):
        x = Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            strides=2,  # Downsample
            padding="same",
            name=f"disc_conv_{i + 1}",
        )(x)
        x = BatchNormalization(name=f"disc_bn_{i + 1}")(x)
        x = Activation("leaky_relu", name=f"disc_leaky_{i + 1}")(x)
        x = Dropout(dropout_rate, name=f"disc_dropout_{i + 1}")(x)

    # Global pooling and classification
    x = GlobalAveragePooling1D(name="disc_global_pool")(x)
    x = Dense(128, activation="relu", name="disc_dense1")(x)
    x = Dropout(dropout_rate, name="disc_final_dropout")(x)

    # Binary classification: real (0) vs watermarked (1)
    output = Dense(1, activation="sigmoid", name="disc_output")(x)

    discriminator = Model(inputs=audio_input, outputs=output, name="discriminator")

    return discriminator


if __name__ == "__main__":
    # Test the models
    extractor = create_watermark_extractor()
    discriminator = create_discriminator()

    print("Watermark Extractor:")
    extractor.summary()

    print("\nDiscriminator:")
    discriminator.summary()

    # Test forward pass
    dummy_audio = tf.random.normal((2, 2400, 1))  # batch_size=2

    extracted = extractor(dummy_audio)
    disc_output = discriminator(dummy_audio)

    print(
        f"\nExtracted bits shape: {extracted.shape}"
    )  # Should be (2, 960) = 2 * 15 * 64
    print(f"Discriminator output shape: {disc_output.shape}")  # Should be (2, 1)