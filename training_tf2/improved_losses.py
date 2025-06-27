#!/usr/bin/python3
"""
Improved loss functions for LPCNet+Watermark system
Implements Phase 1 high-priority losses for better audio quality and watermark imperceptibility
"""

import tensorflow as tf
import numpy as np


def multi_scale_stft_loss(y_true, y_pred, fft_sizes=[512, 1024, 2048]):
    """
    Multi-scale STFT loss for perceptual audio quality

    Args:
        y_true: Ground truth audio (B, T, 1)
        y_pred: Predicted/watermarked audio (B, T, 1)
        fft_sizes: List of FFT sizes for multi-scale analysis

    Returns:
        Combined spectral magnitude and convergence loss
    """
    # Ensure float32 dtype and normalize to [-1, 1] range
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Simple normalization - always divide by max range
    # This avoids tf.cond gradient issues
    y_true = y_true / 32768.0
    y_pred = y_pred / 32768.0

    if len(y_true.shape) > 2:
        y_true = y_true[..., 0]  # Remove channel dimension
    if len(y_pred.shape) > 2:
        y_pred = y_pred[..., 0]

    total_loss = 0.0

    for fft_size in fft_sizes:
        hop_size = fft_size // 4

        # Compute STFT for both signals
        stft_true = tf.signal.stft(y_true, fft_size, hop_size, pad_end=True)
        stft_pred = tf.signal.stft(y_pred, fft_size, hop_size, pad_end=True)

        # Magnitude spectra
        mag_true = tf.abs(stft_true)
        mag_pred = tf.abs(stft_pred)

        # Spectral magnitude loss (L1)
        mag_loss = tf.reduce_mean(tf.abs(mag_true - mag_pred))

        # Spectral convergence loss (normalized L1)
        convergence_loss = tf.reduce_mean(
            tf.abs(mag_true - mag_pred) / (mag_true + 1e-7)
        )

        total_loss += mag_loss + convergence_loss

    return total_loss / len(fft_sizes)


def mel_spectrogram_loss(y_true, y_pred, sample_rate=16000, n_mels=80):
    """
    Mel-spectrogram perceptual loss

    Args:
        y_true: Ground truth audio (B, T, 1)
        y_pred: Predicted/watermarked audio (B, T, 1)
        sample_rate: Audio sample rate
        n_mels: Number of mel bands

    Returns:
        Log mel-spectrogram L1 loss
    """
    # Ensure float32 dtype and normalize to [-1, 1] range
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Simple normalization - always divide by max range
    # This avoids tf.cond gradient issues
    y_true = y_true / 32768.0
    y_pred = y_pred / 32768.0

    if len(y_true.shape) > 2:
        y_true = y_true[..., 0]
    if len(y_pred.shape) > 2:
        y_pred = y_pred[..., 0]

    # STFT parameters
    fft_size = 1024
    hop_size = 256

    # Compute STFT
    stft_true = tf.signal.stft(y_true, fft_size, hop_size, pad_end=True)
    stft_pred = tf.signal.stft(y_pred, fft_size, hop_size, pad_end=True)

    # Power spectra
    power_true = tf.abs(stft_true) ** 2
    power_pred = tf.abs(stft_pred) ** 2

    # Mel filter bank
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=fft_size // 2 + 1,
        sample_rate=sample_rate,
        lower_edge_hertz=50.0,
        upper_edge_hertz=sample_rate / 2.0,
    )

    # Apply mel filter bank
    mel_true = tf.tensordot(power_true, mel_matrix, 1)
    mel_pred = tf.tensordot(power_pred, mel_matrix, 1)

    # Log mel-spectrogram loss
    log_mel_true = tf.math.log(mel_true + 1e-7)
    log_mel_pred = tf.math.log(mel_pred + 1e-7)

    return tf.reduce_mean(tf.abs(log_mel_true - log_mel_pred))


def snr_imperceptibility_loss(y_clean, y_watermarked, target_snr_db=25.0):
    """
    SNR-based imperceptibility loss to ensure watermark remains inaudible

    Args:
        y_clean: Clean audio signal (B, T, 1)
        y_watermarked: Watermarked audio signal (B, T, 1)
        target_snr_db: Target SNR in dB

    Returns:
        SNR penalty loss (0 if SNR >= target, positive penalty otherwise)
    """
    # Ensure float32 dtype and normalize to [-1, 1] range
    y_clean = tf.cast(y_clean, tf.float32)
    y_watermarked = tf.cast(y_watermarked, tf.float32)

    # Simple normalization - always divide by max range
    # This avoids tf.cond gradient issues
    y_clean = y_clean / 32768.0
    y_watermarked = y_watermarked / 32768.0

    # Calculate watermark noise
    noise = y_watermarked - y_clean

    # Calculate power
    signal_power = tf.reduce_mean(tf.square(y_clean), axis=-2, keepdims=True)
    noise_power = tf.reduce_mean(tf.square(noise), axis=-2, keepdims=True)

    # SNR in dB
    snr_db = (
        10.0 * tf.math.log(signal_power / (noise_power + 1e-10)) / tf.math.log(10.0)
    )

    # Penalty for low SNR
    snr_penalty = tf.maximum(0.0, target_snr_db - snr_db)

    return tf.reduce_mean(snr_penalty)


def psychoacoustic_masking_loss(y_clean, y_watermarked):
    """
    Simplified psychoacoustic masking loss
    Ensures watermark modifications stay below auditory masking threshold

    Args:
        y_clean: Clean audio signal (B, T, 1)
        y_watermarked: Watermarked audio signal (B, T, 1)

    Returns:
        Masking threshold violation penalty
    """
    # Ensure float32 dtype and normalize to [-1, 1] range
    y_clean = tf.cast(y_clean, tf.float32)
    y_watermarked = tf.cast(y_watermarked, tf.float32)

    # Simple normalization - always divide by max range
    # This avoids tf.cond gradient issues
    y_clean = y_clean / 32768.0
    y_watermarked = y_watermarked / 32768.0

    if len(y_clean.shape) > 2:
        y_clean = y_clean[..., 0]
    if len(y_watermarked.shape) > 2:
        y_watermarked = y_watermarked[..., 0]

    # STFT analysis
    fft_size = 1024
    hop_size = 256

    stft_clean = tf.signal.stft(y_clean, fft_size, hop_size, pad_end=True)
    stft_watermarked = tf.signal.stft(y_watermarked, fft_size, hop_size, pad_end=True)

    # Magnitude spectra
    mag_clean = tf.abs(stft_clean)
    mag_watermarked = tf.abs(stft_watermarked)

    # Simplified masking threshold: fraction of clean signal magnitude
    masking_threshold = 0.1 * mag_clean

    # Watermark-induced spectral changes
    watermark_magnitude = tf.abs(mag_watermarked - mag_clean)

    # Penalty for exceeding masking threshold
    masking_violation = tf.maximum(0.0, watermark_magnitude - masking_threshold)

    return tf.reduce_mean(masking_violation)


def residual_magnitude_loss(residual_clean, residual_watermarked, max_ratio=2.0):
    """
    Residual magnitude preservation loss
    Prevents over-modification of LPC residual signal

    Args:
        residual_clean: Original LPC residual (B, T, 1)
        residual_watermarked: Watermarked LPC residual (B, T, 1)
        max_ratio: Maximum allowed magnitude ratio

    Returns:
        Combined magnitude preservation loss
    """
    # Ensure float32 dtype
    residual_clean = tf.cast(residual_clean, tf.float32)
    residual_watermarked = tf.cast(residual_watermarked, tf.float32)

    # Magnitude comparison
    mag_clean = tf.abs(residual_clean)
    mag_watermarked = tf.abs(residual_watermarked)

    # L2 loss on magnitude difference
    magnitude_loss = tf.reduce_mean(tf.square(mag_clean - mag_watermarked))

    # Penalty for excessive magnitude changes
    magnitude_ratio = mag_watermarked / (mag_clean + 1e-7)
    ratio_penalty = tf.reduce_mean(tf.maximum(0.0, magnitude_ratio - max_ratio))

    # Energy preservation constraint
    energy_clean = tf.reduce_mean(tf.square(residual_clean))
    energy_watermarked = tf.reduce_mean(tf.square(residual_watermarked))
    energy_loss = tf.abs(energy_clean - energy_watermarked) / (energy_clean + 1e-7)

    return magnitude_loss + 0.1 * ratio_penalty + 0.05 * energy_loss


def adversarial_robustness_loss(
    watermarked_audio, bits_true, extractor_model, noise_std=0.01
):
    """
    Adversarial robustness loss for watermark extraction
    Trains watermark to survive common perturbations

    Args:
        watermarked_audio: Watermarked audio signal (B, T, 1)
        bits_true: Ground truth watermark bits (B, 64)
        extractor_model: Watermark extractor model
        noise_std: Standard deviation of adversarial noise

    Returns:
        Binary cross-entropy loss on noisy extraction
    """
    # Ensure float32 dtype
    watermarked_audio = tf.cast(watermarked_audio, tf.float32)

    # Add Gaussian noise attack
    noise = tf.random.normal(tf.shape(watermarked_audio), stddev=noise_std)
    attacked_audio = watermarked_audio + noise

    # Extract watermark from attacked audio
    bits_extracted = extractor_model(attacked_audio, training=False)

    # Binary cross-entropy loss for robustness
    return tf.keras.losses.binary_crossentropy(bits_true, bits_extracted)


# Composite loss functions for easy integration
def composite_pcm_loss(y_true, y_pred):
    """
    Composite loss for watermarked PCM signal combining:
    - Multi-scale STFT loss (perceptual quality)
    - SNR imperceptibility loss (inaudibility)
    - Psychoacoustic masking loss (perceptual masking)

    Args:
        y_true: Clean PCM target (B, T, 1)
        y_pred: Watermarked PCM prediction (B, T, 1)
    """
    stft_loss = multi_scale_stft_loss(y_true, y_pred)
    snr_loss = snr_imperceptibility_loss(y_true, y_pred)
    masking_loss = psychoacoustic_masking_loss(y_true, y_pred)

    # Adjusted weights for normalized scale
    return 1.0 * stft_loss + 0.1 * snr_loss + 1.0 * masking_loss


def composite_residual_loss(y_true, y_pred):
    """
    Simplified residual regularization loss
    Prevents excessive residual modifications without needing clean residual target

    Args:
        y_true: Placeholder target (ignored)
        y_pred: Watermarked residual prediction (B, T, 1)
    """
    # Simple regularization: penalize very large residual values
    y_pred = tf.cast(y_pred, tf.float32)

    # Simple normalization - always divide by max range
    # This avoids tf.cond gradient issues
    y_pred = y_pred / 32768.0

    # L2 regularization on residual magnitude
    magnitude_penalty = tf.reduce_mean(tf.square(y_pred))

    # Penalty for extreme values
    extreme_penalty = tf.reduce_mean(tf.maximum(0.0, tf.abs(y_pred) - 0.1))

    return 0.01 * magnitude_penalty + 0.1 * extreme_penalty


def composite_bits_loss(bits_true, bits_pred, watermarked_audio, extractor_model):
    """
    Composite loss for watermark bit extraction combining:
    - Standard binary cross-entropy
    - Adversarial robustness loss
    """
    bce_loss = tf.keras.losses.binary_crossentropy(bits_true, bits_pred)
    adv_loss = adversarial_robustness_loss(
        watermarked_audio, bits_true, extractor_model
    )

    # Weighted combination
    return bce_loss + 0.15 * adv_loss
