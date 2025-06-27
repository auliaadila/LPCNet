#!/usr/bin/python3
"""Adversarial training layers for watermark robustness"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class AdversarialDistortion(Layer):
    """
    Adversarial distortion layer that applies various audio corruptions
    to make watermark extraction more robust.

    Applies multiple types of distortions:
    - Additive Gaussian noise
    - Uniform noise
    - Compression artifacts simulation
    - High-pass filtering effects
    - Amplitude scaling
    """

    def __init__(
        self,
        noise_std=0.05,
        uniform_std=0.03,
        compression_prob=0.3,
        filter_prob=0.2,
        scale_prob=0.2,
        scale_range=(0.7, 1.3),
        trainable_strength=True,
        **kwargs,
    ):
        super(AdversarialDistortion, self).__init__(**kwargs)

        self.noise_std = noise_std
        self.uniform_std = uniform_std
        self.compression_prob = compression_prob
        self.filter_prob = filter_prob
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.trainable_strength = trainable_strength

        if trainable_strength:
            self.strength = self.add_weight(
                name="distortion_strength",
                shape=(),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.strength = 1.0

    def call(self, inputs, training=None):
        if not training:
            return inputs

        batch_size = tf.shape(inputs)[0]

        # Apply distortions with probability
        distorted = inputs

        # 1. Additive Gaussian noise
        gaussian_mask = tf.random.uniform([batch_size, 1, 1]) < 0.5
        gaussian_noise = tf.random.normal(tf.shape(inputs), stddev=self.noise_std)
        distorted = tf.where(
            gaussian_mask, distorted + self.strength * gaussian_noise, distorted
        )

        # 2. Uniform noise
        uniform_mask = tf.random.uniform([batch_size, 1, 1]) < 0.3
        uniform_noise = tf.random.uniform(
            tf.shape(inputs), minval=-self.uniform_std, maxval=self.uniform_std
        )
        distorted = tf.where(
            uniform_mask, distorted + self.strength * uniform_noise, distorted
        )

        # 3. Compression simulation (quantization)
        compress_mask = tf.random.uniform([batch_size, 1, 1]) < self.compression_prob
        quantization_levels = tf.random.uniform(
            [batch_size, 1, 1], minval=64, maxval=256, dtype=tf.int32
        )
        quantization_levels = tf.cast(quantization_levels, tf.float32)
        compressed = tf.round(distorted * quantization_levels) / quantization_levels
        distorted = tf.where(compress_mask, compressed, distorted)

        # 4. High-pass filtering simulation
        filter_mask = tf.random.uniform([batch_size, 1, 1]) < self.filter_prob
        # Simple high-pass: subtract low-pass (moving average)
        kernel_size = 5
        kernel = tf.ones([kernel_size, 1, 1]) / kernel_size
        low_pass = tf.nn.conv1d(distorted, kernel, stride=1, padding="SAME")
        high_pass = distorted - 0.3 * low_pass
        distorted = tf.where(filter_mask, high_pass, distorted)

        # 5. Amplitude scaling
        scale_mask = tf.random.uniform([batch_size, 1, 1]) < self.scale_prob
        scale_factor = tf.random.uniform(
            [batch_size, 1, 1], minval=self.scale_range[0], maxval=self.scale_range[1]
        )
        scaled = distorted * scale_factor
        distorted = tf.where(scale_mask, scaled, distorted)

        # Clip to prevent overflow
        distorted = tf.clip_by_value(distorted, -1.0, 1.0)

        return distorted

    def get_config(self):
        config = {
            "noise_std": self.noise_std,
            "uniform_std": self.uniform_std,
            "compression_prob": self.compression_prob,
            "filter_prob": self.filter_prob,
            "scale_prob": self.scale_prob,
            "scale_range": self.scale_range,
            "trainable_strength": self.trainable_strength,
        }
        base_config = super(AdversarialDistortion, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdversarialTrainingCallback(tf.keras.callbacks.Callback):
    """
    Callback to manage adversarial training phases and distortion strength scheduling.

    Implements three-phase training:
    - Phase 1 (epochs 1-40): No adversarial attacks (strength=0.0)
    - Phase 2 (epochs 41-80): Gentle attacks (strength=0.3)
    - Phase 3 (epochs 81-120): Full attacks (strength=1.0)
    """

    def __init__(self, verbose=1):
        super(AdversarialTrainingCallback, self).__init__()
        self.verbose = verbose

        # Three-phase training schedule
        self.phase_schedule = {
            (1, 40): 0.0,  # Phase 1: No attacks
            (41, 80): 0.3,  # Phase 2: Gentle attacks
            (81, 120): 1.0,  # Phase 3: Full attacks
        }

    def on_epoch_begin(self, epoch, logs=None):
        # Determine current distortion strength (epoch is 0-indexed)
        current_epoch = epoch + 1
        strength = self.get_strength_for_epoch(current_epoch)

        # Update distortion layer strength if it exists
        try:
            distortion_layer = self.model.get_layer("adversarial_distortion")
            if hasattr(distortion_layer, "strength"):
                if hasattr(distortion_layer.strength, "assign"):
                    # Trainable variable
                    distortion_layer.strength.assign(strength)
                else:
                    # Regular attribute
                    distortion_layer.strength = strength

                if self.verbose:
                    phase = self.get_phase_name(current_epoch)
                    print(
                        f"Epoch {current_epoch}: {phase} - Adversarial strength = {strength:.1f}"
                    )
        except ValueError:
            # No distortion layer found (normal for non-adversarial training)
            if self.verbose and current_epoch == 1:
                print(
                    "No adversarial distortion layer found - running standard training"
                )

    def get_strength_for_epoch(self, epoch):
        """Get distortion strength for given epoch"""
        for (start, end), strength in self.phase_schedule.items():
            if start <= epoch <= end:
                return float(strength)
        return 1.0  # Default to full strength after phase 3

    def get_phase_name(self, epoch):
        """Get phase name for given epoch"""
        if 1 <= epoch <= 40:
            return "Phase 1 (Standard)"
        elif 41 <= epoch <= 80:
            return "Phase 2 (Gentle Adversarial)"
        elif 81 <= epoch <= 120:
            return "Phase 3 (Full Adversarial)"
        else:
            return "Phase 3+ (Full Adversarial)"


def adversarial_watermark_loss(y_true, y_pred):
    """
    Combined loss for adversarial watermark training.

    Args:
        y_true: Ground truth watermark bits (batch_size, 64)
        y_pred: Predicted watermark bits (batch_size, 64)

    Returns:
        Binary crossentropy loss for watermark extraction
    """
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)


def audio_quality_loss(y_true, y_pred):
    """
    L2 distance between clean and corrupted watermarked audio.

    Args:
        y_true: Clean watermarked audio (batch_size, time, 1)
        y_pred: Adversarially corrupted audio (batch_size, time, 1)

    Returns:
        Mean squared error between clean and corrupted audio
    """
    # Ensure both tensors are float32 to avoid type mismatch
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    return tf.reduce_mean(tf.square(y_true - y_pred))


class WatermarkRobustnessMetric(tf.keras.metrics.Metric):
    """
    Custom metric to track watermark extraction robustness.
    Computes bit-level accuracy for watermark extraction.
    """

    def __init__(self, name="watermark_robustness", **kwargs):
        super(WatermarkRobustnessMetric, self).__init__(name=name, **kwargs)
        self.total_bits = self.add_weight(name="total_bits", initializer="zeros")
        self.correct_bits = self.add_weight(name="correct_bits", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to binary (threshold at 0.5)
        y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        y_true_binary = tf.cast(y_true, tf.float32)

        # Count correct predictions
        correct = tf.cast(tf.equal(y_pred_binary, y_true_binary), tf.float32)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            correct = tf.multiply(correct, sample_weight)

        # Update counters
        self.correct_bits.assign_add(tf.reduce_sum(correct))
        self.total_bits.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.correct_bits, self.total_bits)

    def reset_state(self):
        self.total_bits.assign(0)
        self.correct_bits.assign(0)


def create_adversarial_model_with_distortion(
    base_model, noise_std=0.05, compression_prob=0.3, filter_prob=0.2, scale_prob=0.2
):
    """
    Utility function to add adversarial distortion to an existing model.

    Args:
        base_model: Existing LPCNet model
        noise_std: Standard deviation for Gaussian noise
        compression_prob: Probability of compression attack
        filter_prob: Probability of filtering attack
        scale_prob: Probability of amplitude scaling

    Returns:
        Modified model with adversarial distortion layer
    """

    # This is a helper function - actual integration happens in lpcnet.py
    # during model construction, not as a post-processing step

    raise NotImplementedError(
        "Adversarial distortion should be integrated during model construction "
        "in lpcnet.py, not as post-processing. See adversarial-plan.md for details."
    )


# Export key components
__all__ = [
    "AdversarialDistortion",
    "AdversarialTrainingCallback",
    "adversarial_watermark_loss",
    "audio_quality_loss",
    "WatermarkRobustnessMetric",
]
