import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Activation, Concatenate, Conv1D, Layer


class WatermarkEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        frame_size=160,
        bits_per_frame=64,
        alpha_init=0.05,
        trainable_alpha=False,
        learnable_carriers=True,          # NEW
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frame_size = frame_size
        self.bpf = bits_per_frame
        self.alpha_init = alpha_init
        self.trainable_alpha = trainable_alpha
        self.learnable_carriers = learnable_carriers

    def build(self, input_shape):
        if self.trainable_alpha:
            self.alpha = self.add_weight(
                name="alpha",
                shape=(),
                initializer=tf.constant_initializer(self.alpha_init),
                trainable=True,
            )
        else:
            self.alpha = tf.constant(self.alpha_init, dtype=tf.float32)

        # one scalar carrier per bit  (shape = (1, 1, 64))
        init = tf.random_uniform_initializer(-1.0, 1.0)
        self.carriers = self.add_weight(
            "carriers",
            shape=(1, 1, self.bpf),
            initializer=init,
            trainable=self.learnable_carriers,
        )
        super().build(input_shape)

    def call(self, inputs):
        bits, residual = inputs                      # bits:(B,T,64) residual:(B,T,1)
        bits = tf.cast(bits * 2 - 1, tf.float32)     # 0/1 → ±1

        # Apply carrier gains (broadcast over time)
        # (B,T,64) * (1,1,64) → (B,T,64)
        spread = bits * self.carriers

        # Modulate host residual and sum
        wm = tf.reduce_sum(spread * residual, axis=-1, keepdims=True)

        # Normalise energy (~keep var constant) and scale by alpha
        wm = self.alpha / tf.math.sqrt(
            tf.cast(self.bpf, tf.float32)
        ) * wm
        return wm

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            dict(
                frame_size=self.frame_size,
                bits_per_frame=self.bpf,
                alpha_init=self.alpha_init,
                trainable_alpha=self.trainable_alpha,
                learnable_carriers=self.learnable_carriers,
            )
        )
        return cfg


class WatermarkAddition(Layer):
    def __init__(self, trainable_beta=False, beta_init=0.1, **kw):
        super().__init__(**kw)
        self.trainable_beta = trainable_beta
        self.beta_init = beta_init

    def build(self, input_shape):
        if self.trainable_beta:
            self.beta = self.add_weight(
                name="beta",
                shape=(),
                initializer=tf.constant_initializer(self.beta_init),
                trainable=True,
            )
        else:
            self.beta = tf.constant(self.beta_init, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        host, wm = inputs  # expect (B,T,1) each
        return host + wm  # simple addition

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            dict(
                trainable_beta=self.trainable_beta,
                beta_init=self.beta_init,
                beta=self.beta,
            )
        )
        return cfg
