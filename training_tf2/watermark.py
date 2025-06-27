import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Activation, Concatenate, Conv1D, Layer


class WatermarkEmbedding(Layer):
    def __init__(
        self,
        frame_size=160,
        bits_per_frame=64,
        alpha_init=0.05,
        trainable_alpha=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.frame_size = frame_size
        self.bpf = bits_per_frame
        self.alpha_init = alpha_init
        self.trainable_alpha = trainable_alpha

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
        super().build(input_shape)

    def call(self, inputs):
        wm_bpf, residual = inputs  # unpack
        wm_bpf = tf.cast(wm_bpf * 2 - 1, tf.float32)  # (B,T,BPF) -> {-1,+1}

        # Element-wise multiplication with broadcasting (B,T,64) * (B,T,1) = (B,T,64)
        wm_per_bit = wm_bpf * residual

        # Sum across bits and apply alpha scaling
        wm_single = self.alpha * tf.reduce_sum(
            wm_per_bit, axis=-1, keepdims=True
        )  # (B,T,1)

        return wm_single

    def get_config(self):
        base = super().get_config()
        base.update(
            dict(
                frame_size=self.frame_size,
                bpf=self.bits_per_frame,
                alpha_init=self.alpha_init,
                trainable_alpha=self.trainable_alpha,
            )
        )
        return base


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
