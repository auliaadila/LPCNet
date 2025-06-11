import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, Concatenate, Activation
from tensorflow.keras import initializers


class WatermarkEmbedding(Layer):
    """
    Direct-Spread watermark layer.

    Inputs
    ------
    msg_bits  : int32  shape = (B, F, bits_per_frame)
                values 0 / 1  – one payload bit per frame slot
    residual  : float32 shape = (B, T, 1)
                LPC residual / excitation samples

    Parameters
    ----------
    frame_size       : samples per conditioning frame   (default 160)
    bits_per_frame   : number of bits embedded per frame
    alpha_init       : initial global strength
    trainable_alpha  : if True, alpha becomes a learnable scalar

    Output
    ------
    residual_marked  : watermarked residual, same shape as `residual`
    """
    def __init__(self,
                 frame_size     = 160,
                 bits_per_frame = 64,
                 alpha_init     = 0.05,
                 trainable_alpha= False,
                 **kwargs):
        super().__init__(**kwargs)
        self.frame_size      = frame_size
        self.bits_per_frame  = bits_per_frame
        self.alpha_init      = alpha_init
        self.trainable_alpha = trainable_alpha

    def build(self, input_shape):
        if self.trainable_alpha:
            self.alpha = self.add_weight(name='alpha',
                                         shape=(),
                                         initializer=tf.constant_initializer(
                                             self.alpha_init),
                                         trainable=True)
        else:
            self.alpha = tf.constant(self.alpha_init, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        msg_bits, residual = inputs            # unpack

        # squeeze channel dim if present (B,T,1) -> (B,T)
        if residual.shape.rank == 3:
            residual = tf.squeeze(residual, axis=-1)        # (B,T)

        B = tf.shape(residual)[0]
        T = tf.shape(residual)[1]
        F = tf.shape(msg_bits)[1]

        # --- safety check (dynamic): F*frame_size must equal T -------------
        with tf.control_dependencies(
            [tf.debugging.assert_equal(F * self.frame_size, T,
             message="msg_bits length × frame_size must equal residual length")]):
            residual = tf.identity(residual)

        # 0/1  ->  ±1
        bits_pm = tf.cast(msg_bits * 2 - 1, tf.float32)      # (B,F,BPF)

        # repeat each bit across its chip region
        chip_len   = self.frame_size // self.bits_per_frame  # integer part
        extra      = self.frame_size - chip_len * self.bits_per_frame

        # build a repeat pattern that fills the 160 samples even if not exact
        rep_pattern = ([chip_len + 1] * extra) + ([chip_len] *
                                                  (self.bits_per_frame - extra))
        rep_pattern = tf.constant(rep_pattern, dtype=tf.int32)    # (BPF,)

        bits_spread = tf.repeat(bits_pm,
                                repeats=rep_pattern,
                                axis=-1)                         # (B,F,160)

        bits_spread = tf.reshape(bits_spread, (B, T))            # (B,T)

        # --- watermark: alpha * bit * residual -----------------------------
        wm = self.alpha * bits_spread * residual                 # (B,T)
        # residual_mark = residual + wm                            # add

        return tf.expand_dims(wm, -1)                 # (B,T,1)
        # return tf.expand_dims(residual_mark, -1)                 # (B,T,1)

    def get_config(self):
        base = super().get_config()
        base.update(dict(frame_size=self.frame_size,
                         bits_per_frame=self.bits_per_frame,
                         alpha_init=self.alpha_init,
                         trainable_alpha=self.trainable_alpha))
        return base


class WatermarkAddition(Layer):
    """
    Additive watermark combiner.

    Inputs
    ------
      host : float32  (B, T, 1)   – original / clean waveform
      wm   : float32  (B, T, 1)   – watermark component you already built

    Parameters
    ----------
      learnable_mask : bool        – if True, learn g(n) with a small CNN
      beta           : float        – |g(n)| ≤ beta  (mask dynamic range)
      filters        : int          – conv channels for the mixer CNN

    Output
    ------
      marked : float32 (B, T, 1)   – host + g(n)*wm
    """

    def __init__(self, learnable_mask=True, beta=0.1, filters=32, **kw):
        super().__init__(**kw)
        self.learnable_mask = learnable_mask
        self.beta = beta
        self.filters = filters
        if learnable_mask:
            self.conv1 = Conv1D(filters, 9, padding='same', activation='relu')
            self.conv2 = Conv1D(filters, 9, padding='same', activation='relu')
            self.conv3 = Conv1D(1, 9, padding='same', activation='tanh')  # → (-1,+1)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        host, wm = inputs                   # expect (B,T,1) each

        if not self.learnable_mask:
            return host + wm               # simple addition

        # ----- learn g(n) ---------------------------------------------------
        x = Concatenate()([host, wm])       # (B,T,2)
        x = self.conv1(x)
        x = self.conv2(x)
        g = self.beta * self.conv3(x)       # scale tanh to (-β, β)
        # Learnable “mixer” mask – a tiny 1-D CNN learns a gain envelope g(n)\in[-\beta,+\beta] so the network can hide the watermark where the ear is least sensitive

        return host + g * wm

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(learnable_mask=self.learnable_mask,
                        beta=self.beta,
                        filters=self.filters))
        return cfg