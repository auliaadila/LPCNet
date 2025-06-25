import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, Concatenate
from tensorflow.keras import initializers

# -------------------------------------------------------------------- #
#  Watermark-Embedding  (α · bit · residual)                           #
# -------------------------------------------------------------------- #
class WatermarkEmbedding(Layer):
    """
    Direct-Spread watermark embedding.

    Inputs
    -------
      bits_bpf : int32 (B, F, BPF)   {0,1}  – 64 bits per 10-ms frame
      residual : float (B, T, 1)            – LPC residual (T = F*frame_size)

    Parameters
    ----------
      mode     : "fixed" | "global" | "frame" | "cnn"
      frame_size     : hop length (160)
      bits_per_frame : 64
      alpha_init     : initial gain
      alpha_max      : hard ceiling for |α|
    """
    def __init__(self,
                 mode="global",
                 frame_size=160,
                 bits_per_frame=64,
                 alpha_init=0.05,
                 alpha_max=0.2,
                 cnn_filters=16,
                 **kw):
        super().__init__(**kw)
        assert mode in ("fixed", "global", "frame", "cnn")
        self.mode  = mode
        self.Fs    = frame_size
        self.BPF   = bits_per_frame
        self.a0    = alpha_init
        self.a_max = alpha_max
        self.cnn_f = cnn_filters
        if mode == "cnn":
            self.conv1 = Conv1D(cnn_filters, 9, padding="same", activation="relu")
            self.conv2 = Conv1D(1, 9, padding="same", activation="tanh")

    # --------------------------------------------------------------
    def build(self, input_shape):
        if self.mode == "global":
            self.alpha = self.add_weight("alpha",
                                         shape=(),
                                         initializer=tf.constant_initializer(self.a0),
                                         trainable=True)
        elif self.mode == "frame":
            T = input_shape[1]        # may be None at build time
            if T is None: T = 2400    # fallback for static-graph build
            nF = int(np.ceil(T / self.Fs))
            self.alpha = self.add_weight("alpha_f",
                                         shape=(1, nF, 1),
                                         initializer=tf.constant_initializer(self.a0),
                                         trainable=True)
        super().build(input_shape)

    # --------------------------------------------------------------
    def _broadcast_alpha(self, residual):
        """Return α broadcast to residual.shape"""
        if self.mode == "fixed":
            return tf.constant(self.a0, residual.dtype)
        if self.mode == "global":
            return tf.clip_by_value(self.alpha, 0.0, self.a_max)
        if self.mode == "frame":
            # repeat each α_f over its 160 samples
            alpha_bc = tf.repeat(self.alpha, self.Fs, axis=1)
            return tf.clip_by_value(alpha_bc[:, :tf.shape(residual)[1]], 0.0, self.a_max)
        # mode == "cnn"
        x = self.conv1(residual)          # (B,T,filters)
        a = self.a_max * self.conv2(x)    # tanh→(-a_max,a_max)
        return a

    # --------------------------------------------------------------
    def call(self, inputs):
        bits, residual = inputs                    # residual (B,T,1)
        bits_pm  = tf.cast(bits * 2 - 1, tf.float32)           # {0,1}→{-1,+1}
        residual = residual * tf.ones_like(bits_pm)            # broadcast to (B,T,BPF)

        alpha_bc = self._broadcast_alpha(residual[..., :1])    # (B,T,1) or (B,T,BPF)
        wm_all   = alpha_bc * residual * bits_pm               # (B,T,BPF)
        wm_single= tf.reduce_sum(wm_all, axis=-1, keepdims=True)  # collapse 64→1

        return wm_single                                       # (B,T,1)

    # --------------------------------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(mode=self.mode, frame_size=self.Fs,
                        bits_per_frame=self.BPF, alpha_init=self.a0,
                        alpha_max=self.a_max, cnn_filters=self.cnn_f))
        return cfg

# -------------------------------------------------------------------- #
#  Watermark-Addition  (host + β · wm  or  host + g(n)·wm)             #
# -------------------------------------------------------------------- #
class WatermarkAddition(Layer):
    """
    Combines host signal and watermark.

    Parameters
    ----------
      mode        : "fixed" | "scalar" | "frame" | "cnn"
      beta_init   : starting gain
      beta_max    : clip limit for β
      frame_size  : 160  (for mode="frame")
      filters     : conv channels when mode="cnn"
    """
    def __init__(self,
                 mode="fixed",
                 beta_init=0.1,
                 beta_max=0.3,
                 frame_size=160,
                 filters=32,
                 **kw):
        super().__init__(**kw)
        assert mode in ("fixed", "scalar", "frame", "cnn")
        self.mode  = mode
        self.b0    = beta_init
        self.b_max = beta_max
        self.Fs    = frame_size
        self.filters = filters
        if mode == "cnn":
            self.c1 = Conv1D(filters, 9, padding="same", activation="relu")
            self.c2 = Conv1D(filters, 9, padding="same", activation="relu")
            self.c3 = Conv1D(1, 9, padding="same", activation="tanh")

    # --------------------------------------------------------------
    def build(self, input_shape):
        if self.mode == "scalar":
            self.beta = self.add_weight("beta",
                                        shape=(),
                                        initializer=tf.constant_initializer(self.b0),
                                        trainable=True)
        elif self.mode == "frame":
            T = input_shape[0][1] or 2400
            nF = int(np.ceil(T / self.Fs))
            self.beta = self.add_weight("beta_f",
                                        shape=(1, nF, 1),
                                        initializer=tf.constant_initializer(self.b0),
                                        trainable=True)
        super().build(input_shape)

    # --------------------------------------------------------------
    def _broadcast_beta(self, host):
        if self.mode == "fixed":
            return tf.constant(1.0, host.dtype)     # equivalent to host+wm
        if self.mode == "scalar":
            return tf.clip_by_value(self.beta, -self.b_max, self.b_max)
        if self.mode == "frame":
            beta_bc = tf.repeat(self.beta, self.Fs, axis=1)
            return tf.clip_by_value(beta_bc[:, :tf.shape(host)[1]], -self.b_max, self.b_max)
        # mode == "cnn"
        x = tf.concat([host, tf.zeros_like(host)], axis=-1)   # dummy second channel for convs
        g = self.c3(self.c2(self.c1(x)))                      # (-1,+1)
        return self.b_max * g

    # --------------------------------------------------------------
    def call(self, inputs):
        host, wm = inputs
        beta_bc  = self._broadcast_beta(host)                 # (B,T,1)
        return host + beta_bc * wm

    # --------------------------------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(mode=self.mode, beta_init=self.b0, beta_max=self.b_max,
                        frame_size=self.Fs, filters=self.filters))
        return cfg

'''
# 1) Global trainable α, fixed addition
wm_embed = WatermarkEmbedding(mode="global", alpha_init=0.05, alpha_max=0.2)
wm_add   = WatermarkAddition(mode="fixed")

# 2) Per-frame α and β
wm_embed = WatermarkEmbedding(mode="frame", frame_size=160, alpha_init=0.05)
wm_add   = WatermarkAddition(mode="frame", frame_size=160, beta_init=0.05)

# 3) Content-aware CNN mixer (β envelope)
wm_embed = WatermarkEmbedding(mode="global", alpha_init=0.05)
wm_add   = WatermarkAddition(mode="cnn",  filters=64, beta_max=0.15)

'''