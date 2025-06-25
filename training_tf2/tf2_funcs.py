"""
Tensorflow/Keras helper functions:
    1. mu-law <-> Linear domain conversion
    2. Differentiable LP prediction
    3. Differentiable RC <-> LPC
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Concatenate
from tensorflow.keras import backend as K

# mu-law constants
scale = 255.0 / 32768.0
scale_1 = 32768.0 / 255.0

@tf.function
def tf_l2u(x):
    s = tf.sign(x)
    x = tf.abs(x)
    u = s * (128 * tf.math.log(1 + scale * x) / tf.math.log(256.0))
    u = tf.clip_by_value(128 + u, 0.0, 255.0)
    return u

@tf.function
def tf_u2l(u):
    u = tf.cast(u, "float32")
    u = u - 128.0
    s = tf.sign(u)
    u = tf.abs(u)
    return s * scale_1 * (tf.math.exp(u / 128.0 * tf.math.log(256.0)) - 1)

# Differentiable Prediction Layer
class diff_pred(Layer):
    def call(self, inputs, lpcoeffs_N=16, frame_size=160):
        xt = inputs[0]  # shape: (batch, time, 1)
        lpc = inputs[1] # shape: (batch, frame, lpcoeffs_N)

        rept = tf.repeat(lpc, repeats=frame_size, axis=1)

        zpX = tf.concat([tf.zeros_like(xt[:, 0:lpcoeffs_N, :]), xt], axis=1)

        # assumes 2400 samples
        cX = tf.concat([zpX[:, (lpcoeffs_N - i):(lpcoeffs_N - i + 2400), :] for i in range(lpcoeffs_N)], axis=2)

        pred = -rept * cX

        return tf.reduce_sum(pred, axis=2, keepdims=True)

# RC -> LPC
class diff_rc2lpc(Layer):
    def call(self, inputs, lpcoeffs_N=16):
        def pred_lpc_recursive(input):
            lpc_in, ki = input
            ki_exp = tf.repeat(ki, repeats=tf.shape(lpc_in)[-1], axis=2)
            temp = lpc_in + ki_exp * tf.reverse(lpc_in, axis=[2])
            temp = tf.concat([temp, ki], axis=2)
            return temp

        lpc = inputs[:, :, :lpcoeffs_N]
        for i in range(1, lpcoeffs_N):
            lpc = pred_lpc_recursive([lpc[:, :, :i], tf.expand_dims(lpc[:, :, i], axis=-1)])
        return lpc

# LPC -> RC
class diff_lpc2rc(Layer):
    def call(self, inputs, lpcoeffs_N=16):
        def pred_rc_recursive(input):
            lpc_in, ki_in = input
            ki = tf.repeat(tf.expand_dims(ki_in[:, :, 0], axis=-1), repeats=tf.shape(lpc_in)[-1], axis=2)
            temp = (lpc_in - ki * tf.reverse(lpc_in, axis=[2])) / (1.0 - ki * ki)
            temp = tf.concat([temp, ki_in], axis=2)
            return temp

        rc = inputs
        for i in range(1, lpcoeffs_N):
            j = lpcoeffs_N - i + 1
            rc = pred_rc_recursive([rc[:, :, :(j - 1)], rc[:, :, (j - 1):]])
        return rc