### LPCNet - Fully Fixed Version for TF2.x

import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Reshape, Concatenate, GaussianNoise
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer

from mdense import MDense
from watermark import WatermarkEmbedding, WatermarkAddition
from tf2_funcs import *
from diffembed import diff_Embed
from parameters import set_parameter

frame_size = 160
pcm_bits = 8
embed_size = 128
pcm_levels = 2**pcm_bits

def interleave(p, samples):
    p2 = tf.expand_dims(p, 3)
    nb_repeats = pcm_levels // (2 * p.shape[2])
    p3 = tf.reshape(tf.repeat(tf.concat([1 - p2, p2], 3), nb_repeats), (-1, samples, pcm_levels))
    return p3

# === Custom Layers to avoid Lambda problems ===

class ErrorCalc(tf.keras.layers.Layer):
    def call(self, inputs):
        pcm, preds = inputs
        err = tf_l2u(pcm - tf.roll(preds, shift=1, axis=1))
        return err

class RepeatLayer(tf.keras.layers.Layer):
    def __init__(self, repeats, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.repeats = repeats
        self.axis = axis

    def call(self, inputs):
        return tf.repeat(inputs, repeats=self.repeats, axis=self.axis)

class TreeToPdf(tf.keras.layers.Layer):
    def __init__(self, samples, **kwargs):
        super().__init__(**kwargs)
        self.samples = samples

    def call(self, p):
        return interleave(p[:,:,1:2], self.samples) * interleave(p[:,:,2:4], self.samples) * \
               interleave(p[:,:,4:8], self.samples) * interleave(p[:,:,8:16], self.samples) * \
               interleave(p[:,:,16:32], self.samples) * interleave(p[:,:,32:64], self.samples) * \
               interleave(p[:,:,64:128], self.samples) * interleave(p[:,:,128:256], self.samples)

def new_lpcnet_model(rnn_units1=384, rnn_units2=16, nb_used_features=20, batch_size=128, training=False, 
                     quantize=False, flag_e2e=False, cond_size=128, lpc_order=16, lpc_gamma=1.):

    pcm = Input(shape=(None, 1), batch_size=batch_size)
    dpcm = Input(shape=(None, 3), batch_size=batch_size)
    feat = Input(shape=(None, nb_used_features), batch_size=batch_size)
    pitch = Input(shape=(None, 1), batch_size=batch_size)
    bits_in = Input(shape=(None, 1), batch_size=batch_size)

    fconv1 = tf.keras.layers.Conv1D(cond_size, 3, padding='same', activation='tanh', name='feature_conv1')
    fconv2 = tf.keras.layers.Conv1D(cond_size, 3, padding='same', activation='tanh', name='feature_conv2')
    pembed = Embedding(256, 64, name='embed_pitch')

    cat_feat = Concatenate()([feat, Reshape((-1, 64))(pembed(pitch))])
    cfeat = fconv2(fconv1(cat_feat))

    fdense1 = Dense(cond_size, activation='tanh', name='feature_dense1')
    fdense2 = Dense(cond_size, activation='tanh', name='feature_dense2')
    cfeat = fdense2(fdense1(cfeat))

    lpcoeffs = Input(shape=(None, lpc_order), batch_size=batch_size)
    real_preds = diff_pred(name="real_lpc2preds")([pcm, lpcoeffs])

    weighting = lpc_gamma ** tf.range(1, lpc_order+1, dtype=tf.float32)
    weighted_lpcoeffs = lpcoeffs * weighting
    tensor_preds = diff_pred(name="lpc2preds")([pcm, weighted_lpcoeffs])

    error_calc = ErrorCalc(name="error_calc")
    residual = error_calc([pcm, tensor_preds])

    wm_embed = WatermarkEmbedding(frame_size=160, bits_per_frame=64, alpha_init=0.04, trainable_alpha=True, name='wm_embed')
    residual_w = wm_embed([bits_in, residual])

    wm_add = WatermarkAddition(trainable_beta=False, beta_init=0.1, name='wm_add')
    pcm_w = wm_add([pcm, residual_w])

    residual_u = tf_l2u(residual)
    embed = diff_Embed(name='embed_sig', initializer=PCMInit())
    cpcm = Concatenate()([tf_l2u(pcm), tf_l2u(tensor_preds), residual_u])
    cpcm = GaussianNoise(.3)(cpcm)
    cpcm = Reshape((-1, embed_size * 3))(embed(cpcm))

    rep = RepeatLayer(repeats=frame_size, name="repeat_layer")
    rnn_in = Concatenate()([cpcm, rep(cfeat)])

    md = MDense(pcm_levels, activation='sigmoid', name='dual_fc')

    rnn = CuDNNGRU(rnn_units1, return_sequences=True, return_state=True, name='gru_a', stateful=True)
    rnn2 = CuDNNGRU(rnn_units2, return_sequences=True, return_state=True, name='gru_b', stateful=True)

    gru_out1, _ = rnn(rnn_in)
    gru_out1 = GaussianNoise(.005)(gru_out1)
    gru_out2, _ = rnn2(Concatenate()([gru_out1, rep(cfeat)]))

    tree_to_pdf_layer = TreeToPdf(samples=2400, name="tree_to_pdf_train")
    ulaw_prob = tree_to_pdf_layer(md(gru_out2))

    m_out = Concatenate(name='pdf')([tensor_preds, real_preds, ulaw_prob])

    model = Model([pcm, feat, pitch, bits_in, lpcoeffs],
                  outputs={'pdf': m_out, 'residual_w': residual_w, 'pcm_w': pcm_w})

    set_parameter(model, 'lpc_gamma', lpc_gamma, dtype='float64')
    set_parameter(model, 'flag_e2e', flag_e2e, dtype='bool')
    set_parameter(model, 'lookahead', 2, dtype='int32')

    return model

### === End of File ===