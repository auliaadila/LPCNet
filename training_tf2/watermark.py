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

    •	__init__ → remember configuration, no tensor logic.
	•	build → allocate weights once input shapes are known.
	•	call → run the actual computation each forward pass.

    """
    def __init__(self,
                 frame_size     = 160,
                 bits_per_frame = 64,
                #  bits_in = None,
                #  sampling_frequency = 16000,
                 alpha_init     = 0.05,
                #  ssl_db = -25.0,
                #  eps=1e-8,
                 trainable_alpha= False,
                #  adaptive_alpha=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.frame_size      = frame_size
        self.bpf = bits_per_frame
        # self.bits_in=bits_in
        # self.fs  = sampling_frequency
        # self.ssl = ssl_db
        # self.eps = eps
        self.alpha_init      = alpha_init
        self.trainable_alpha = trainable_alpha
        # self.adaptive_alpha = adaptive_alpha

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
        wm_bpf, residual = inputs            # unpack
        residual = residual * tf.ones_like(wm_bpf)  # broadcast residual:  (B,2400,1) → (B,2400,64)
        wm_bpf = tf.cast(wm_bpf * 2 - 1, tf.float32)      # (B,F,BPF)
        wm = self.alpha * wm_bpf * residual                 # (B,T,BPF) # per-bit contribution
        wm_single = tf.reduce_sum(wm, axis=-1, keepdims=True)  # (B,2400,1) # collapse → single channel

        return wm_single





    '''
    def call(self, inputs):
        wm_bpf, residual = inputs            # unpack
        # need host signal, but input is in feat.32

        # squeeze channel dim if present (B,T,1) -> (B,T)
        # if residual.shape.rank == 3:
        #     residual = tf.squeeze(residual, axis=-1)        # (B,T)

        B = tf.shape(residual)[0] #batch size
        T = tf.shape(residual)[1] #number of samples
        F = tf.shape(wm_bpf)[1]


        print("===> EMBED")
        print("wm_bpf:",wm_bpf.shape)
        print("residual:",residual.shape)

        
        In [4]: wm_bpf.shape
        Out[4]: TensorShape([128, 2400, 64])

        In [5]: residual.shape
        Out[5]: TensorShape([128, 2400, 1])
        

        

>>>>>>> 1e5fc9e (refine watermark and dataloader)
        # print("Embedding: B, N, F")
        # print(B,T,F)

        # Tensor("wm_embed/strided_slice:0", shape=(), dtype=int32)
        # Tensor("wm_embed/strided_slice_1:0", shape=(), dtype=int32)
        # Tensor("wm_embed/strided_slice_2:0", shape=(), dtype=int32)




        # duration = N / self.fs
        # total_bits = int(np.ceil(bps * duration))

        # # Prepare watermark bits: repeat if shorter
        # wm = np.asarray(wm_bps, dtype=int).flatten()
        # if wm.size < total_bits:
        #     reps = int(np.ceil(total_bits / wm.size))
        #     wm = np.tile(wm, reps)
        # wm = wm[:total_bits] # {0,1} #to whole frame, unextended version
        # bipolar = 2 * wm - 1  # {0,1} -> {-1,+1}  #to whole frame, unextended version

        # # Frame and bit calculations
        # num_frames = N // self.frame_size
        # frames_per_sec = self.fs / self.frame_size
        # frames_per_bit = int(np.ceil(frames_per_sec / bps))

        # bit_s = np.repeat(wm, frames_per_bit)[:num_frames] #to whole frame, extended version
        # bit_stream = np.repeat(bipolar, frames_per_bit)[:num_frames] #to whole frame, extended version


        # --- safety check (dynamic): F*frame_size must equal T -------------
        # error
        # with tf.control_dependencies(
        #     [tf.debugging.assert_equal(F * self.frame_size, T,
        #      message="msg_bits length × frame_size must equal residual length")]):
        #     residual = tf.identity(residual)



        # 0/1  ->  ±1
        bipolar = tf.cast(wm_bpf * 2 - 1, tf.float32)      # (B,F,BPF)
        # print("wm_bpf:",wm_bpf.shape) #wm_bpf: (128, None, 1)
        # print(wm_bpf)
        # print("bipolar:", bipolar.shape) #bipolar: (128, None, 1)
        # print(bipolar)

        # repeat each bit across its chip region
        chip_len   = self.frame_size // self.bpf  # integer part
        extra      = self.frame_size - chip_len * self.bpf

        # build a repeat pattern that fills the 160 samples even if not exact
        rep_pattern = ([chip_len + 1] * extra) + ([chip_len] *
                                                  (self.bpf - extra))
        rep_pattern = tf.constant(rep_pattern, dtype=tf.int32)    # (BPF,)

        # bits_spread = tf.repeat(bipolar,
        #                         repeats=rep_pattern,
        #                         axis=-1)                         # (B,F,160)
        # bits_pm: (B, F, BPF)

        # rep_pattern: Python list or 1D Tensor of length BPF
        # We want bits_spread: (B, F, sum(rep_pattern)) == (B, F, T)

        # 1) collect a list of (B, F, rep_pattern[i]) tensors
        spread_slices = []
        for i in range(self.bpf):
            # select the single-bit slice
            slice_i = bipolar[:, :, i:i+1]          # shape (B, F, 1)
            # tile it rep_pattern[i] times along the last axis
            tiled_i = tf.tile(slice_i, [1, 1, rep_pattern[i]])  # (B, F, rep_pattern[i])
            spread_slices.append(tiled_i)

        # print("Spread slice:",len(spread_slices)) #64

        # 2) concatenate all those slices along the last axis
        bits_spread = tf.concat(spread_slices, axis=-1)  # shape (B, F, T)
        # print("bits spread:", bits_spread) #bits spread: Tensor("wm_embed/concat:0", shape=(128, None, 3), dtype=float32)

        # 3) finally reshape to (B, T)
        bits_spread = tf.reshape(bits_spread, (B, T))            # (B,T)
        # print("bits spread reshape:", bits_spread.shape) #bits spread reshape: (128, None)

        # import IPython
        # IPython.embed()

        # --- watermark: alpha * bit * residual -----------------------------
        # print("alpha:", self.alpha)
        # alpha: MirroredVariable:{
        #     0: <tf.Variable 'wm_embed/alpha:0' shape=() dtype=float32>
        #     }
        # print("residual:", residual.shape) #residual: (128, None)

        # if constant alpha
        wm = self.alpha * bits_spread * residual                 # (B,T)
        # print("wm:", wm.shape) #wm: (128, None)

        
        # if adaptive alpha
        # calculate power per frame ()
        pow_host  = tf.reduce_mean(tf.square(host_f),  axis=-1) + self.eps #need host signal
        pow_resid = tf.reduce_mean(tf.square(residual), axis=-1) + self.eps
        log10     = lambda x: tf.math.log(x)/tf.math.log(tf.constant(10.))
        s_db      = 10.0 * log10(pow_host)
        c_db      = 10.0 * log10(pow_resid)

        # 3. α per frame according to MATLAB formula
        lev_db    = s_db - c_db + self.SSL          # (B,F)
        alpha_f   = tf.pow(10.0, lev_db / 20.0)     # (B,F)
        


        # residual_mark = residual + wm                            # add

        return tf.expand_dims(wm, -1)                 # (B,T,1)
        # return tf.expand_dims(residual_mark, -1)                 # (B,T,1)

        # pcm_w: (128, None, 1)
        # m_out: (128, 2400, 258)

    '''
    def get_config(self):
        base = super().get_config()
        base.update(dict(frame_size=self.frame_size,
                         bpf=self.bits_per_frame,
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

    def __init__(self, trainable_beta=False, beta_init=0.1, **kw):
        super().__init__(**kw)
        self.trainable_beta = trainable_beta
        self.beta_init = beta_init
        # self.filters = filters #32
        # if learnable_mask:
        #     self.conv1 = Conv1D(filters, 9, padding='same', activation='relu')
        #     self.conv2 = Conv1D(filters, 9, padding='same', activation='relu')
        #     self.conv3 = Conv1D(1, 9, padding='same', activation='tanh')  # → (-1,+1)

    def build(self, input_shape):
        if self.trainable_beta:
            self.beta = self.add_weight(name='beta',
                                         shape=(),
                                         initializer=tf.constant_initializer(
                                             self.beta_init),
                                         trainable=True)
        else:
            self.beta = tf.constant(self.beta_init, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        host, wm = inputs                   # expect (B,T,1) each

        

        if not self.trainable_beta:
            return host + wm               # simple addition

        # TO DO: change this
        # ----- learn g(n) ---------------------------------------------------
        x = Concatenate()([host, wm])       # (B,T,2)
        x = self.conv1(x)
        x = self.conv2(x)
        g = self.beta * self.conv3(x)       # scale tanh to (-β, β)
        # Learnable “mixer” mask – a tiny 1-D CNN learns a gain envelope g(n)\in[-\beta,+\beta] so the network can hide the watermark where the ear is least sensitive

        return host + g * wm

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(trainable_beta=self.trainable_beta,
                        beta_init=self.beta_init,
                        beta=self.beta))
        return cfg