# test_dss_layer.py  ----------------------------------------------
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa

from watermark2 import DSSWatermarkDynamicAlpha   # your class

# host_file = 
# residue_file = 
# wm_matlab_file =

wav, sr = sf.read(host_file, dtype='float32')       # mono 16-kHz
wav     = wav[:160*10]                                     # 10 frames
wav     = wav.reshape(1, -1, 1)                            # (B,T,1)

# residue = read npy

bits= np.random.randint(0, 2, size=(1, F, 64), dtype='int32')
layer = DSSWatermarkDynamicAlpha(frame_size=160,
                                 bits_per_frame=64,
                                 ssl_db=-25.0)
wm_tf = layer([bits, wav, residue])          # (1,T,1) tensor
wm_tf = wm_tf.numpy().squeeze()            # to NumPy (T,)

# wm_np = np.empty_like(wm_tf)
# for k in range(F):
#     s   = wav.squeeze()[k*160:(k+1)*160]
#     e   = resid.squeeze()[k*160:(k+1)*160]
#     slev= 10*np.log10(np.mean(s*s)+1e-8)
#     clev= 10*np.log10(np.mean(e*e)+1e-8)
#     lev = slev - clev - 25.0              # SSL = −25 dB
#     alpha=10**(lev/20)
#     chip_len, extra = 2, 160 - 2*64
#     bit_sig = np.repeat(bits[0,k]*2-1,
#                         [chip_len+1]*extra + [chip_len]*(64-extra))
#     wm_np[k*160:(k+1)*160] = alpha * bit_sig * e

# import from matlab result

wm_matlab, sr = sf.read(wm_matlab_file, dtype='float32')       # mono 16-kHz
wm_matlab     = wm_matlab[:160*10]                                     # 10 frames
wm_matlab     = wm_matlab.reshape(1, -1, 1)                            # (B,T,1)

err = np.abs(wm_tf - wm_matlab)
print("max|diff| =", err.max(), "   mean|diff| =", err.mean())