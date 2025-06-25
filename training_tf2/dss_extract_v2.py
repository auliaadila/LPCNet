#!/usr/bin/env python3
# dss_extract.py  –  classical DSS detector (Python re-implementation
#                    of lpdss_dec.m logic)
# --------------------------------------------------------------------
import argparse
import os
import numpy as np
# import scipy.signal as sg
import soundfile as sf
import h5py
import lpcnet

# ---------- 0. command-line -----------------------------------------
p = argparse.ArgumentParser()
p.add_argument("host_wav",        help="clean speech (wav, 16-kHz mono)")
p.add_argument("wm_wav",          help="water-marked speech (same length)")
p.add_argument("residual_frame",          help="sw_residual_frames.npy") #sw?
p.add_argument("model_file",          help="model.h5; to get the frame_size (number of sample processed per frame)") #sw?
p.add_argument("feature_file",          help="feature.f32; to get the feature_chunk_size (number of frames)") #sw?
p.add_argument("--payload_bits",  help="ground-truth bits .npy (optional)")
p.add_argument("--bps",  type=int, default=64,   help="bits per speech")
p.add_argument("--lp_ord", type=int, default=16, help="LPC order")
args = p.parse_args()

# ---------- 1. load audio -------------------------------------------
host, sr  = sf.read(args.host_wav, dtype='float32') #fail to read
wm,   srw = sf.read(args.wm_wav,   dtype='float32')

assert sr == 16000 and srw == 16000, "Expect 16-kHz files"
assert host.shape == wm.shape,      "Host/WM length mismatch"
host = host.squeeze() #(X,1) -> (X) remove dimension 1
wm   = wm.squeeze()

# ---------- 2. derive frame boundaries ------------------------------
filename = args.model_file

with h5py.File(filename, "r") as f:
    units = min(f['model_weights']['gru_a']['gru_a']['recurrent_kernel:0'].shape)
    units2 = min(f['model_weights']['gru_b']['gru_b']['recurrent_kernel:0'].shape)
    cond_size = min(f['model_weights']['feature_dense1']['feature_dense1']['kernel:0'].shape)
    e2e = 'rc2lpc' in f['model_weights']

model, _, _ = lpcnet.new_lpcnet_model(training = False, rnn_units1=units, rnn_units2=units2, flag_e2e = e2e, cond_size=cond_size, batch_size=1)
# add new objects here (defined in lpcnet)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy']) #to build internal tensor, required. but the loss functions described are arbitrary, it won't be used because there's no gradient flow
model.summary()

# feature_file = sys.argv[2]
# out_file = sys.argv[3]
feature_file  = args.feature_file
frame_size = model.frame_size #160
nb_features = 36
nb_used_features = model.nb_used_features #20

features = np.fromfile(feature_file, dtype='float32') #per file #(7056,) #(17640,)

# import IPython
# IPython.embed()

features = np.resize(features, (int(features.shape[0]/nb_features), nb_features)) #(196, 36) #(490, 36)
# features = np.resize(features, (-1, nb_features))
nb_frames = 1
feature_chunk_size = features.shape[0] #depends on the audio; 196, 490
pcm_chunk_size = frame_size*feature_chunk_size #78,400


features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
periods = (.1 + 50*features[:,:,18:19]+100).astype('int16')


# model.load_weights(filename);

# order = 16

# pcm = np.zeros((nb_frames*pcm_chunk_size, ))
#-----
total_len   = host.size #78383 (wav)

import IPython
IPython.embed()


assert total_len == pcm_chunk_size, "shapes must match" #shape is not match:)
# drate       = total_len // args.bps            # samples per bit-frame #1224
# num_frames  = args.bps #64
# pad_len     = num_frames*drate - total_len     # <= drate-1 #-47





# # zero-pad to exact multiple of drate
# if pad_len > 0:
#     host = np.pad(host, (0,pad_len))
#     wm   = np.pad(wm,   (0,pad_len))


# reshape to (frames , drate)
# host_f = host.reshape(num_frames, drate)
# wm_f   = wm.reshape(num_frames,   drate)

# # ---------- 3. compute LPC residual of HOST -------------------------
# residual_f = np.empty_like(host_f)             # (F,drate)
# for k in range(num_frames):
#     # LPC coefficients (librosa returns a[0]=1, a[1..])
#     a = sg.lfilter([1.0], [1.0], np.zeros(1)) # dummy placeholder
#     # use autocorrelation LPC (burg) via librosa
#     import librosa
#     a = librosa.lpc(host_f[k], args.lp_ord)    # length lp_ord+1
#     pred = sg.lfilter(-a[1:], 1.0, host_f[k]) # ∑ a_i s[n-i]
#     residual_f[k] = host_f[k] - pred

# ---------- 3.  LOAD residual frames directly ---------------------
# Replace the LPC-computation block with this.

'''

residual_f = np.load(args.residual_frame)        # shape (F,160)


# Sanity check: make sure we have the same number of frames & samples
assert residual_f.shape == host_f.shape, (
        f"Residual frame shape {residual_f.shape} "
        f"does not match host frame shape {host_f.shape}")

# ------------------------------------------------------------------

# ---------- 4. DSS detection (FFT-DC test) --------------------------
bits_hat = np.empty(num_frames, dtype=np.int8)
for k in range(num_frames):
    # wm × residual  (same as MATLAB)
    prod = wm_f[k] * residual_f[k]
    dc   = np.fft.fft(prod).real[0]            # DC bin
    bits_hat[k] = 1 if dc > 0 else 0 #490
    # bits_hat[k] = 1 if dc > 0 else -1 #490 (compare with +-1)

# ---------- 5. BER and save ----------------------------------------
if args.payload_bits and os.path.exists(args.payload_bits):
    truth = np.load(args.payload_bits).astype(np.int8).ravel()
    min_len = min(truth.size, bits_hat.size)
    ber = (bits_hat[:min_len] != truth[:min_len]).mean()
    print(f"BER = {ber:.4%} ({min_len} bits compared)")
else:
    print("Ground-truth not provided; skipping BER.")

out_path = os.path.splitext(args.wm_wav)[0] + "_detected_bits.npy"
np.save(out_path, bits_hat)
print("Detected bits saved to", out_path)
'''