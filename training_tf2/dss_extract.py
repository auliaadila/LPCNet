#!/usr/bin/env python3
# extract_bits.py  – classical DSS detector for the LPCNet watermark

import numpy as np
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("residual_frames",      help="residual_frames.npy")
parser.add_argument("wm_residual_frames",   help="wm_residual_frames.npy")
parser.add_argument("payload",   help="payload_bits.npy")
parser.add_argument("--alpha", type=float, default=0.04,
                    help="α used at embed-time (default 0.04)")
args = parser.parse_args()

# ------------------------------------------------------------------
# 1. Load per-frame residual and watermark
# ------------------------------------------------------------------
res = np.load(args.residual_frames)        # shape (F,160)
wm  = np.load(args.wm_residual_frames)     # shape (F,160)

assert res.shape == wm.shape, "shapes must match"

# ------------------------------------------------------------------
# 2. DSS detection:   corr = Σ wm*res  ;  sign → bit
# ------------------------------------------------------------------
def dc_bin(wm, ndigits=12):
    """Return rounded DC term of FFT, mimicking MATLAB round(c(1),n)."""
    return np.round(np.fft.fft(wm)[0], ndigits)

corr = (wm * res).sum(axis=1) / args.alpha     # divide by α
bits_hat = (corr > 0).astype(np.int8)          # 0/1 array  (F,) (490,)

# optional: BER if you stored the ground-truth bits
truth = np.load(args.payload)  # shape (F,64) flattened later

import IPython
IPython.embed()

ber = (bits_hat != truth).mean()

# ------------------------------------------------------------------
# 3. Reshape into (frames, 64) if you need the per-bit layout
# ------------------------------------------------------------------
BITS_PER_FRAME = 64
bits_hat = bits_hat.reshape(-1, BITS_PER_FRAME)   # (num_frames, 64)

out_path = os.path.splitext(args.wm_residual_frames)[0] + "_detected_bits.npy"
np.save(out_path, bits_hat)
print("Detected bits saved to", out_path)