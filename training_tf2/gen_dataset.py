## TO DO: check with lpcnet; probably not using this
#!/usr/bin/env python3
"""
Generate two TFRecord files:

1. embed.tfrecord   – everything stage-1/3/4 need
   Features per 2400-sample chunk:
       pcm          (int16)         – clean waveform
       feat         (float32, 20)   – acoustic features
       pitch        (float32, 1)
       lpcoeffs     (float32, 16)
       pn           (int8, 2400)    – +1/-1 spreading vector
       sigma        (float32, 2400) – masking threshold
       bits         (int8, 64)      – payload

2. extractor.tfrecord  – residual + label for CNN extractor
       residual_w   (float32, 160) – water-marked residual, *one LPC frame*
       bits         (int8, 64)

Usage
-----
$ python gen_dataset.py \
      --wav-list  wav_scp.txt \
      --out-dir   data/

`wav_scp.txt` is a plain text file (one path per line).
"""

import argparse, os, sys, math, random, pathlib
import numpy as np
import soundfile as sf
import tensorflow as tf

# ───────────────────────────────── HELPER SIGNAL CODE ────────────────────────────────── #
FRAME       = 160            # samples per LPC frame  (10 ms @16 kHz)
CHUNK       = 15 * FRAME     # 2400-sample training chunk
LP_ORDER    = 16
SSL         = -20.0          # embedding depth (dB)

# ---- light-weight LPC helpers --------------------------------------------------------- #
## Done by tf_funcs.py > diff_pred(Layer)
def lpc_analysis(frame, order=LP_ORDER, gamma=0.9):
    """Return LPC coeffs (including a[0]=1) and prediction"""
    # Levinson–Durbin via numpy.linalg
    autocorr = np.correlate(frame, frame, mode='full')[FRAME-1:FRAME+order]
    R = autocorr[:order]
    r = autocorr[1:order+1]
    a = np.ones(order+1)
    E = R[0]
    for i in range(1, order+1):
        lam = (r[i-1] - (R[1:i] @ a[1:i][::-1])) / E
        a[i]   = lam
        a[1:i] = a[1:i] - lam * a[i-1:0:-1]
        E *= 1 - lam**2
    # γ-weight
    a[1:] *= gamma ** np.arange(1, order+1)
    pred = np.convolve(frame, -a[1:], mode='same')
    return a.astype(np.float32), pred.astype(np.float32)

## originally not using mfcc
def mfcc_dummy(frame):
    """Placeholder 20-D feature: log-mel of power spectrum"""
    S = np.abs(np.fft.rfft(frame * np.hanning(FRAME)))**2
    mel = np.log(S + 1e-10)[:20]
    return mel.astype(np.float32)

def pitch_dummy(frame):
    """Very rough pitch tracker (autocorr peak)"""
    ac = np.correlate(frame, frame, 'full')[FRAME:]
    peak = np.argmax(ac[20:200]) + 20   # ignore first 20 lags
    return np.array([16000/peak], np.float32)

# ---- TFRecord utilities -------------------------------------------------------------- #
def _bytes_feature(arr):  # flattened
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[arr.tobytes()]))

def make_embed_example(pcm, feat, pitch, lp, pn, sigma, bits):
    feat_map = {
        'pcm'     : _bytes_feature(pcm.astype(np.int16)),
        'feat'    : _bytes_feature(feat),
        'pitch'   : _bytes_feature(pitch),
        'lpcoeffs': _bytes_feature(lp),
        'pn'      : _bytes_feature(pn.astype(np.int8)),
        'sigma'   : _bytes_feature(sigma),
        'bits'    : _bytes_feature(bits.astype(np.int8))
    }
    return tf.train.Example(
        features=tf.train.Features(feature=feat_map)
    )

def make_extractor_example(residual_w, bits):
    feat_map = {
        'residual_w': _bytes_feature(residual_w),
        'bits'      : _bytes_feature(bits.astype(np.int8))
    }
    return tf.train.Example(
        features=tf.train.Features(feature=feat_map)
    )

# ─────────────────────────────────── MAIN LOOP ───────────────────────────────────────── #
def process_wav(path, writer_embed, writer_ext):
    sig, fs = sf.read(path, always_2d=False)
    if fs != 16000:
        raise ValueError(f"{path} not 16 kHz")
    # Trim to multiple of CHUNK
    trim = len(sig) - len(sig)%CHUNK
    sig  = sig[:trim]
    # --------------- iterate over training chunks (15 LPC frames) -------------- #
    for t0 in range(0, len(sig), CHUNK):
        chunk = sig[t0:t0+CHUNK]
        # per-frame arrays to accumulate
        pns, sigmas, bits64 = [], [], []
        lpcs, feats, pitches = [], [], []
        residual_frames, residualW_frames = [], []
        slev_db = 10*np.log10(np.mean(chunk**2) + 1e-12)

        # ---- frame-wise processing (160 samples) ---- #
        for f in range(15):
            fr = chunk[f*FRAME:(f+1)*FRAME]
            lp, pred  = lpc_analysis(fr)
            resid     = fr - pred
            clev_db   = 10*np.log10(np.mean(resid**2) + 1e-12)
            # payload
            bits = np.random.randint(0,2,64).astype(np.uint8)
            pn   = np.repeat(bits*2-1, FRAME)      # ±1, length 160
            alpha = 10**(((slev_db - clev_db) + SSL)/20.)
            resid_w = resid + alpha*resid*pn
            # store per-LPC-frame items
            residual_frames.append(resid)
            residualW_frames.append(resid_w)
            bits64.append(bits)
            pns.append(pn)
            sigmas.append(np.abs(resid)*0.1)
            lpcs.append(lp[1:])        # exclude a[0]
            feats.append(mfcc_dummy(fr))
            pitches.append(pitch_dummy(fr))

        # ---- pack 15 frames → one Example (embed) ---- #
        pcm_2400  = chunk.astype(np.float32)
        example_e = make_embed_example(
            pcm      = pcm_2400,
            feat     = np.vstack(feats),           # (15,20)
            pitch    = np.vstack(pitches),         # (15,1)
            lp       = np.vstack(lpcs),            # (15,16)
            pn       = np.concatenate(pns)[:,None],# (2400,1)
            sigma    = np.concatenate(sigmas)[:,None],
            bits     = np.vstack(bits64)           # (15,64)
        )
        writer_embed.write(example_e.SerializeToString())

        # ---- one extractor Example per LPC frame ---- #
        for resid_w, bits in zip(residualW_frames, bits64):
            ex_ext = make_extractor_example(resid_w, bits)
            writer_ext.write(ex_ext.SerializeToString())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav-list', required=True,
                    help='Text file: one absolute wav path per line')
    ap.add_argument('--out-dir',  required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    path_embed = os.path.join(args.out_dir, 'embed.tfrecord')
    path_ext   = os.path.join(args.out_dir, 'extractor.tfrecord')
    writer_embed = tf.io.TFRecordWriter(path_embed)
    writer_ext   = tf.io.TFRecordWriter(path_ext)

    with open(args.wav_list) as f:
        wavs = [l.strip() for l in f if l.strip()]
    random.shuffle(wavs)

    for w in wavs:
        print('⇢', w)
        try:
            process_wav(w, writer_embed, writer_ext)
        except Exception as e:
            print('  ! skip:', e, file=sys.stderr)

    writer_embed.close(); writer_ext.close()
    print('Finished. Wrote', path_embed, 'and', path_ext)

if __name__ == '__main__':
    main()
