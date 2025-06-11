import numpy as np
import soundfile as sf           # pip install soundfile

residual = np.load('out_residual.npy')     # shape (N,) float32

# 1. clamp to ±1 and convert to int16 for WAV
residual_pcm = np.clip(residual, -1.0, 1.0)
residual_pcm = (residual_pcm * 32767).astype('int16')

# 2. write a temp WAV (16-kHz mono)
sf.write('residual.wav', residual_pcm, 16000, subtype='PCM_16')
print("Play residual.wav with any audio player.")