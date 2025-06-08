#!/usr/bin/env bash
# pcm16k_to_wav.sh
#
# Convert a folder of 16kHz mono 16-bit PCM files into WAV files using ffmpeg.
#
# Usage: ./pcm16k_to_wav.sh /path/to/pcm16k  /path/to/wav_output
# -----------------------------------------------------------------

set -euo pipefail

# -------- 1. Parse arguments -------------------------------------
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <in_pcm_dir> <out_wav_dir>" >&2
  exit 1
fi
IN_DIR="$(realpath "$1")"
OUT_DIR="$(realpath "$2")"

# -------- 2. Verify prerequisites --------------------------------
command -v ffmpeg >/dev/null 2>&1 || {
  echo "ffmpeg not found. Install it and retry." >&2
  exit 1
}

# -------- 3. Convert PCM to WAV ----------------------------------
echo "Converting PCM files in $IN_DIR → $OUT_DIR"

find "$IN_DIR" -type f -iname '*.pcm' | while read -r PCM; do
  REL_PATH="${PCM#$IN_DIR/}"
  DST_PATH="${OUT_DIR}/${REL_PATH%.*}.wav"
  mkdir -p "$(dirname "$DST_PATH")"

  ffmpeg -loglevel error -y \
         -f s16le -ar 16000 -ac 1 \
         -i "$PCM" "$DST_PATH" \
    && echo "✓ $(basename "$PCM") → ${REL_PATH%.*}.wav" \
    || echo "✗ Failed: [$PCM]"
done

echo "Done. WAV files are saved in $OUT_DIR"
