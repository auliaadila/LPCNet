#!/usr/bin/env bash
# s16_to_wav.sh
#
# Convert a folder tree of 16-bit little-endian raw PCM files (*.s16)
# sampled at 16 kHz into standard WAV files.
#
# Usage:
#   ./s16_to_wav.sh <input_s16_dir>  <output_wav_dir>

set -euo pipefail

[[ $# -eq 2 ]] || { echo "Usage: $0 <in_s16_dir> <out_wav_dir>"; exit 1; }

# Ensure trailing slash for correct prefix matching
IN_DIR="$(realpath "$1")/"
OUT_DIR="$(realpath "$2")"

command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg not found"; exit 1; }

echo "Converting .s16 files in $IN_DIR → $OUT_DIR"

find "$IN_DIR" -type f -iname '*.s16' -print0 |
while IFS= read -r -d '' S16; do
  [[ -f $S16 ]] || { echo "?? file disappeared: $S16"; continue; }

  REL_PATH="${S16#"$IN_DIR"}"
  DST_PATH="$OUT_DIR/${REL_PATH%.*}.wav"

  mkdir -p "$(dirname "$DST_PATH")"

  # Debug
  echo "[DEBUG] S16=$S16"
  echo "[DEBUG] REL_PATH=$REL_PATH"
  echo "[DEBUG] DST_PATH=$DST_PATH"

  if ffmpeg -loglevel error -y \
            -f s16le -ar 16000 -ac 1 \
            -i "$S16" "$DST_PATH"; then
      echo "✓ $REL_PATH → ${REL_PATH%.*}.wav"
  else
      echo "✗ ffmpeg failed: $S16"
  fi
done