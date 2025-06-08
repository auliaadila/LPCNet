#!/usr/bin/env bash
# wav24k_to_pcm16k.sh
#
# Convert a tree of 24-kHz WAV files into 16-kHz, mono, 16-bit raw PCM
# Uses ffmpeg for the resample + format change.
#
# Usage: ./wav24k_to_pcm16k.sh /path/to/24k_wavs  /path/to/pcm16k
# -----------------------------------------------------------------

set -euo pipefail

# -------- 1. Parse arguments -------------------------------------
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <in_dir> <out_dir>" >&2
  exit 1
fi
IN_DIR="$(realpath "$1")"
OUT_DIR="$(realpath "$2")"

# -------- 2. Verify prerequisites --------------------------------
command -v ffmpeg >/dev/null 2>&1 || {
  echo "ffmpeg not found. Install it and retry." >&2
  exit 1
}

# -------- 3. Find & process WAV files ----------------------------
echo "Converting WAVs in $IN_DIR → $OUT_DIR"

# find "$IN_DIR" -type f -iname '*.wav' -print0 | while IFS= read -r -d '' WAV; do
#   [[ -f "$WAV" ]] || { echo "File not found: $WAV"; continue; } # sanity check after read line

#   REL_PATH="${WAV#$IN_DIR/}"
#   DST_PATH="${OUT_DIR}/${REL_PATH%.*}.pcm"

#   mkdir -p "$(dirname "$DST_PATH")"

#   ffmpeg -loglevel error -y \
#          -i "$WAV"          \
#          -ac 1              \
#          -ar 16000          \
#          -f s16le           \
#          "$DST_PATH"

#   printf "✓ %s → %s\n" "$(basename "$WAV")" "${REL_PATH%.*}.pcm"
# done

find "$IN_DIR" -type f -iname '*.wav' -exec bash -c '
  for WAV; do
    REL_PATH="${WAV#'"$IN_DIR"'/}"
    DST_PATH="'"$OUT_DIR"'/${REL_PATH%.*}.pcm"
    mkdir -p "$(dirname "$DST_PATH")"
    ffmpeg -loglevel error -y -i "$WAV" -ac 1 -ar 16000 -f s16le "$DST_PATH" \
      && echo "✓ $(basename "$WAV") → ${REL_PATH%.*}.pcm" \
      || echo "✗ Failed: [$WAV]"
  done
' _ {} +



echo "Done. PCM files are under $OUT_DIR"
