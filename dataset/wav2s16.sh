#!/usr/bin/env bash
# wav_to_s16.sh
#
# Convert a tree of WAV files into 16-kHz, mono, 16-bit raw PCM (.s16) files.
# Uses ffmpeg for resampling and format change.
#
# Usage: ./wav_to_s16.sh /path/to/wav_input /path/to/output_s16
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
echo "Converting WAVs in $IN_DIR → $OUT_DIR (.s16 format)"

find "$IN_DIR" -type f -iname '*.wav' -exec bash -c '
  for WAV; do
    REL_PATH="${WAV#'"$IN_DIR"'/}"
    DST_PATH="'"$OUT_DIR"'/${REL_PATH%.*}.s16"
    mkdir -p "$(dirname "$DST_PATH")"
    ffmpeg -loglevel error -y -i "$WAV" -ac 1 -ar 16000 -f s16le "$DST_PATH" \
      && echo "✓ $(basename "$WAV") → ${REL_PATH%.*}.s16" \
      || echo "✗ Failed: [$WAV]"
  done
' _ {} +

echo "Done. .s16 files are under $OUT_DIR"
