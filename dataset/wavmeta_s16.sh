#!/usr/bin/env bash
# wav_meta_to_s16.sh
#
# Convert WAV files listed in a metadata file into 16-kHz, mono,
# 16-bit raw PCM (.s16) files.  Uses ffmpeg for resampling.
#
# Each line of the metadata must contain a WAV pathname (absolute
# or relative).  If the file has more columns (e.g. CSV), specify
# which 1-based column holds the path with --col.
#
# Usage
# -----
#   ./wav_meta_to_s16.sh <metadata_file> <out_dir> [--col N]
#
# Example
#   ./wav_meta_to_s16.sh librivox.csv pcm16k --col 1
# -----------------------------------------------------------------

set -euo pipefail

# -------- 1. Parse CLI arguments ---------------------------------
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <metadata_file> <out_dir> [--col N]" >&2
  exit 1
fi

META_FILE="$1"
OUT_DIR="$(realpath "$2")"
COL=1                                # default column index (1-based)

if [[ $# -ge 4 && "$3" == "--col" ]]; then
  COL="$4"
fi

# -------- 2. Verify prerequisites --------------------------------
command -v ffmpeg >/dev/null 2>&1 || {
  echo "ffmpeg not found. Install it and retry." >&2
  exit 1
}

[[ -f "$META_FILE" ]] || { echo "Metadata file not found: $META_FILE" >&2; exit 1; }

# -------- 3. Process each WAV listed in metadata -----------------
echo "Converting WAVs listed in $META_FILE → $OUT_DIR (.s16)"

line_no=0
while IFS= read -r line || [[ -n $line ]]; do
  (( line_no++ ))
  # Skip empty lines
  [[ -z "$line" ]] && continue

  # Extract the N-th field (handles CSV, TSV, or space-separated)
  WAV=$(echo "$line" | awk -v col="$COL" -F'[,\t ]+' '{print $col}') ## error

  # If header row contains non-existing file, skip it
  [[ ! -f "$WAV" ]] && { echo "Skipping line $line_no (file not found): $WAV"; continue; }

  # Build output path mirroring the directory structure of the WAV
  ABS_WAV=$(realpath "$WAV")
  REL_PATH="${ABS_WAV#/}"                       # strip leading /
  DST_PATH="$OUT_DIR/${REL_PATH%.*}.s16"

  mkdir -p "$(dirname "$DST_PATH")"

  if ffmpeg -loglevel error -y -i "$ABS_WAV" -ac 1 -ar 16000 -f s16le "$DST_PATH" ; then
    echo "✓ $(basename "$WAV") → ${REL_PATH%.*}.s16"
  else
    echo "✗ Failed: [$WAV]"
  fi

done < "$META_FILE"

echo "Done. .s16 files are under $OUT_DIR"
