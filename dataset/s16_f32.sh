#!/usr/bin/env bash
# s16_to_f32.sh
#
# Convert all .s16 files in a directory into .f32 files using ./dump_data
# Output .f32 files will be saved in the specified output directory
#
# Usage: ./s16_to_f32.sh /path/to/s16_dir /path/to/f32_dir
# -----------------------------------------------------------------

set -euo pipefail

# -------- 1. Parse arguments -------------------------------------
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <s16_dir> <f32_out_dir>" >&2
  exit 1
fi

IN_DIR="$(realpath "$1")"
OUT_DIR="$(realpath "$2")"

# -------- 2. Check prerequisites ---------------------------------
if [[ ! -x "./dump_data" ]]; then
  echo "Error: ./dump_data not found or not executable." >&2
  exit 1
fi

# -------- 3. Process each .s16 file ------------------------------
echo "Processing .s16 files from $IN_DIR to $OUT_DIR"

find "$IN_DIR" -type f -name '*.s16' | while read -r S16_FILE; do
  REL_PATH="${S16_FILE#$IN_DIR/}"
  F32_FILE="${OUT_DIR}/${REL_PATH%.s16}.f32"

  mkdir -p "$(dirname "$F32_FILE")"

  echo "→ $S16_FILE → $F32_FILE"
  ./dump_data -test "$S16_FILE" "$F32_FILE" || echo "✗ Failed: $S16_FILE"
done

echo "Done. .f32 files are saved in $OUT_DIR"
