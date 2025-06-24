#!/usr/bin/env bash
# -------------------------------------------------------------------
# Joint Watermark Training launch script
# -------------------------------------------------------------------
# Usage:
#   ./train_watermark.sh /path/to/features.f32 /path/to/audio.u8
#
# The first two positional arguments are REQUIRED:
#   1) binary feature file   (float32, produced by your feature-extractor)
#   2) binary audio data     (uint8   µ-law or 16-bit PCM re-encoded to 8-bit)
#
# All other hyper-parameters can be tweaked below or overridden on the
# command line after the two required paths.
# -------------------------------------------------------------------

set -euo pipefail

# -------------------------------------------------
# 🎯 Positional arguments (required)
# -------------------------------------------------
if [[ $# -lt 2 ]]; then
  echo "ERROR: need <features.bin> and <audio.bin>" >&2
  echo "Example: $0 feats.f32 audio.u8 --epochs 20" >&2
  exit 1
fi
FEATURES="$1"
DATA="$2"
shift 2          # keep any extra CLI overrides intact

# -------------------------------------------------
# 💾 Output & logging
# -------------------------------------------------
OUT_DIR="checkpoints/watermark"
LOG_DIR="logs/watermark"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# -------------------------------------------------
# ⚙️  Default hyper-parameters
#      (override by appending flags when you call
#       this .sh script, e.g.  --lr 3e-4)
# -------------------------------------------------
EPOCHS=10
BATCH=32
LR=1e-4

ALPHA_EXTRACT=1.0
ALPHA_EMBED=0.1
ALPHA_ADV=0.01

ALPHA_MSE=1.0
ALPHA_SPECTRAL=0.5
ALPHA_TEMPORAL=0.3

# Optionally point to pretrained weights (leave empty to start from scratch)
LPCNET_WEIGHTS=""
EXTRACTOR_WEIGHTS=""
DISCRIM_WEIGHTS=""

# -------------------------------------------------
# 🚀  Launch
# -------------------------------------------------
python training_tf2/train_watermark_joint_v1.py \
  "${FEATURES}" \
  "${DATA}" \
  --epochs        "${EPOCHS}" \
  --batch-size    "${BATCH}" \
  --lr            "${LR}" \
  --alpha-extract "${ALPHA_EXTRACT}" \
  --alpha-embed   "${ALPHA_EMBED}" \
  --alpha-adv     "${ALPHA_ADV}" \
  --alpha-mse     "${ALPHA_MSE}" \
  --alpha-spectral "${ALPHA_SPECTRAL}" \
  --alpha-temporal "${ALPHA_TEMPORAL}" \
  --output-dir    "${OUT_DIR}" \
  --log-dir       "${LOG_DIR}" \
  ${LPCNET_WEIGHTS:+--lpcnet-weights "$LPCNET_WEIGHTS"} \
  ${EXTRACTOR_WEIGHTS:+--extractor-weights "$EXTRACTOR_WEIGHTS"} \
  ${DISCRIM_WEIGHTS:+--discriminator-weights "$DISCRIM_WEIGHTS"} \
  "$@"

# End of script