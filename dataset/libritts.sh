# wget https://openslr.elda.org/resources/12/train-clean-100.tar.gz
# wget https://openslr.elda.org/resources/12/test-clean.tar.gz
# wget https://openslr.elda.org/resources/12/dev-clean.tar.gz
# set -euo pipefail

ROOT="/home/adila/Data/audio/LibriTTS"
tar -zxvf "$ROOT/train-clean-100.tar.gz" -C "$ROOT"
tar -zxvf "$ROOT/test-clean.tar.gz" -C "$ROOT"
tar -zxvf "$ROOT/dev-clean.tar.gz" -C "$ROOT"
python move_wav.py