#!/bin/bash

mkdir -p dataset/Libri1h/train_feat dataset/Libri1h/train_pcm
LOGFILE="dump_log.txt"
echo "Dumping started at $(date)" > "$LOGFILE"

for f in dataset/Libri1h/train/*.s16; do
  name=$(basename "$f" .s16)
  echo -n "Processing $name ... " | tee -a "$LOGFILE"
  start=$(date +%s)

  ./dump_data -train "$f" dataset/Libri1h/train_feat/"$name.f32" dataset/Libri1h/train_pcm/"$name.s16"

  end=$(date +%s)
  duration=$((end - start))
  echo "Done in ${duration}s" | tee -a "$LOGFILE"
done

echo "Dumping completed at $(date)" >> "$LOGFILE"