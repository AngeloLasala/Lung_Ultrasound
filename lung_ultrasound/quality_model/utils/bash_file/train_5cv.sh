#!/usr/bin/env bash

for fold in fold_1 fold_2 fold_3 fold_4 fold_5; do
    echo "Training $fold..."
    start=$(date +%s)
 
    python -m lung_ultrasound.quality_model.tools.train --fold "$fold" --log info --keep_log
 
    end=$(date +%s)
    elapsed=$(( end - start ))
    echo "$fold completed in $(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m $(( elapsed % 60 ))s"
done