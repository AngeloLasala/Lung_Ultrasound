#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=23:00:00              # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=train_5cv_2.err              # standard error file
#SBATCH --output=train_5cv_2.out             # standard output file
#SBATCH --account=IscrC_FouGenAI     # account name

TIMESTAMP=$(date +'%d-%m-%Y_%H-%M')   # shared timestamp across all folds

for fold in fold_1 fold_2 fold_3 fold_4 fold_5; do
    echo "Training $fold..."
    start=$(date +%s)
 
    python -m lung_ultrasound.quality_model.tools.train --fold "$fold" --timestamp "$TIMESTAMP"  --splitting "splitting_ext_plax.json" --log info --keep_log
 
    end=$(date +%s)
    elapsed=$(( end - start ))
    echo "$fold completed in $(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m $(( elapsed % 60 ))s"
done