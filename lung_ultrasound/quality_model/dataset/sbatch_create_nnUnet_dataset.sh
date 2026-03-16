#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --error=create_dataset.err
#SBATCH --output=create_dataset.out
#SBATCH --account=IscrC_FouGenAI

SCRIPTS_PATH="/leonardo_work/IscrC_FouGenAI/Angelo/Lung_ultrasound/Lung_Ultrasound/lung_ultrasound/quality_model/dataset"
DATASET_PATH="/leonardo_work/IscrC_FouGenAI/Angelo/Lung_ultrasound/Extrapolates_frames_v2"

python ${SCRIPTS_PATH}/create_nnUnet_dataset.py \
    --dataset_path $DATASET_PATH\
    --splitting "splitting.json"\
    --name_dataset "only_lung"