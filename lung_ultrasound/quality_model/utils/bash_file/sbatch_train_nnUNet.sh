#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=23:00:00              # time limits: 23 hours
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=nnUnet.err      # standard error file
#SBATCH --output=nnUnet.out     # standard output file
#SBATCH --account=IscrC_FouGenAI     # account name

set -e  # stop on error

# ============================================================
# CONFIGURATION - edit these variables
# ============================================================

DATASET_ID=1                           # e.g. 1 for Dataset001_NAME
DATASET_NAME="Dataset001_only_lung"    # must match folder name in nnUNet_raw

SCRIPTS_PATH="/leonardo_work/IscrC_FouGenAI/Angelo/Lung_ultrasound/Lung_Ultrasound/lung_ultrasound/quality_model/dataset/create_nnUnet_split.py"    # full path to the script (standalone)

ORIGINAL_DATASET_PATH="/leonardo_work/IscrC_FouGenAI/Angelo/Lung_ultrasound/Extrapolates_frames_v2"
SPLITTING_FILE="splitting.json"

NNUNET_RAW="/leonardo_work/IscrC_FouGenAI/Angelo/Lung_ultrasound/nnUNet_raw"
NNUNET_PREPROCESSED="/leonardo_work/IscrC_FouGenAI/Angelo/Lung_ultrasound/nnUNet_preprocessed"
NNUNET_RESULTS="/leonardo_work/IscrC_FouGenAI/Angelo/Lung_ultrasound/results/Extrapolates_frames_v2/nnUNet"

CONFIG="2d"
NUM_FOLDS=5

# ============================================================
# ENVIRONMENT VARIABLES (required by nnUNet)
# ============================================================

export nnUNet_raw=$NNUNET_RAW
export nnUNet_preprocessed=$NNUNET_PREPROCESSED
export nnUNet_results=$NNUNET_RESULTS

# ============================================================
# STEP 1 - Plan and preprocess
# ============================================================

echo "============================================================"
echo "STEP 1: plan_and_preprocess Dataset${DATASET_ID}"
echo "============================================================"

nnUNetv2_plan_and_preprocess \
    -d $DATASET_ID \
    --verify_dataset_integrity

# ============================================================
# STEP 2 - Generate custom splits_final.json
# ============================================================

echo "============================================================"
echo "STEP 2: Generate custom splits_final.json"
echo "============================================================"

python $SCRIPTS_PATH \
    --dataset_path $ORIGINAL_DATASET_PATH \
    --splitting $SPLITTING_FILE \
    --nnunet_raw ${NNUNET_RAW}/${DATASET_NAME} \
    --nnunet_preprocessed ${NNUNET_PREPROCESSED}/${DATASET_NAME}

# ============================================================
# STEP 3 - Train all folds sequentially
# ============================================================

echo "============================================================"
echo "STEP 3: Training ${CONFIG} - all ${NUM_FOLDS} folds"
echo "============================================================"

for FOLD in $(seq 0 $((NUM_FOLDS - 1))); do
    echo ""
    echo "------------------------------------------------------------"
    echo "Training fold ${FOLD} / $((NUM_FOLDS - 1))"
    echo "------------------------------------------------------------"

    nnUNetv2_train \
        $DATASET_ID \
        $CONFIG \
        $FOLD

    echo "Fold ${FOLD} completed."
done

echo ""
echo "============================================================"
echo "Pipeline completed successfully!"
echo "Results saved in: ${NNUNET_RESULTS}/${DATASET_NAME}"
echo "============================================================"