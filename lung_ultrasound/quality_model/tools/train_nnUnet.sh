#!/bin/bash
# ============================================================
# Full nnUNet pipeline: preprocess -> splitting -> train
# Usage: bash run_nnunet.sh
# ============================================================

set -e  # stop on error

# ============================================================
# CONFIGURATION - edit these variables
# ============================================================

DATASET_ID=1                          # e.g. 1 for Dataset001_NAME
DATASET_NAME="Dataset001_only_lung"    # must match folder name in nnUNet_raw

SCRIPTS_PATH="/home/angelo/Documenti/Lung_Ultrasound/lung_ultrasound/quality_model/dataset" # path to the folder of create_nn

ORIGINAL_DATASET_PATH="/media/angelo/PortableSSD/Assistant_Researcher/Predict/OpenPOCUS/Extrapolates_frames_v2"
SPLITTING_FILE="splitting.json"

NNUNET_RAW="/media/angelo/PortableSSD/Assistant_Researcher/Predict/OpenPOCUS/nnUNet_raw"
NNUNET_PREPROCESSED="/media/angelo/PortableSSD/Assistant_Researcher/Predict/OpenPOCUS/nnUnet_preprocessed"
NNUNET_RESULTS="/media/angelo/PortableSSD/Assistant_Researcher/Predict/OpenPOCUS/results/Extrapolates_frames_v2/nnUnet"

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

python ${SCRIPTS_PATH}/create_nnUnet_split.py \
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
        $FOLD\

    echo "Fold ${FOLD} completed."
done

echo ""
echo "============================================================"
echo "Pipeline completed successfully!"
echo "Results saved in: ${NNUNET_RESULTS}/${DATASET_NAME}"
echo "============================================================"