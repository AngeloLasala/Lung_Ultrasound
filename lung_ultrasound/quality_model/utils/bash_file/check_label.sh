#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --partition=boost_usr_prod
#SBATCH --error=check_labels.err
#SBATCH --output=check_labels.out
#SBATCH --account=IscrC_FouGenAI

python -c "
import numpy as np
import SimpleITK as sitk
from pathlib import Path

label_dir = '/leonardo_work/IscrC_FouGenAI/Angelo/Lung_ultrasound/nnUNet_raw/Dataset001_only_lung/labelsTr'
files = sorted(Path(label_dir).glob('*'))[:10]
for f in files:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(f)))
    print(f.name, '-> unique values:', np.unique(arr), '| shape:', arr.shape)
"