#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=23:00:00              # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=inference.err              # standard error file
#SBATCH --output=inference.out             # standard output file
#SBATCH --account=IscrC_FouGenAI     # account name


python -m lung_ultrasound.quality_model.tools.train --splitting "splitting_inference.json" --keep_log
 
