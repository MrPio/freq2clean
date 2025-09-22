#!/bin/bash

#SBATCH --job-name=deepcad_train_residual
#SBATCH --output=deepcad_train_residual.log
#SBATCH --error=deepcad_train_residual.log
#SBATCH --time=03:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --mem=32G

cd /leonardo_scratch/fast/IscrC_MACRO/CalciumImagingDenoising/2-sota

srun python deepcad_train.py