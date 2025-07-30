#!/bin/bash

#SBATCH --job-name=deepcad_train
#SBATCH --output=deepcad_train.log
#SBATCH --error=deepcad_train.log
#SBATCH --time=04:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:3
#SBATCH --mem=24G

cd /leonardo_scratch/fast/IscrC_MACRO/CalciumImagingDenoising/2-denoise

srun python deepcad_train.py