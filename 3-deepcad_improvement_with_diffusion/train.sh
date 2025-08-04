#!/bin/bash

#SBATCH --job-name=trainer1
#SBATCH --output=trainer1.log
#SBATCH --error=trainer1.log
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

cd /leonardo_scratch/fast/IscrC_MACRO/CalciumImagingDenoising/3-diffusion_training

# accelerate config
# accelerate launch trainer.py
srun python trainer.py