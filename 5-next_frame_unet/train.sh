#!/bin/bash

#SBATCH --job-name=next_frame_unet
#SBATCH --output=next_frame_unet.log
#SBATCH --error=next_frame_unet.log
#SBATCH --time=16:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

cd /leonardo_scratch/fast/IscrC_MACRO/CalciumImagingDenoising/5-next_frame_unet

# accelerate config
# accelerate launch trainer.py
srun python train.py