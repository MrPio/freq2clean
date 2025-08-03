#!/bin/bash

#SBATCH --job-name=trainer
#SBATCH --output=trainer.log
#SBATCH --error=trainer.log
#SBATCH --time=08:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --mem=24G

cd /leonardo_scratch/fast/IscrC_MACRO/CalciumImagingDenoising/3-diffusion_training

accelerate config
accelerate launch trainer.py