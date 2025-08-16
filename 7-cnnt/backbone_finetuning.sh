#!/bin/bash

#SBATCH --job-name=CNNT
#SBATCH --output=finetune.log
#SBATCH --error=finetune.log
#SBATCH --time=01:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --mem=32G

cd /leonardo_scratch/fast/IscrC_MACRO/CalciumImagingDenoising/7-cnnt

srun python CNNT_Microscopy/main.py \
--h5files \
    "dataset/Denoising/Demo.h5" \
--fine_samples 15 \
--test_case "dataset/Denoising/Demo.h5" \
--global_lr 0.000025 --skip_LSUV \
--num_epochs 30 --batch_size 2 \
--time 16 --width 128 160 --height 128 160 \
--loss ssim mse --loss_weights 1.0 0.1 \
--save_cycle 99 \
--wandb_entity valerio-morelli \
--run_name finetuning_astro --run_notes finetuning_astro_with_30_epochs \
--load_path "logs/check/08-10-2025_T19-47-09_epoch-60.pth"