#!/bin/bash

#SBATCH --job-name=CNNT
#SBATCH --output=train.log
#SBATCH --error=train.log
#SBATCH --time=24:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --mem=256G

cd /leonardo_scratch/fast/IscrC_MACRO/CalciumImagingDenoising/7-cnnt

srun python CNNT_Microscopy/main.py \
--h5files \
    "dataset/Denoising/Actin_Training.h5" \
    "dataset/Denoising/ER_Training.h5" \
    "dataset/Denoising/Golji_Training.h5" \
    "dataset/Denoising/Lysosome_Training.h5" \
    "dataset/Denoising/Matrix_Mitochondria_Training.h5" \
    "dataset/Denoising/Microtubule_Training.h5" \
    "dataset/Denoising/Tomm20_Mitochondria_Training.h5" \
--test_case "dataset/Denoising/Demo.h5" \
--ratio 100 0 0 \
--global_lr 0.0001 \
--num_epochs 300 --batch_size 4 \
--save_cycle 5 \
--time 16 --width 128 160 --height 128 160 \
--loss ssim --loss_weights 1.0 \
--wandb_entity valerio-morelli \
--run_name backbone_training_run_0 \
--run_notes backbone_default_model_300_epochs