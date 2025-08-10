"""
Train DeepCAD RT on OABF dataset

Usage: srun --mem=24G --gres=gpu:3 --time=00:30:00 --partition=boost_usr_prod --qos=boost_qos_dbg python depcad_train.py

OOM: `select_img_num` and `train_datasets_size` only affect RAM requirement, not VRAM
"""

import torch
from deepcad.train_collection import training_class

training_class(
    {
        "patch_x": 150,  # the width of 3D patches
        "patch_y": 150,  # the height of 3D patches
        "patch_t": 150,  # the time dimension (frames) of 3D patches
        "overlap_factor": 0.4,  # overlap factor
        "scale_factor": 1,  # the factor for image intensity scaling
        # "select_img_num": 500,  # select the number of frames used for training (use all frames by default)
        "train_datasets_size": 1500,  # datasets size for training (how many 3D patches)
        "datasets_path": "dataset/astro",  # folder containing files for training
        "pth_dir": "./pth",  # the path for pth file and result images
        # network related parameters
        "n_epochs": 10,  # the number of training epochs
        "lr": 0.00005,  # learning rate
        "b1": 0.5,  # Adam: bata1
        "b2": 0.999,  # Adam: bata2
        "fmap": 16,  # model complexity
        "GPU": ",".join(map(str, range(torch.cuda.device_count()))),  # GPU index
        "num_workers": 0,  # if you use Windows system, set this to 0.
        "visualize_images_per_epoch": False,  # whether to show result images after each epoch
        "save_test_images_per_epoch": True,  # whether to save result images after each epoch
        "max_frames": 8_000,
        # "UNet_type": "ResidualUNet3D",
    }
).run()
