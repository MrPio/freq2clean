from pathlib import Path
import torch
from deepcad.train_collection import training_class
from deepcad.test_collection import testing_class
import sys

sys.path.append("..")
from src import *

dataset_name = "mouse_neuronal_populations"
meta = DATASETS[dataset_name]
patch_t = 15
SKIP_DEEPCAD = False

if not SKIP_DEEPCAD:
    training_class(
        {
            "patch_x": 150,  # the width of 3D patches
            "patch_y": 150,  # the height of 3D patches
            "patch_t": patch_t,  # the time dimension (frames) of 3D patches
            "overlap_factor": 0.6,  # overlap factor
            "scale_factor": 1,  # the factor for image intensity scaling
            "select_img_num": 6000,  # select the number of frames used for training
            "train_datasets_size": 1200,  # datasets size for training (how many 3D patches)
            # "max_frames": 6_000,
            "datasets_path": f"dataset/{dataset_name}_x",  # folder containing files for training
            "pth_dir": "./deepcad_pth",  # the path for pth file and result images
            # network related parameters
            "n_epochs": 10,  # the number of training epochs
            "lr": 0.00005,  # learning rate
            "b1": 0.5,  # Adam: bata1
            "b2": 0.999,  # Adam: bata2
            "fmap": 16,  # model complexity
            "GPU": ",".join(map(str, range(torch.cuda.device_count()))),  # GPU index
            "num_workers": 0,  # if you use Windows system, set this to 0.
            "visualize_images_per_epoch": False,  # whether to show result images after each epoch
            "save_test_images_per_epoch": False,  # whether to save result images after each epoch
            # "UNet_type": "ResidualUNet3D",
        }
    ).run()

    pth_dir = sorted(Path("deepcad_pth/").glob("*/"), key=lambda folder: folder.stem.split("_")[-1])[-1]
    print("Pth_dir=", pth_dir)
    testing_class(
        {
            "patch_x": 150,  # the width of 3D patches
            "patch_y": 150,  # the height of 3D patches
            "patch_t": 150,  # the time dimension (frames) of 3D patches
            "overlap_factor": 0.6,  # overlap factor,
            "scale_factor": 1,  # the factor for image intensity scaling
            "test_datasize": meta.shape[0],  # the number of frames to be tested
            "datasets_path": f"dataset/{dataset_name}_x",  # folder containing all files to be tested
            "pth_dir": "./deepcad_pth",  # pth file root path
            "denoise_model": pth_dir.stem,  # A folder containing all models to be tested
            "output_dir": "./deepcad_results",  # result file root path
            # network related parameters
            "fmap": 16,  # number of feature maps
            "GPU": ",".join(map(str, range(torch.cuda.device_count()))),  # GPU index
            "num_workers": 0,  # if you use Windows system, set this to 0.
            "visualize_images_per_epoch": False,  # whether to display inference performance after each epoch
            "save_test_images_per_epoch": True,  # whether to save inference image after each epoch in pth path
        }
    ).run()

    deepcad_tiff_path = sorted(
        Path("deepcad_results/").glob("**/*.tif"), key=lambda folder: folder.stem.split("_")[-1]
    )[-1]
    print("deepcad_tiff_path=", deepcad_tiff_path)
