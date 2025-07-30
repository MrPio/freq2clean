from deepcad.test_collection import testing_class
from deepcad.movie_display import display
from deepcad.utils import get_first_filename, download_demo
from pathlib import Path

dataset = Path("../dataset/dati_per_erzelli/mini2p_astro/good_example/2024Feb06-007")
sample_dir = Path("../dataset/sample")
dataset = dataset / "motion_corrected"
dataset_small = sample_dir / "motion_corrected"
checkpoint = "motion_corrected_202507301400"

test_datasize = 600  # the number of frames to be tested
GPU = "0"  # the index of GPU you will use for computation (e.g. '0', '0,1', '0,1,2')
patch_xy = 150  # the width and height of 3D patches
patch_t = 150  # the time dimension (frames) of 3D patches
overlap_factor = 0.6  # the overlap factor between two adjacent patches
# Since the receptive field of 3D-Unet is ~90, seamless stitching requires an overlap (patch_xyt*overlap_factor）of at least 90 pixels.
num_workers = 0  # if you use Windows system, set this to 0.

# Setup some parameters for result visualization during the test (optional)
visualize_images_per_epoch = False  # whether to display inference performance after each epoch
save_test_images_per_epoch = True  # whether to save inference image after each epoch in pth path

test_dict = {
    # dataset dependent parameters
    "patch_x": patch_xy,  # the width of 3D patches
    "patch_y": patch_xy,  # the height of 3D patches
    "patch_t": patch_t,  # the time dimension (frames) of 3D patches
    "overlap_factor": overlap_factor,  # overlap factor,
    "scale_factor": 1,  # the factor for image intensity scaling
    "test_datasize": test_datasize,  # the number of frames to be tested
    "datasets_path": str(dataset),  # folder containing all files to be tested
    "pth_dir": "./pth",  # pth file root path
    "denoise_model": checkpoint,  # A folder containing all models to be tested
    "output_dir": "./results",  # result file root path
    # network related parameters
    "fmap": 16,  # number of feature maps
    "GPU": GPU,  # GPU index
    "num_workers": num_workers,  # if you use Windows system, set this to 0.
    "visualize_images_per_epoch": visualize_images_per_epoch,  # whether to display inference performance after each epoch
    "save_test_images_per_epoch": save_test_images_per_epoch,  # whether to save inference image after each epoch in pth path
}

tc = testing_class(test_dict)

tc.run()