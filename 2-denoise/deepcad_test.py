import torch
from deepcad.test_collection import testing_class

testing_class(
    {
        "patch_x": 150,  # the width of 3D patches
        "patch_y": 150,  # the height of 3D patches
        "patch_t": 150,  # the time dimension (frames) of 3D patches
        "overlap_factor": 0.6,  # overlap factor,
        "scale_factor": 1,  # the factor for image intensity scaling
        "test_datasize": 6000,  # the number of frames to be tested
        "datasets_path": "dataset/synthetic",  # folder containing all files to be tested
        "pth_dir": "./pth",  # pth file root path
        "denoise_model": "synthetic_202509191151",  # A folder containing all models to be tested
        "output_dir": "./results",  # result file root path
        # network related parameters
        "fmap": 16,  # number of feature maps
        "GPU": ",".join(map(str, range(torch.cuda.device_count()))),  # GPU index
        "num_workers": 0,  # if you use Windows system, set this to 0.
        "visualize_images_per_epoch": False,  # whether to display inference performance after each epoch
        "save_test_images_per_epoch": True,  # whether to save inference image after each epoch in pth path
    }
).run()
