__author__ = "Valerio Morelli"
__email__ = "valeriomorelli50@gmail.com"
__license__ = "Apache-2.0"


# Dataset
from .dataset.dataset import DATASETS
from .dataset.noisy_clean_dataset import NoisyCleanDataset
from .dataset.noisy_dataset import NoisyDataset

# Networks
from .networks.networks import (
    DiffDenoiseUNet,
    DeepCADImprovementUNet,
    NextFramesUNet,
    VideoEncoder,
    NextFramesUNetStacked,
)
from .networks.losses import lf_hf_tv

# Video
from .video.editor import Editor
from .video.recording import Recording

# Metrics
from .metrics.ssim3d import ssim3D
from .metrics.psnr3d import psnr3d

# Utils
from .utils import imshow, cprint, log, get_gpu_memory, tensor2pil, pil_stack, gauss1D

# Configuration
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "axes.grid": True,
        "grid.linestyle": "--",
    }
)

# Constants
from pathlib import Path

_ROOT_DIR = Path(__file__).parents[1]
