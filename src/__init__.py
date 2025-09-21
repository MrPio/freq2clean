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
from .metrics.ssim import ssim, ssim3d
from .metrics.psnr import psnr, psnr3d
from .metrics.ale import ale
from .metrics.pearson import pearson3d

# Utils
from .utils import imshow, cprint, get_gpu_memory, get_cpu_memory, print_mem, tensor2pil, pil_stack, gauss1D, elapsed

# Configuration
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "axes.grid": True,
        "grid.linestyle": "--",
    }
)
# Used libraries
import pandas as pd
from tqdm import tqdm
import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False
cprint("Cupy Available=", CUPY_AVAILABLE)

# Constants
from pathlib import Path

_ROOT_DIR = Path(__file__).parents[1]
