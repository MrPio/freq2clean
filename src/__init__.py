__author__ = "Valerio Morelli"
__email__ = "valeriomorelli50@gmail.com"
__license__ = "Apache-2.0"


# Networks
from .networks import DiffDenoiseUNet, DeepCADImprovementUNet, NextFramesUNet, VideoEncoder

# Dataset
from .dataset.dataset import DATASETS
from .dataset.noisy_clean_dataset import NoisyCleanDataset
from .dataset.noisy_dataset import NoisyDataset

# Utils
from .recording import Recording
from .utils import imshow, cprint, log, get_gpu_memory, tensor2pil, pil_stack

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
