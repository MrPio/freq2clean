__author__ = "Valerio Morelli"
__email__ = "valeriomorelli50@gmail.com"
__license__ = "Apache-2.0"


from .recording import Recording
from .utils import imshow, cprint, log, get_gpu_memory, tensor2pil,pil_stack
from .networks import DiffDenoiseUNet, DeepCADImprovementUNet
from .dataset import Dataset2PM
import matplotlib.pyplot as plt
from pathlib import Path

_ROOT_DIR = Path(__file__).parents[1]
DATASETS = {
    "oabf_astro": _ROOT_DIR / "dataset/oabf/astro",
    "oabf_vpm": _ROOT_DIR / "dataset/oabf/vpm",
    "oabf_resonant_neuro": _ROOT_DIR / "dataset/oabf/resonant_neuro",
}
SAMPLE_DIR = _ROOT_DIR / "dataset/sample"

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "axes.grid": True,
        "grid.linestyle": "--",
    }
)
