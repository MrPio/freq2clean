__author__ = "Valerio Morelli"
__email__ = "valeriomorelli50@gmail.com"
__license__ = "Apache-2.0"


from .recording import Recording
from .utils import imshow, cprint, log
import matplotlib.pyplot as plt
from pathlib import Path

__ROOT_DIR = Path(__file__).parents[1]
DATASETS = {
    "oabf_astro": __ROOT_DIR / "dataset/oabf/astro",
    "oabf_vpm": __ROOT_DIR / "dataset/oabf/vpm",
    "oabf_resonant_neuro": __ROOT_DIR / "dataset/oabf/resonant_neuro",
}
SAMPLE_DIR = __ROOT_DIR / "dataset/sample"

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "axes.grid": True,
        "grid.linestyle": "--",
    }
)
