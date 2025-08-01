__author__ = "Valerio Morelli"
__email__ = "valeriomorelli50@gmail.com"
__license__ = "Apache-2.0"


from .recording import Recording
from .utils import imshow, cprint, log
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
