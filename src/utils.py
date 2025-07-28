import math
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import logging
from termcolor import colored

logging.basicConfig(level="INFO", format="%(message)s"
                    # , handlers=[RichHandler(markup=True, show_path=False)]
                    )
logger = logging.getLogger("src")

COLORS = [
    "black",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
    "light_grey",
    "dark_grey",
    "light_red",
    "light_green",
    "light_yellow",
    "light_blue",
    "light_magenta",
    "light_cyan",
]


def log(*vals, use_log=True):
    """Log values, highlighting any prefixed by a color tag (e.g., 'red:error')."""

    def fmt(v):
        if isinstance(v, (int, float)):
            v = f"blue:{v:,}"
        else:
            v = str(v)
        for c in COLORS:
            tag = f"{c}:"
            if v.startswith(tag):
                return f"[bold {c}]{v[len(tag):]}[/bold {c}]" if use_log else colored(v[len(tag) :], c, attrs=["bold"])
        return v

    vals = map(fmt, vals)
    if use_log:
        logger.info(" ".join(vals))
    else:
        print(*vals)


def cprint(*vals):
    log(*vals, use_log=False)

def imshow(
    images: list[Image.Image | np.ndarray | str | Path] | dict[str, Image.Image | np.ndarray | str | Path],
    size=3,
    cols: int = None,
):
    """Plot a list of PIL images in a grid

    Args:
        images (list[Image.Image]): the list of images to show
        size (int, optional): the size in inch of the images
        col (int, optional): The number of columns of the grid. Defaults to 1.
    """
    if isinstance(images, (Image.Image, str, Path, np.ndarray)):
        images = [images]
    titles = None
    if isinstance(images, dict):
        titles, images = list(images.keys()), list(images.values())
    else:
        images = list(images)
        if not images:
            return
    for i in range(len(images)):
        if not isinstance(images[i], (Image.Image, np.ndarray)):
            images[i] = Image.open(images[i])

    if not cols:
        cols = min(10, len(images))
    rows = math.ceil(len(images) / cols)
    max_ratio = max(
        (image.size[0] / image.size[1] if isinstance(image, (Image.Image)) else image.shape[0] / image.shape[1])
        for image in images
    )
    _, axes = plt.subplots(rows, cols, figsize=(cols * size, int(rows * size * max_ratio)))
    if rows > 1 or cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    for i, img in enumerate(images):
        axes[i].imshow(img)
        if titles:
            axes[i].set_title(titles[i])
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()