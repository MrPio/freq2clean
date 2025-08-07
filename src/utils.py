import math
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import logging
from termcolor import colored
import subprocess as sp
from torchvision.transforms import ToPILImage
import torch
from csbdeep.utils import normalize

logging.basicConfig(
    level="INFO",
    format="%(message)s",
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
        if isinstance(v, tuple):
            v = f"blue:{v}"
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
    size=4,
    cols: int = None,
    cmap="grey",
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
        axes[i].imshow(img, cmap=cmap)
        if titles:
            axes[i].set_title(titles[i])
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def tensor2pil(tensor: torch.Tensor):
    img = tensor.cpu().detach().numpy()
    img -= np.min(img)
    img /= np.max(img)
    img = np.clip(normalize(img, 1, 99.5), min=0, max=1)
    return Image.fromarray(np.uint8(img[0] * 255), mode="L")


def pil_stack(imgs, horizontally=True):
    """
    Stacks a list of PIL Images horizontally.

    Args:
        imgs (List[Image.Image]): List of PIL Image objects.
        bg_color (tuple): Background color (for any padding), e.g. (0,0,0) for black.

    Returns:
        Image.Image: New PIL image with all inputs concatenated side by side.
    """
    imgs = list(imgs)
    widths, heights = zip(*(im.size for im in imgs))
    total_width = (sum if horizontally else max)(widths)
    max_height = (max if horizontally else sum)(heights)
    new_img = Image.new(imgs[0].mode, (total_width, max_height))
    offset = 0
    for im in imgs:
        new_img.paste(im, (offset, 0) if horizontally else (0, offset))
        offset += im.width if horizontally else im.height

    return new_img

def gauss1D(size, mu=None, sigma=None):
    if not sigma:
        sigma = size / 6 # ~99% of Gaussian
    if not mu:
        mu = size // 2
    gaussian_weights = np.exp(-0.5 * ((np.arange(size) - mu) / sigma) ** 2)
    return gaussian_weights / gaussian_weights.sum()