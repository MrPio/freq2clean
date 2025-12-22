import math
import os
from pathlib import Path
import random
from time import time_ns
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import logging
import psutil
from termcolor import colored
import subprocess as sp
from torchvision.transforms import ToPILImage
import torch
from csbdeep.utils import normalize
from scipy.ndimage import zoom as ndi_zoom
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from .video.editor import Editor
import matplotlib as mpl
import argparse
from collections.abc import Collection, Generator
import requests
from tqdm import tqdm

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    # , handlers=[RichHandler(markup=True, show_path=False)]
)
logger = logging.getLogger("src")

COLORS = [
    # "black",
    "blue",
    "cyan",
    "green",
    "yellow",
    "red",
    "magenta",
    # "white",
    "light_blue",
    "light_cyan",
    "light_green",
    "light_yellow",
    "light_red",
    "light_magenta",
    "dark_grey",
    "light_grey",
]
__counter = 0
__eta = time_ns()


dcad_cmap = LinearSegmentedColormap.from_list(
    "dcad",
    list(
        zip(
            [p / 255 for p, _ in Editor.GREEN_GRADIENT],
            [(c[0] / 255, c[1] / 255, c[2] / 255) for _, c in Editor.GREEN_GRADIENT],
        )
    ),
)
base_mpl_font_size = mpl.rcParams["font.size"]


def clog(*vals, sep=" "):
    if len(vals) == 1 and ":" not in vals[0]:
        vals = (f"rand:{vals[0]}", *vals[1:])
    cprint(
        *vals,
        f"light_red:[{print_mem()}]",
        f"light_yellow:[{elapsed()}s]",
        sep=sep,
        reset_counter=False,
    )


def cprint(*vals, sep=" ", reset_counter=True):
    """Log values, highlighting any prefixed by a color tag (e.g., 'red:error'). cprint stands for colored-print btw"""
    global __counter
    if reset_counter:
        __counter = 0

    def fmt(v):
        if isinstance(v, (int, float)):
            v = f"blue:{v:,}"
        if isinstance(v, tuple):
            v = f"blue:{v}"
        else:
            v = str(v)

        if v.startswith("rand:"):
            global __counter
            v = v.replace("rand:", f"{COLORS[__counter%len(COLORS)]}:")
            __counter += 1
        for c in COLORS:
            tag = f"{c}:"
            if v.startswith(tag):
                return colored(v[len(tag) :], c, attrs=["bold"])
        return v

    vals = map(fmt, vals)
    print(*vals, sep=sep)


def jprint(json):
    def pprint(
        json,
        level=0,
        tab=4,
        ck="red",
        cv=("blue", "green", "cyan", "yellow", "light_grey"),
    ):
        space = " " * tab
        if isinstance(json, dict):
            args: list = [(space * level, "{")]
            for k, v in json.items():
                _v = pprint(v, level=level + 1, ck=ck, cv=cv)
                arg = [space * (level + 1), f"{ck}:{k}", ":"]
                if isinstance(_v, str):
                    arg.append(_v)
                args.append(tuple(arg))
                if not isinstance(_v, str):
                    args.extend(_v)
            return args + [(space * level, "}")]
        elif isinstance(json, list) and (len(json) == 0 or isinstance(json[0], str)):
            return f"{cv[3]}:[" + ", ".join(json) + "]"
        elif isinstance(json, list):
            args: list = [(space * level, "[")]
            for v in json:
                _v = pprint(v, level=level + 1, ck=ck, cv=cv)
                arg = [space * (level + 1)]
                if isinstance(_v, str):
                    arg.append(_v)
                args.append(tuple(arg))
                if not isinstance(_v, str):
                    args.extend(_v)
            return args + [(space * level, "]")]
        elif isinstance(json, bool):
            return f"{cv[2]}:{json}"
        elif isinstance(json, (int, float)):
            return f"{cv[0]}:{json}"
        elif isinstance(json, str):
            return f"{cv[1]}:{json}"
        else:
            return f"{cv[-1]}:{json}"

    for args in pprint(json):
        cprint(*args)


def imshow(
    images: (
        list[Image.Image | np.ndarray | str | Path]
        | dict[str, Image.Image | np.ndarray | str | Path]
    ),
    size=4,
    dpi=150,
    cols: int = None,
    cmap=None,
    vrange=(None, None),
    zoom=1.0,
    shift=(0, 0),
    path: Path | str = None,
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
        (
            image.size[0] / image.size[1]
            if isinstance(image, (Image.Image))
            else image.shape[0] / image.shape[1]
        )
        for image in images
    )
    _, axes = plt.subplots(
        rows, cols, figsize=(cols * size, int(rows * size * max_ratio)), dpi=dpi
    )
    if rows > 1 or cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    for i, img in enumerate(images):
        axes[i].imshow(
            zoom_img(img, zoom, shift),
            cmap=cmap if cmap else dcad_cmap,
            vmin=vrange[0],
            vmax=vrange[1],
        )
        if titles:
            axes[i].set_title(titles[i])
        axes[i].axis("off")
    plt.tight_layout()
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def vidshow(
    vid, alpha=0.25, dpi=150, cmap=None, path: Path | str = None, grid=False, step=1
):
    vid = vid[::step, ::step, ::step]
    idx = np.indices(vid.shape).reshape(vid.ndim, -1).T
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        idx[:, 0],
        idx[:, 1],
        idx[:, 2],
        c=vid,
        cmap=cmap if cmap else dcad_cmap,
        s=100000 / vid.shape[0] ** 2,
        alpha=alpha,
        linewidths=0,
    )

    if not grid:
        ax.set_axis_off()
        ax.grid(False)
    else:
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_zlabel("y")
    ax.view_init(elev=30, azim=225)
    plt.tight_layout()
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def get_cpu_memory():
    process = psutil.Process(os.getpid())
    used = process.memory_info().rss / (1024**3)
    free = psutil.virtual_memory().available / (1024**3)
    return used, free


def print_mem():
    cpu_used, cpu_free = get_cpu_memory()
    return f"{cpu_used:.1f}/{cpu_free:.1f} GiB"


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
        sigma = size / 6  # ~99% of Gaussian
    if not mu:
        mu = size // 2
    gaussian_weights = np.exp(-0.5 * ((np.arange(size) - mu) / sigma) ** 2)
    return gaussian_weights / gaussian_weights.sum()


def zoom_img(x, factor: float = 1, shift: tuple[int, int] = (0, 0)):
    x = np.array(x)
    h, w = x.shape[:2]
    new_h, new_w = int(h / factor), int(w / factor)
    top, left = (h - new_h) // 2 + shift[1], (w - new_w) // 2 + shift[0]
    return x[top : top + new_h, left : left + new_w]


def elapsed():
    return (time_ns() - __eta) // 10**9


def mkdir(path: str | Path, clear=False) -> Path:
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    if clear:
        for file in path.glob("*"):
            file.unlink()
    return path


def barchart(
    data: dict[str, float],
    bounds: tuple[float, float] = None,
    yaxis=None,
    title=None,
    ysteps=None,
    color="steelblue",
    figsize=(8, 6),
    dpi=300,
    ax=None,
):
    labels = list(data.keys())
    values = list(data.values())

    if ax:
        sns.barplot(x=labels, y=values, color=color, ax=ax)
    else:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = sns.barplot(x=labels, y=values, color=color)
    if ysteps and bounds:
        ax.set_yticks(
            [
                i / ysteps * (bounds[1] - bounds[0]) + bounds[0]
                for i in range(ysteps + 1)
            ]
        )

    if bounds:
        ax.set_ylim(bounds)
    ax.set_xlabel("")
    if yaxis:
        ax.set_ylabel(yaxis)
    if title:
        ax.set_title(title)

    plt.tight_layout()


def parse_args(args: dict[str, type | list | object]):
    parser = argparse.ArgumentParser()
    for k, v in args.items():
        if isinstance(v, type):
            parser.add_argument(f"--{k}", type=v, required=True)
        elif v and isinstance(v, (Collection, Generator)):
            parser.add_argument(f"--{k}", choices=list(v), required=True)
        else:
            parser.add_argument(f"--{k}", default=v, required=False)

    return parser.parse_args()


def download(url: str | Path, path: str | Path, chunk_size: int = 8192):
    with requests.get(str(url), stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))

        with open(path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=str(path),
            leave=False,
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    bar.update(len(chunk))
