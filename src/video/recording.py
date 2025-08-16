from pathlib import Path
from typing import Literal
from matplotlib import pyplot as plt
import pandas as pd
import tifffile as tiff
from csbdeep.utils import normalize
import numpy as np
import imageio.v3 as iio
from os import PathLike
from IPython.display import Video
from tqdm import tqdm
from src.utils import gauss1D

""" 
The `Recording` is the high-level abstraction of a TIFF/TIF file.

You can access the `ndarray` by accessing the `np` property. You also can:
    - Render the recording as a MP4 video,
    - Plot the intensities distribution,
    - Rolling a sliding window with a kernel function over the footage.
"""


class Recording:
    __AGGREGATIONS = {
        "box": lambda voxel, frame, start, end: np.mean(voxel[start:end], axis=0),
        "gauss": lambda voxel, frame, start, end: np.tensordot(
            gauss1D(end - start, mu=frame - start), voxel, axes=([0], [0])
        ),
    }

    def __init__(self, video: PathLike | np.ndarray, max_frames: int = 300):
        self.np = (
            video
            if isinstance(video, np.ndarray)
            else tiff.imread(str(video), key=range(max_frames) if max_frames else None)
        )

    @property
    def frames(self) -> int:
        return self.np.shape[0]

    @property
    def normalized(self) -> np.ndarray:
        return np.clip(normalize(self.np, 0.1, 99.9), min=0, max=1)

    def normalize(self, a: int, b: int = 2**16 - 1) -> None:
        """Normalize the maximum value of the `uint16` recording.
        The range (a, b) indicates the actual and the new max values.
        Note: this is heavy because casts to `np.float64` and then back to `uint16`
        """
        self.np = (self.np / a * b)

    def save_sample(self, path: Path | str, length=300):
        tiff.imwrite(str(path), self.np[: min(self.frames, length)], dtype=np.float32)

    def render(self, path: Path | str, start=None, end=None, bitrate=4500, fps=30):
        iio.imwrite(
            uri=str(path),
            image=(self.normalized * 255).astype(np.uint8),
            fps=fps,
            codec="libx264",
            bitrate=f"{bitrate}k",
            output_params=["-loglevel", "quiet"],
        )
        return Video(path)

    def hist(self, figsize=(12, 5), bins=100):
        ax = pd.Series(self.np.flatten()).hist(figsize=figsize, bins=bins, edgecolor="white")
        ax.set_yscale("log")

    def avg_frame(self, frame: int, window=1, type: Literal["box", "gauss"] = "box") -> np.ndarray:
        start = max(0, frame - window // 2)
        end = min(self.frames, frame + window // 2)
        voxel = self.np[start:end]
        return self.__AGGREGATIONS[type](voxel, frame, start, end)

    def avg(self, window, type: Literal["box", "gauss"] = "box") -> "Recording":
        averaged = np.empty_like(self.np)
        length = self.frames
        for i in tqdm(range(length)):
            start = max(0, i - window // 2)
            end = min(length, i + window // 2)
            averaged[i] = self.__AGGREGATIONS[type](self.np, i, start, end)
        return Recording(averaged)

    def __getitem__(self, i):
        return Recording(self.np[i])
