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
from multiprocessing import Pool
from scipy.ndimage import uniform_filter1d

""" 
The `Recording` is the high-level abstraction of a TIFF/TIF file.

You can access the `ndarray` by accessing the `np` property. You also can:
    - Render the recording as a MP4 video,
    - Plot the intensities distribution,
    - Rolling a sliding window with a kernel function over the footage.
"""


class Recording:
    __AGGREGATIONS = {
        "box": lambda voxel, frame, start, end: np.mean(voxel, axis=0),
        "gauss": lambda voxel, frame, start, end: np.tensordot(
            gauss1D(end - start, mu=frame - start), voxel, axes=([0], [0])
        ),
    }

    def __init__(self, video: PathLike | np.ndarray, max_frames: int | None = 300):
        self.np = (
            video
            if isinstance(video, np.ndarray)
            else (
                np.load(str(video))[:max_frames]
                if str(video).endswith(".npy")
                else tiff.imread(
                    str(video), key=range(max_frames) if max_frames else None
                )
            )
        )

    @property
    def frames(self) -> int:
        return self.np.shape[0]

    @property
    def normalized(self) -> np.ndarray:
        return np.clip(normalize(self.np, 0.25, 99.9), 0, 1)

    def normalize(self, a: int, b: int = 2**16 - 1) -> None:
        """Normalize the maximum value of the `uint16` recording.
        The range (a, b) indicates the actual and the new max values.
        Note: this is heavy because casts to `np.float64` and then back to `uint16`
        """
        self.np = self.np / a * b

    def save(self, path: Path | str, max_frames=None):
        tiff.imwrite(str(path), self.np[:max_frames], dtype=self.np.dtype)

    def render(
        self,
        path: Path | str,
        start=None,
        end=None,
        bitrate=4000,
        fps=30,
        codec="libx265",
        silent=True,
    ):
        iio.imwrite(
            uri=str(path),
            image=(self.normalized[start:end] * 255).astype(np.uint8),
            fps=fps,
            codec=codec,
            bitrate=f"{bitrate}k",
            output_params=["-loglevel", "quiet"] if silent else [],
        )
        return Video(path)

    def hist(self, figsize=(12, 5), bins=100):
        ax = pd.Series(self.np.flatten()).hist(
            figsize=figsize, bins=bins, edgecolor="white"
        )
        ax.set_yscale("log")

    def avg_frame(
        self, frame: int, window=1, type: Literal["box", "gauss"] = "box"
    ) -> np.ndarray:
        if window == 1:
            return self.np[frame]
        start = max(0, frame - window // 2)
        end = min(self.frames, frame + window // 2)
        return self.__AGGREGATIONS[type](self.np[start:end], frame, start, end)

    def avg(self, window, type: Literal["box", "gauss"] = "box") -> "Recording":
        averaged = np.empty_like(self.np)
        length = self.frames
        for i in tqdm(range(length), desc="Averaging frames..."):
            start = max(0, i - window // 2)
            end = min(length, i + window // 2)
            averaged[i] = self.__AGGREGATIONS[type](self.np[start:end], i, start, end)
        return Recording(averaged)

    def avg_fast(self, window) -> np.ndarray:
        if window <= 1:
            return self.np
        return uniform_filter1d(self.np, size=window, axis=0, mode="reflect")

    def __getitem__(self, i):
        return Recording(self.np[i])
