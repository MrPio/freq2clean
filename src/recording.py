from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import tifffile as tiff
from csbdeep.utils import normalize
import numpy as np
import imageio.v3 as iio
from os import PathLike
from IPython.display import Video


class Recording:
    def __init__(self, video: PathLike | np.ndarray, max_frames: int = 300):
        self.np = (
            video
            if isinstance(video, np.ndarray)
            else tiff.imread(str(video), key=range(max_frames) if max_frames else None)
        )
        self.np = self.np.astype(np.float32)

    @property
    def frames(self) -> int:
        return self.np.shape[0]

    @property
    def normalized(self) -> np.ndarray:
        return np.clip(normalize(self.np, 1, 99.5), min=0, max=1)

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

    def avg(self, frame: int, window=1) -> np.ndarray:
        start = max(0, frame - window // 2)
        end = min(self.frames, frame + (window - (frame - start)))
        return np.mean(self.np[start:end], axis=0)

    def __getitem__(self, i):
        return Recording(self.np[i])
