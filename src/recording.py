from pathlib import Path
import tifffile as tiff
from csbdeep.utils import normalize
import numpy as np
import imageio.v3 as iio
from os import PathLike


class Recording:
    def __init__(self, video: PathLike | np.ndarray):
        self.video = video if isinstance(video, np.ndarray) else tiff.imread(str(video))
        self.video = self.video.astype(np.float64) / np.max(self.video)

    @property
    def frames(self) -> int:
        return self.video.shape[0]

    @property
    def normalized(self) -> np.ndarray:
        return np.clip(normalize(self.video, 0.5, 99.5), max=1)

    def save_sample(self, path: Path | str, length=500):
        tiff.imwrite(str(path), self.video[:length], dtype=np.float64)

    def render(self, path: Path | str, start=None, end=None, bitrate=5000, fps=30):
        iio.imwrite(
            uri=str(path),
            image=self.normalized,
            fps=fps,
            codec="libx264",
            bitrate=f"{bitrate}k",
            output_params=["-loglevel", "quiet"],
        )
