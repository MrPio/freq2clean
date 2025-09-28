"""Converts each TIFF pair (RAW, GT) into an HDF5 top-level group that contains the `noisy_im` and `clean_im` datasets.

No GPU is required. CWD-dependent. No support for parrallelism.
"""

import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py
import tifffile as tiff
from typing import Literal

sys.path.append("..")
from src import *


def dataset2hdf5(dir: Path, subdirs: list[Literal["Training", "Test"]] = ["Training"], gt_suffix="_decon"):
    """Expecting the following hierarcy:

    <DATASET NAME>
        ├── Test
        │   ├── `GT`
        │   └── `Raw`
        ├── Training
        │   ├── `GT`
        │   └── `Raw`
    """
    name = dir.parts[-1]
    # For each subdir (Training, Test)
    for raw_dir, gt_dir in [(dir / _ / "Raw", dir / _ / "GT") for _ in subdirs]:
        filename = f"{name}_{raw_dir.parts[-2]}.h5"
        if Path(filename).exists():
            continue
        with h5py.File(filename, "w") as h5f:
            # For each TIF file
            for raw in tqdm(sorted(raw_dir.glob("*.tif"))):
                gt = gt_dir / f"{raw.stem.replace('5%','80%').replace('C1-','C2-').replace('raw','gt')}{gt_suffix}.tif"
                g = h5f.create_group(raw.stem)
                for ds, data in {"noisy_im": raw, "clean_im": gt}.items():
                    g.create_dataset(
                        ds,
                        data=tiff.imread(data).astype(np.float32),
                        compression="lzf",
                    )


dataset_dir = Path("dataset/Denoising/")
for subdir in tqdm(sorted(dataset_dir.glob("*"))):
    dataset2hdf5(subdir, subdirs=["Test"])
