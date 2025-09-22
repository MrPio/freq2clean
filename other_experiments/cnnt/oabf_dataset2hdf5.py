"""Converts each TIFF pair (RAW, GT) into an HDF5 top-level group that contains the `noisy_im` and `clean_im` datasets.

No GPU is required. CWD-dependent. No support for parrallelism.
"""

import sys
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path("..").resolve()))
from src import *


def dataset2hdf5(dataset: str, window, time, patches, out):
    metadata = DATASETS[dataset]
    x, y = (Recording(metadata.dir / f"{_}.tiff", max_frames=None) for _ in ["x", "y"])
    x.normalize(metadata.max_val_x)
    y.normalize(metadata.max_val_y)
    x = x.avg(window)
    with h5py.File(out / f"{dataset}.h5", "w") as h5f:
        for i in tqdm(list(range(0, x.frames - time * window, x.frames // patches))[:patches]):
            x_p, y_p = (_.np[i : i + time * window : window] for _ in [x, y])
            g = h5f.create_group(str(i))
            for ds, data in {"noisy_im": x_p, "clean_im": y_p}.items():
                g.create_dataset(ds, data=data, compression="lzf")


dataset_dir = Path("dataset")
dataset2hdf5("oabf_astro", window=4, time=16, patches=15, out=dataset_dir)
