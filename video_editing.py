"""Use this to merge the videos output by `render_dataset.py`.
Note: datasets should have a `x.tif` and `gt.tif` in the dir specified in `DATASETS`
CWD-independent. GPU may help H265 codec.
"""

import sys

sys.path.append(".")
from src import *

# ARGS ========================================
dataset = "oabf_astro"
CODEC = "libx265"
BITRATE = 5_000
DURATION = 10
ZOOM = 1

OUT_DIR = mkdir(f"renderings/{dataset}")
Editor().alternate(
    {
        "DeepCAD-RT": f"renderings/{dataset}/y.mp4",
        "Freq2Clean": f"renderings/{dataset}/fft.mp4",
    },
    OUT_DIR / f"{ZOOM}x.mp4",
    codec=CODEC,
    bitrate=BITRATE,
    duration=DURATION,
    speed=1,
    zoom=ZOOM,
)
