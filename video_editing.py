"""Use this to merge the videos output by `render_dataset.py`.
Note: datasets should have a `x.tif` and `gt.tif` in the dir specified in `DATASETS`
CWD-independent. GPU may help H265 codec.
"""

import sys

sys.path.append(".")
from src import *

# ARGS ========================================
datasets = ["oabf_resonant_neuro"]#["oabf_astro", "oabf_vpm", "oabf_resonant_neuro"]
ZOOMS = [1, 2.5]
CODEC = "libx265"
BITRATE = 5_000
DURATION = 20

for dataset in datasets:
    for zoom in ZOOMS:
        OUT_DIR = mkdir(f"renderings/{dataset}")
        Editor().alternate(
            {
                "DeepCAD-RT": f"renderings/{dataset}/y.mp4",
                "Freq2Clean": f"renderings/{dataset}/fft.mp4",
            },
            OUT_DIR / f"alternated_{zoom}x.mp4",
            codec=CODEC,
            bitrate=BITRATE,
            duration=DURATION,
            speed=1,
            zoom=zoom,
        )
