"""Use this to render the 4 stages of a given dataset.
Note: datasets should have a `x.tif` and `gt.tif` in the dir specified in `DATASETS`
CWD-independent. GPU may help H265 codec.
"""

import sys

sys.path.append(".")
from src import Recording, DATASETS, tqdm, np, mkdir

# ARGS ========================================
CODEC = "libx264"
BITRATE = 20_000
FPS = 30
MAX_FRAMES = 3000

dataset = "zebrafish"
y_path = "2-sota/deepcad_results/tif/synthetic_150-150.tif"
fft_path = "3-video_fusion/results/zebrafish/ftt_zebrafish_frame3000_alphas1_60-150.npy"

OUT_DIR = mkdir(f"renderings/{dataset}")
metadata = DATASETS[dataset]
recs = {
    k: Recording(path, max_frames=MAX_FRAMES)
    for k, path in {
        # "x": metadata.x,
        # "y": y_path,
        # "gt": metadata.gt,
    }.items()
}
recs["fft"] = Recording(np.load(fft_path)[:MAX_FRAMES])

for k, rec in tqdm(recs.items()):
    rec.render(OUT_DIR / f"{k}.mp4", codec=CODEC, bitrate=BITRATE, silent=False, fps=FPS)
