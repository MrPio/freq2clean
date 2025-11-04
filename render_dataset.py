"""Use this to render the 4 stages of a given dataset.
Note: datasets should have a `x.tif` and `gt.tif` in the dir specified in `DATASETS`
CWD-independent. GPU may help H265 codec.
"""

import sys
from skimage.exposure import match_histograms

sys.path.append(".")
from src import *

# ARGS ========================================
CODEC = "libx264"
BITRATE = 20_000
FPS = 30
MAX_FRAMES = 1000

dataset = "oabf_resonant_neuro"
meta = DATASETS[dataset]
y_path = meta.dir / "y.tiff"
fft_path = f"3-video_fusion/results/{dataset}/ftt_oabf_resonant_neuro_frame3000_alphas0.85deepcad_theirs.npy"
OUT_DIR = mkdir(f"renderings/{dataset}")

recs = {
    k: Recording(path, max_frames=MAX_FRAMES)
    for k, path in {
        "x": meta.x,
        "y": y_path,
        # "gt": meta.gt,
    }.items()
}
recs["fft"] = Recording(np.load(fft_path)[:MAX_FRAMES])

# Remove high intensities to prevent dark video
recs["y"].np[recs["y"].np > 2**16 - 1_000] = 0
recs["fft"].np[recs["fft"].np > 2**16 - 1_000] = 0

cprint(f"blue:Matching hist...")
recs["fft"].np = match_histograms(recs["fft"].np, recs["y"].np)

for k, rec in tqdm(recs.items()):
    rec.render(OUT_DIR / f"{k}.mp4", codec=CODEC, bitrate=BITRATE, silent=False, fps=FPS)
