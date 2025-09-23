"""Use this to render the 4 stages of a given dataset.
Note: datasets should have a `x.tif` and `gt.tif` in the dir specified in `DATASETS`
CWD-independent. GPU may help H265 codec.
"""

import sys

sys.path.append(".")
from src import Recording, DATASETS, tqdm, np

# ARGS ========================================
CODEC = "libx265"
BITRATE = 12_000
FPS = 30
MAX_FRAMES = FPS * 20

dataset = "synthetic"
y_path = "2-sota/n2v/output/n2v_synthetic_frames6000_t32_ep10-v2.npy"
fft_path = "3-video_fusion/results/synthetic/ftt_synthetic_frame3000_alphas0.85_32.npy"
# =============================================

metadata = DATASETS[dataset]
recs = {
    k: Recording(path, max_frames=MAX_FRAMES)
    for k, path in {
        # "x": metadata.x,
        "y": y_path,
        # "gt": metadata.gt,
    }.items()
}
recs["fft"] = Recording(np.load(fft_path)[:MAX_FRAMES])

for k, rec in tqdm(recs.items()):
    rec.render(f"{k}.mp4", codec=CODEC, bitrate=BITRATE, silent=False, fps=FPS)
