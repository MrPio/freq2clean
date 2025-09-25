"""Use this to render the 4 stages of a given dataset.
Note: datasets should have a `x.tif` and `gt.tif` in the dir specified in `DATASETS`
CWD-independent. GPU may help H265 codec.
"""

import sys

sys.path.append(".")
from src import Recording, DATASETS, tqdm, np, Path

# ARGS ========================================
CODEC = "libx265"
BITRATE = 10_000
FPS = 30
MAX_FRAMES = FPS * 20

dataset = "oabf_vpm"
y_path = "dataset/oabf/vpm/y.tiff"
fft_path = "3-video_fusion/results/oabf_vpm/ftt_oabf_vpm_frame1000_alphas0.85deepcad_theirs.npy"
# =============================================

OUT_DIR = Path(f"renderings/{dataset}")
OUT_DIR.mkdir(exist_ok=True, parents=True)
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
    rec.render(OUT_DIR / f"{k}.mp4", codec=CODEC, bitrate=BITRATE, silent=False, fps=FPS)
