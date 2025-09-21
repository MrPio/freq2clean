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
MAX_FRAMES = FPS * 30

dataset = "neutrophils"
y_path = "2-denoise/results/DataFolderIs_neutrophils_202509211945_ModelFolderIs_neutrophils_202509211933/E_10_Iter_1200/xf_E_10_Iter_1200_output.tif"
fft_path = "6-video_fusion/results/neutrophils/ftt_neutrophils_frame3000_alphas1_60-150.npy"
# =============================================

metadata = DATASETS[dataset]
x_path = metadata.dir / "x.tif"
gt_path = metadata.dir / "gt.tif"
recs = {
    k: Recording(path, max_frames=MAX_FRAMES)
    for k, path in {
        "x": x_path,
        "y": y_path,
        "gt": gt_path,
    }.items()
}
recs["fft"] = Recording(np.load(fft_path)[:MAX_FRAMES])

for k, rec in tqdm(recs.items()):
    rec.render(f"{k}.mp4", codec=CODEC, bitrate=BITRATE, silent=False, fps=FPS)
