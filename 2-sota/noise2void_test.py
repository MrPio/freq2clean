import sys
from pathlib import Path
from careamics import CAREamist
import numpy as np
import torch
import tifffile as tiff
from tqdm import trange

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(str(FILE_DIR.parent))
from src import Recording, clog, DATASETS, imshow

dataset = "synthetic"
checkpoint = "n2v_synthetic_frames3000_t64_ep10-v2"
max_frames = None
max_render_frames = None
patch_t = 64
patch_xy = 256

# Init
torch.set_float32_matmul_precision("high")
WORK_DIR = Path("n2v")
OUT_DIR = WORK_DIR / "output"
OUT_DIR.mkdir(exist_ok=True, parents=True)

clog("red:Loading Dataset...")
metadata = DATASETS[dataset]
x, gt = (Recording(_, max_frames=max_frames) for _ in [metadata.x, metadata.gt])

clog("cyan:Loading checkpoint")
engine = CAREamist(WORK_DIR / f"checkpoints/{checkpoint}.ckpt", WORK_DIR, enable_progress_bar=True)
y = np.empty_like(x.np)
T, H, W = x.np.shape
for t in trange(0, T, patch_t, leave=False):
    for i in trange(0, H, patch_xy, leave=False):
        for j in trange(0, W, patch_xy, leave=False):
            y[t : t + patch_t, i : i + patch_xy, j : j + patch_xy] = engine.predict(
                x.np[t : t + patch_t, i : i + patch_xy, j : j + patch_xy]
            )[0][0, 0]

tiff.imwrite(OUT_DIR / f"{checkpoint}.tiff", y)

clog("yellow:Rendering...")
for zoom in [1, 3]:
    imshow(
        [_[i] for _ in [x.np, y, gt.np] for i in [0, 100, -1]],
        zoom=zoom,
        cols=3,
        size=8,
        path=OUT_DIR / f"{checkpoint}_{zoom}x.png",
    )
    if max_render_frames:
        Recording(y[:max_render_frames]).render(OUT_DIR / f"{checkpoint}_{zoom}x.mp4", codec="libx264")
