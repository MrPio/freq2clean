import sys
from pathlib import Path
import bm3d, bm4d
from tqdm import trange

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(str(FILE_DIR.parent))
from src import Recording, clog, cprint, DATASETS, np, imshow

dataset = "synthetic"
max_frames = 64
patch_t = 8
σ = 10
max_render_frames = 60
stage = bm3d.BM3DStages.ALL_STAGES

# Init
OUT_DIR = Path("bm4d")
OUT_DIR.mkdir(exist_ok=True, parents=True)
experiment = f"dm4d_{dataset}_frames{max_frames}_t{patch_t}_σ{σ}_stage{stage}"
cprint(f"magenta:{experiment}")

clog("red:Loading Dataset...")
meta = DATASETS[dataset]
x, gt = (Recording(_, max_frames=max_frames) for _ in [meta.x, meta.gt])

# Block Matching
x_gpu = cp.asarray(x.np)
y = []
for i in trange(x.frames // patch_t):
    y.append(bm4d.bm4d(x_gpu[i * patch_t : (i + 1) * patch_t], sigma_psd=σ, stage_arg=stage))
y = cp.concatenate(y, axis=0).get()
np.save(OUT_DIR / f"{experiment}.npy", y)

clog("yellow:Rendering...")
for zoom in [1, 3]:
    imshow(
        [_[i] for _ in [x.np, y, gt.np] for i in [0, 100, -1]],
        zoom=zoom,
        cols=3,
        size=8,
        path=OUT_DIR / f"{experiment}_{zoom}x.png",
    )
    Recording(y[:max_render_frames]).render(OUT_DIR / f"{experiment}_{zoom}x.mp4", codec="libx264")
