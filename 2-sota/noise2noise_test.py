import sys
from pathlib import Path
from careamics import CAREamist
import torch
from tqdm import trange

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(str(FILE_DIR.parent))
from src import Recording, clog, cprint, DATASETS, np, imshow

dataset = "synthetic"
checkpoint = "n2n_synthetic_frames6000_t32_ep10-v2"
max_frames = None
max_render_frames = 200
pred_step = 200

# Init
torch.set_float32_matmul_precision("high")
WORK_DIR = Path("n2n")
OUT_DIR = WORK_DIR / "output" / checkpoint
OUT_DIR.mkdir(exist_ok=True, parents=True)

clog("red:Loading Dataset...")
metadata = DATASETS[dataset]
x, gt = (Recording(_, max_frames=max_frames) for _ in [metadata.x, metadata.gt])

clog("cyan:Loading checkpoint")
engine = CAREamist(WORK_DIR / f"checkpoints/{checkpoint}.ckpt", WORK_DIR, enable_progress_bar=False)
y = []
for i in trange(x.frames // pred_step):
    y.append(engine.predict(x.np[i * pred_step : (i + 1) * pred_step, :488, :488])[0][0, 0])
y = np.concatenate(y, axis=0)
np.save(OUT_DIR / f"{checkpoint}.npy", y)

clog("yellow:Rendering...")
for zoom in [1, 3]:
    imshow(
        [_[i] for _ in [x.np, y, gt.np] for i in [0, 100, -1]],
        zoom=zoom,
        cols=3,
        size=8,
        path=OUT_DIR / f"{checkpoint}_{zoom}x.png",
    )
    Recording(y[:max_render_frames]).render(OUT_DIR / f"{checkpoint}_{zoom}x.mp4", codec="libx264")
