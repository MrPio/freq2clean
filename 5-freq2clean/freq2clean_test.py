# %% Import
from typing import Literal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sys
from csbdeep.utils import normalize
from scipy.ndimage import uniform_filter1d
from freq2clean import Freq2Clean
import json
import tifffile as tiff

sys.path.append("..")
from src import *

# %% Args
BATCH_SIZE = 1
SELECTED_TRAINING = "20251118-1205-synthetic_deepcad_15"
GRID_SEARCH_VERSION = True
DENOISER_VARIANT: str | None = None

# %% Dataset Loading
cprint("Loading checkpoint", f"yellow:{SELECTED_TRAINING}", "In Grid Search mode" if GRID_SEARCH_VERSION else "")
clog("Loading data...")
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.load(open(f"trainings/{SELECTED_TRAINING}/cfg.json"))
x = Recording(f"dataset/{cfg['dataset_name']}/x.tif", max_frames=None)
y_ = Recording(
    f"dataset/{cfg['dataset_name']}/{cfg['denoiser_name']}{DENOISER_VARIANT or cfg['denoiser_variant']}.tif",
    max_frames=None,
)
gt_ = Recording(f"dataset/{cfg['dataset_name']}/gt.tif", max_frames=None)

# %% Normalization
clog("Normalizing data...")
x = x.normalized
y_ = y_.normalized
gt_ = gt_.normalized

clog("Computing averaged vid/frame...")
x_bar = uniform_filter1d(x, size=cfg["avg_win"], axis=0, mode="reflect")
x_avg = np.mean(x, axis=0)

# %% Batching
clog("Batching...")
n = x.shape[0] // cfg["patch_t"]
idx = (np.arange(n)[:, None] * cfg["patch_t"] + np.arange(cfg["patch_t"])).astype(int)
x = torch.from_numpy(x[idx]).float()
x_bar = torch.from_numpy(x_bar[idx]).float()
y = torch.from_numpy(y_[idx]).float()
gt = torch.from_numpy(gt_[idx]).float()
testset = TensorDataset(x, x_bar, y, gt)
dataloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
del x, x_bar  # , y_, gt_
cprint(f"Testset has", len(idx), "samples.")

# %% Model
clog("Loading model...")
checkpointdir = Path("trainings") / SELECTED_TRAINING / "pth"
ckpt = sorted(checkpointdir.glob("*.pt"), key=lambda file: int(file.stem))[-1]
cprint("cyan:Loading checkpoint", ckpt.stem)
avg_frame = torch.tensor(x_avg).to(device)
model = Freq2Clean(num_frames=cfg["patch_t"], avg_frame=avg_frame)
model.to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

# %% Test
clog("Running test...")
f2cs = []
with torch.no_grad():
    if GRID_SEARCH_VERSION:
        model.alphas[0] = 0.85
        model.alphas[1:] = 0

    for x, x_bar, y, gt in tqdm(dataloader):
        out = model(y.to(device), x_bar.to(device))
        f2cs.append(out.cpu())

f2cs = torch.cat(f2cs, dim=0)
f2c = f2cs.reshape(-1, f2cs.shape[2], f2cs.shape[3]).numpy()

# %% Saving
clog("Saving TIFF...")
tiff_dir = mkdir(f"results/{cfg['dataset_name']}/{cfg['denoiser_name']}{DENOISER_VARIANT or cfg['denoiser_variant']}")
tiff.imwrite(tiff_dir / f"{SELECTED_TRAINING}{'_grid' if GRID_SEARCH_VERSION else ''}.tiff", f2c)
