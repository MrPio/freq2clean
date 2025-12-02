# %% Import
import torch
from torch.utils.data import TensorDataset, DataLoader
import sys
from scipy.ndimage import uniform_filter1d
from freq2clean import Freq2Clean
import json
import tifffile as tiff

sys.path.append("..")
from src import *

# %% Args
BATCH_SIZE = 1

# dft1d (15) --> 20251118-1205-synthetic_deepcad_15
# dft1d (150) --> 20251118-1221-synthetic_deepcad_150
# dct3d (15) --> 20251202-0738-synthetic_deepcad_15
# dct3d (150) --> 20251202-0835-synthetic_deepcad_150
SELECTED_TRAINING = sys.argv[2] if len(sys.argv) > 2 else "20251202-0835-synthetic_deepcad_150"

# Use this to test F2C on a testset that differs from the trainset
DATASET_NAME: str | None = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
DENOISER_VARIANT: str | None = None
AVG_WIN: int | None = None
GT_VARIANT: str | None = None
SAVE_TIFF: bool = False
SKIP_NET: bool = False
FORCE_GRID: bool = True

# %% Dataset Loading
cprint("Loading checkpoint", f"yellow:{SELECTED_TRAINING}")
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.load(open(f"trainings/{SELECTED_TRAINING}/cfg.json"))
dataset_name = DATASET_NAME or cfg["dataset_name"]
variant = DENOISER_VARIANT or cfg["denoiser_variant"]
out_dir = mkdir(f"results/{dataset_name}/{cfg['denoiser_name']}{variant}{AVG_WIN or ''}")
clog(f'Loading dataset {dataset_name}, y_variant="{variant}", gt_variant="{GT_VARIANT}"...')
metrics_path = out_dir / f"metrics.json"
metrics = json.load(metrics_path.open()) if metrics_path.exists() else {}

x = Recording(
    f"dataset/{dataset_name}/x.tif",
    max_frames=None,
    norm=True,
).np
y = Recording(
    f"dataset/{dataset_name}/{cfg['denoiser_name']}{variant}.tif",
    max_frames=None,
    norm=True,
).np
gt = Recording(
    f"dataset/{dataset_name}/gt{GT_VARIANT or ''}.tif",
    max_frames=None,
    norm=True,
).np

clog("Computing averaged vid/frame...")
avg_win = AVG_WIN or cfg["avg_win"]
x_bar = uniform_filter1d(x, size=avg_win, axis=0, mode="reflect")
# x_avg = np.mean(x, axis=0)

# %% Batching
clog("Batching...")
n = x.shape[0] // cfg["patch_t"]
idx = (np.arange(n)[:, None] * cfg["patch_t"] + np.arange(cfg["patch_t"])).astype(int)
x_bar = torch.from_numpy(x_bar[idx]).float()
y_ = torch.from_numpy(y[idx]).float()
patch_shape = x_bar[0].shape
testset = TensorDataset(x_bar, y_)
dataloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
del x, x_bar
cprint(f"Testset has", len(idx), "samples.")

# %% Model
clog("Loading model...")
checkpointdir = Path("trainings") / SELECTED_TRAINING / "pth"
ckpt = sorted(checkpointdir.glob("*.pt"), key=lambda file: int(file.stem))[-1]
cprint("cyan:Loading checkpoint", ckpt.stem)
# avg_frame = torch.tensor(x_avg).to(device)
model = Freq2Clean(shape=patch_shape, mode=cfg.get("frequency_transform", "dft1d"))
cprint("Selected Frequency Transform=", f"green:{model.mode}")
model.to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()


# %% Test
def run_inference(model: Freq2Clean, dataloader, save_path=None) -> np.ndarray:
    f2cs = []
    with torch.no_grad():
        for x_bar, y in tqdm(dataloader):
            out = model(y.to(device), x_bar.to(device))
            f2cs.append(out.cpu())

    f2cs = torch.cat(f2cs, dim=0)
    result = f2cs.reshape(-1, f2cs.shape[2], f2cs.shape[3]).numpy()
    if save_path:
        clog(f"Saving {save_path.stem}...")
        tiff.imwrite(save_path, result)
    return result


if not SKIP_NET:
    clog("Running Freq2Clean (network) test...")
    f2c_net = run_inference(model, dataloader, save_path=out_dir / f"{SELECTED_TRAINING}.tiff" if SAVE_TIFF else None)
    imshow(
        {f"Frame:{i}": f2c_net[i] for i in range(0, f2c_net.shape[0], 500)},
        size=8,
        cols=4,
        path=out_dir / f"{SELECTED_TRAINING}_snap.png",
    )
    gt = gt[: f2c_net.shape[0]]  # The number of frames in F2C is a multiple of the number of patches
    metrics[SELECTED_TRAINING] = {
        "psnr3d": psnr3d(gt, f2c_net),
        "ssim3d": ssim3d(gt, f2c_net),
    }

if (k := f"grid_{model.mode}") not in metrics or FORCE_GRID:
    clog("Running Freq2Clean (grid search) test...")
    model.initialize_mask(device=device)
    f2c_grid = run_inference(model, dataloader, save_path=out_dir / f"{k}.tiff" if SAVE_TIFF else None)
    imshow(
        {f"Frame:{i}": f2c_grid[i] for i in range(0, f2c_grid.shape[0], 500)},
        size=8,
        cols=4,
        path=out_dir / f"{k}_snap.png",
    )
    gt = gt[: f2c_grid.shape[0]]  # The number of frames in F2C is a multiple of the number of patches
    metrics[k] = {
        "psnr3d": psnr3d(gt, f2c_grid),
        "ssim3d": ssim3d(gt, f2c_grid),
    }

if (k := "deepcad") not in metrics:
    metrics[k] = {
        "psnr3d": psnr3d(gt, y),
        "ssim3d": ssim3d(gt, y),
    }
json.dump(metrics, metrics_path.open("w"), indent=4)
jprint(metrics)

