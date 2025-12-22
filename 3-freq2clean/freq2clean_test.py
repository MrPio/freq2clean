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
args = parse_args(
    {
        "checkpoint": (f.stem for f in Path("trainings").glob("*/")),
        "dataset": DATASETS.keys(),
        "denoiser": str,
        "variant": "",
        "batch_size": 1,
    }
)

# Use this to test F2C on a testset that differs from the trainset
AVG_WIN: int | None = None
SAVE_TIFF: bool = False
SKIP_NET: bool = False
SKIP_GRID: bool = True
MAX_FRAMES: int | None = None
SSIM3D_STEPS: int | None = 1  # Increase if GPU raises OOM
SSIM3D_PATCHXY: int | None = 192  # Decrease if GPU raises OOM

# %% Dataset Loading
cprint("Loading checkpoint", f"yellow:{args.checkpoint}")
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.load(open(f"trainings/{args.checkpoint}/cfg.json"))
patch_xy = cfg.get("patch_xy", 128)
out_dir = mkdir(f"results/{args.dataset}/{args.denoiser}{args.variant}")
clog(f'Loading dataset {args.dataset}, denoiser="{args.denoiser}"{args.variant}"')
metrics_path = out_dir / f"metrics.json"
metrics = json.load(metrics_path.open()) if metrics_path.exists() else {}
meta = DATASETS[args.dataset]
meta.download()

x = Recording(meta.x, max_frames=None, norm=True).np
y = Recording(
    meta.dir / f"{args.denoiser}{args.variant}.tif",
    max_frames=MAX_FRAMES,
    norm=True,
).np
gt = Recording(meta.gt, max_frames=MAX_FRAMES, norm=True).np
if "patch_xy" in cfg:
    cprint("patch_xy=", cfg["patch_xy"])
    max_y = cfg["patch_xy"] * (x.shape[1] // cfg["patch_xy"])
    max_x = cfg["patch_xy"] * (x.shape[2] // cfg["patch_xy"])
    x = x[:, :max_x, :max_y]
    y = y[:, :max_x, :max_y]
    gt = gt[:, :max_x, :max_y]

clog("Computing averaged vid/frame...")
avg_win = AVG_WIN or cfg["avg_win"]
x_bar = uniform_filter1d(x, size=avg_win, axis=0, mode="reflect")
del x
x_bar = x_bar[: y.shape[0], : y.shape[1], : y.shape[2]]
gt = gt[: y.shape[0], : y.shape[1], : y.shape[2]]

# %% Batching
clog("Batching...")
n = y.shape[0] // cfg["patch_t"]
idx = (np.arange(n)[:, None] * cfg["patch_t"] + np.arange(cfg["patch_t"])).astype(int)
x_bar = torch.from_numpy(x_bar[idx]).float()
y_ = torch.from_numpy(y[idx]).float()
testset = TensorDataset(x_bar, y_)
dataloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
del x_bar
cprint(f"Testset has", len(idx), "samples.")

# %% Model
clog("Loading model...")
checkpointdir = Path("trainings") / args.checkpoint / "pth"
ckpt = sorted(checkpointdir.glob("*.pt"), key=lambda file: int(file.stem))[-1]
cprint("cyan:Loading checkpoint", ckpt.stem)
# avg_frame = torch.tensor(x_avg).to(device)
patch_shape = (cfg["patch_t"], patch_xy, patch_xy)
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
            B, T, H, W = y.shape
            out = torch.empty_like(y)
            for i in trange(0, H, patch_xy, leave=False):
                for j in trange(0, W, patch_xy, leave=False):
                    y_p = y[:, :, i : i + patch_xy, j : j + patch_xy]
                    x_bar_p = x_bar[:, :, i : i + patch_xy, j : j + patch_xy]
                    op = model(y_p.to(device), x_bar_p.to(device)).cpu()
                    out[:, :, i : i + patch_xy, j : j + patch_xy] = op

            f2cs.append(out)

    f2cs = torch.cat(f2cs, dim=0)
    result = f2cs.reshape(-1, f2cs.shape[2], f2cs.shape[3]).numpy()
    if save_path:
        clog(f"Saving {save_path.stem}...")
        tiff.imwrite(save_path, result)
        tiff.imwrite(save_path, result)
    return result


if not SKIP_NET:
    clog("Running Freq2Clean (network) test...")
    f2c_net = run_inference(
        model,
        dataloader,
        save_path=out_dir / f"{args.checkpoint}.tiff" if SAVE_TIFF else None,
    )
    imshow(
        {f"Frame:{i}": f2c_net[i] for i in range(0, f2c_net.shape[0], 500)},
        size=8,
        cols=4,
        path=out_dir / f"{args.checkpoint}_snap.png",
    )
    # The number of frames in F2C is a multiple of the number of patches
    metrics[args.checkpoint] = {
        "psnr3d": psnr3d(gt[: f2c_net.shape[0]], f2c_net),
        "ssim3d": ssim3d(
            gt[: f2c_net.shape[0]], f2c_net, step=SSIM3D_STEPS, patch_xy=SSIM3D_PATCHXY
        ),
    }

if (k := f"grid_{model.mode}") not in metrics and not SKIP_GRID:
    clog("Running Freq2Clean (grid search) test...")
    model.initialize_mask(device=device)
    f2c_grid = run_inference(
        model, dataloader, save_path=out_dir / f"{k}.tiff" if SAVE_TIFF else None
    )
    imshow(
        {f"Frame:{i}": f2c_grid[i] for i in range(0, f2c_grid.shape[0], 500)},
        size=8,
        cols=4,
        path=out_dir / f"{k}_snap.png",
    )
    # The number of frames in F2C is a multiple of the number of patches
    metrics[k] = {
        "psnr3d": psnr3d(gt[: f2c_grid.shape[0]], f2c_grid),
        "ssim3d": ssim3d(
            gt[: f2c_grid.shape[0]],
            f2c_grid,
            step=SSIM3D_STEPS,
            patch_xy=SSIM3D_PATCHXY,
        ),
    }

if (k := args.denoiser) not in metrics:
    metrics[k] = {
        "psnr3d": psnr3d(gt, y),
        "ssim3d": ssim3d(gt, y, step=SSIM3D_STEPS, patch_xy=SSIM3D_PATCHXY),
    }
json.dump(metrics, metrics_path.open("w"), indent=4)
jprint(metrics)
