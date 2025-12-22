# %%
from datetime import datetime
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sys
from scipy.ndimage import uniform_filter1d
from freq2clean import Freq2Clean
from correlation_loss import CorrelationLoss

sys.path.append("..")
from src import *

# %% Args
cfg = {
    # Data
    "denoiser_name": "deepcad",
    "denoiser_variant": "_15",
    "dataset_name": "synthetic",
    "frequency_transform": "dft1d",
    # Training
    "patch_t": 512,
    "patch_xy": 128,
    "overlap": 0.3,
    "avg_win": 1024,  # doesnt affect the training that much
    "batch_size": 1,
    "epochs": 50,
    "learning_rate": 0.05,
    "save_checkpoints": True,
    "max_frames": 3000,
    "weight_decay": 1e-5,
    "weight_clamp01": True,
    # Loss
    "w_l1": 1e-1,
    "w_mse": 1e-0,
    "w_corr": 2e-1,
    "w_reg": 5e-2,
}
device = "cuda" if torch.cuda.is_available() else "cpu"
cprint("Using device", f"cyan:{device}")
cprint("Using frequency transform", f"red:{cfg['frequency_transform']}")

# %% Dataset
clog("Loading and Normalizing dataset...")
x = Recording(
    f"dataset/{cfg['dataset_name']}/x.tif",
    max_frames=cfg["max_frames"],
    norm=True,
).np
y = Recording(
    f"dataset/{cfg['dataset_name']}/{cfg['denoiser_name']}{cfg['denoiser_variant']}.tif",
    max_frames=cfg["max_frames"],
    norm=True,
).np
gt = Recording(
    f"dataset/{cfg['dataset_name']}/gt.tif",
    max_frames=cfg["max_frames"],
    norm=True,
).np

clog("Computing temporal averaged video...")
x_bar = uniform_filter1d(x, size=cfg["avg_win"], axis=0, mode="reflect")
# x_avg = np.mean(x, axis=0)

# %% Batching
clog("Subdividing dataset in overlapping spatiotemporal patches...")
T, W, H = x.shape
patch_t = cfg["patch_t"]
patch_xy = cfg["patch_xy"]
overlap = cfg["overlap"]
stride_t = int(patch_t * (1 - cfg["overlap"]))
stride_xy = int(patch_xy * (1 - cfg["overlap"]))
nt = (T - patch_t) // stride_t + 1
nx = (W - patch_xy) // stride_xy + 1
ny = (H - patch_xy) // stride_xy + 1

# Generate indices
t_idx = np.arange(nt)[:, None] * stride_t + np.arange(patch_t)[None, :]
x_idx = np.arange(nx)[:, None] * stride_xy + np.arange(patch_xy)[None, :]
y_idx = np.arange(ny)[:, None] * stride_xy + np.arange(patch_xy)[None, :]

# Extract patches
patches_x = []
patches_xbar = []
patches_y = []
patches_gt = []
for ti in tqdm(t_idx, leave=False):
    for xi in tqdm(x_idx, leave=False):
        for yi in tqdm(y_idx, leave=False):
            patches_x.append(x[np.ix_(ti, xi, yi)])
            patches_xbar.append(x_bar[np.ix_(ti, xi, yi)])
            patches_y.append(y[np.ix_(ti, xi, yi)])
            patches_gt.append(gt[np.ix_(ti, xi, yi)])

x_p = torch.from_numpy(np.stack(patches_x)).float()
xbar_p = torch.from_numpy(np.stack(patches_xbar)).float()
y_p = torch.from_numpy(np.stack(patches_y)).float()
gt_p = torch.from_numpy(np.stack(patches_gt)).float()

dataset = TensorDataset(x_p[1:], xbar_p[1:], y_p[1:], gt_p[1:])
validset = (x_p[0], xbar_p[0], y_p[0], gt_p[0])
dataloader = DataLoader(
    dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True
)
del x, x_bar, y, gt
patch_shape = (cfg["patch_t"], cfg["patch_xy"], cfg["patch_xy"])
cprint(f"Dataset has", len(dataloader), "samples, each of shape", patch_shape)

# %% Model
clog("Loading Freq2Clean...")
# avg_frame = torch.tensor(x_avg).to(device)
model = Freq2Clean(shape=(patch_shape), mode=cfg["frequency_transform"]).to(device)
optimizer = optim.Adam(
    model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
)
ssim3d = SSIM3D()
l1 = nn.L1Loss().cuda()
mse = nn.MSELoss().cuda()
correlation = CorrelationLoss().cuda()

# Reg mask
coords = [torch.linspace(0, 1, d) for d in model.mask.shape]
grid = torch.meshgrid(*coords, indexing="ij")
reg_mask = ((sum(grid) / len(grid)) ** 2).to(device)

df = pd.DataFrame(
    columns=["step", "epoch", "l1", "l2", "corr", "reg", "loss"]
).set_index("step")
now_date = datetime.now().strftime("%Y%m%d-%H%M")
suffx = (
    f"{now_date}-{cfg['dataset_name']}_{cfg['denoiser_name']}{cfg['denoiser_variant']}"
)
base_dir = mkdir(f"trainings/{suffx}")
snaps_dir = mkdir(base_dir / "snaps", clear=True)
weights_dir = mkdir(base_dir / "weights", clear=True)
pth_dir = mkdir(base_dir / "pth", clear=True)
metrics_path = base_dir / "metrics.csv"
loss_trend_path = base_dir / "loss_trend.png"
json.dump(cfg, open(base_dir / "cfg.json", "w"))

# %% Training
last_loss = 0
clog("Starting Freq2Clean train...")
for epoch in (pbar := trange(cfg["epochs"])):
    # Validate
    x, x_bar, y, gt = validset
    model.eval()
    f2c = model(y.to(device).unsqueeze(0), x_bar.to(device).unsqueeze(0))
    loss = -ssim3d(
        f2c[:, ::8].unsqueeze(0), gt[::8].to(device).unsqueeze(0).unsqueeze(0)
    )
    imshow(
        {
            "Noisy": x[-1].numpy(),
            "DeepCAD-RT": y[-1].cpu().numpy(),
            "AVG": x_bar[-1].cpu().numpy(),
            f"Freq2Clean (SSIM={loss.item():.4f})": f2c[0][-1].cpu().detach().numpy(),
            "Ground Truth": gt[-1].cpu().numpy(),
        },
        path=snaps_dir / f"{epoch}.png",
        size=5,
        vrange=(0, 1),
    )
    # Save parameters plot
    mask_plot_path = weights_dir / f"{epoch}.png"
    if cfg["frequency_transform"] == "dft1d":
        fig = plt.figure(figsize=(20, 6))
        pd.Series(model.mask.cpu().detach().numpy()).plot()
        fig.savefig(mask_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    elif cfg["frequency_transform"] == "dct3d":
        vidshow(
            model.mask.cpu().detach().numpy()[:128, :128, :128],
            path=mask_plot_path,
            alpha=0.75,
        )
        vidshow(
            model.mask.cpu().detach().numpy()[::8, ::8, ::8],
            path=mask_plot_path.with_name(f"{epoch}_2.png"),
        )

    # Train
    for i, (x, x_bar, y, gt) in enumerate(dataloader):
        model.train()
        optimizer.zero_grad()
        f2c = model(y.to(device), x_bar.to(device))

        # loss_ssim = -ssim3d(f2c[:, ::4].unsqueeze(1), gt[:, ::4].to(device).unsqueeze(1))
        loss_l1 = cfg["w_l1"] * l1(f2c, gt.to(device))
        loss_mse = cfg["w_mse"] * mse(f2c, gt.to(device))
        loss_correlation = cfg["w_corr"] * correlation(f2c, y.to(device))
        loss_reg = cfg["w_reg"] * torch.mean(model.mask * reg_mask)
        loss = loss_l1 + loss_mse + loss_reg + loss_correlation

        loss.backward()
        optimizer.step()
        if cfg["weight_clamp01"]:
            with torch.no_grad():
                model.mask.clamp_(0.0, 1.0)

        pbar.set_description(
            f"[{i+1}/{len(dataloader)}] Loss={loss.item():.4f}{'🔼'if loss.item()>last_loss else '🔽'} [L1={loss_l1.item():.4f},L2={loss_mse.item():.4f},CORR={loss_correlation.item():.4f},REG={loss_reg.item():.4f}]"
        )
        df.loc[i + epoch * len(dataloader)] = [
            epoch,
            loss_l1.item(),
            loss_mse.item(),
            loss_correlation.item(),
            loss_reg.item(),
            loss.item(),
        ]
        df.to_csv(metrics_path)
        last_loss = loss.item()

    if cfg["save_checkpoints"]:
        torch.save(model.state_dict(), pth_dir / f"{epoch}.pt")
    fig, ax = plt.subplots(figsize=(16, 8))
    df.drop(columns="epoch").plot(ax=ax)
    ax.set_yscale("log")
    fig.savefig(loss_trend_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
