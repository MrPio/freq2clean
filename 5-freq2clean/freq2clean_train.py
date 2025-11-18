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

sys.path.append("..")
from src import *


# %%
class CorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        B, T, W, H = x.shape
        N = W * H

        x2 = x.reshape(B, T, N)
        y2 = y.reshape(B, T, N)

        xm = x2.mean(1, keepdim=True)
        ym = y2.mean(1, keepdim=True)
        xs = x2.std(1, unbiased=False, keepdim=True)
        ys = y2.std(1, unbiased=False, keepdim=True)

        xz = (x2 - xm) / torch.max(torch.tensor(1e-6), xs)
        yz = (y2 - ym) / torch.max(torch.tensor(1e-6), ys)

        corr = (xz * yz).mean(1)  # (B, N)
        mean_corr = corr.mean()  # scalar

        return 1.0 - mean_corr


# %% Args
cfg = {
    # Data
    "denoiser_name": "deepcad",
    "denoiser_variant": "_15",
    "dataset_name": "synthetic",
    # Training
    "patch_t": 3000,
    "overlap": 0.8,
    "avg_win": 1024,
    "batch_size": 1,
    "epochs": 30,
    "learning_rate": 0.0075,
    "save_snaps": True,
    "max_frames": 6000,
    "weight_decay": 1e-5,
    "alpha_clamp01": True,
    # Loss
    "w_l1": 1e-1,
    "w_mse": 1e-0,
    "w_corr": 2e-1,
    "w_reg": 5e-2,
}
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
x = Recording(f"dataset/{cfg["dataset_name"]}/x.tif", max_frames=cfg["max_frames"]).normalized
y = Recording(
    f"dataset/{cfg["dataset_name"]}/{cfg["denoiser_name"]}{cfg["denoiser_variant"]}.tif", max_frames=cfg["max_frames"]
).normalized
gt = Recording(f"dataset/{cfg["dataset_name"]}/gt.tif", max_frames=cfg["max_frames"]).normalized
x_bar = uniform_filter1d(x, size=cfg["avg_win"], axis=0, mode="reflect")
x_avg = np.mean(x.np, axis=0)

# %%
new_patcht = cfg["patch_t"] * (1 - cfg["overlap"])
discard = math.ceil(1 / (1 - cfg["overlap"]) - 1)
idx = (np.arange(x.shape[0] // new_patcht)[:-discard, None] * new_patcht + np.arange(cfg["patch_t"])).astype(int)
x = torch.from_numpy(x[idx]).float()
x_bar = torch.from_numpy(x_bar[idx]).float()
y = torch.from_numpy(y[idx]).float()
gt = torch.from_numpy(gt[idx]).float()
dataset = TensorDataset(x[1:], x_bar[1:], y[1:], gt[1:])
validset = (x[0], x_bar[0], y[0], gt[0])
dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
del x, x_bar, y, gt
cprint(f"Dataset has", len(idx), "samples.")

# %%
model = Freq2Clean(num_frames=cfg["patch_t"]).to(device)
optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
ssim3d = SSIM3D()
l1 = nn.L1Loss().cuda()
mse = nn.MSELoss().cuda()
correlation = CorrelationLoss().cuda()
init_alphas = model.alphas.detach().clone()
df = pd.DataFrame(columns=["step", "epoch", "l1", "l2", "corr", "reg", "loss"]).set_index("step")

suffx = (
    f"{datetime.now().strftime("%Y%m%d-%H%M")}-{cfg['dataset_name']}_{cfg['denoiser_name']}{cfg['denoiser_variant']}"
)
base_dir = mkdir(f"trainings/{suffx}")
snaps_dir = mkdir(base_dir / "snaps", clear=True)
weights_dir = mkdir(base_dir / "weights", clear=True)
pth_dir = mkdir(base_dir / "pth", clear=True)
metrics_path = base_dir / "metrics.csv"
loss_trend_path = base_dir / "loss_trend.png"
json.dump(cfg, open(base_dir / "cfg.json", "w"))

last_loss = 0
for epoch in (pbar := trange(cfg["epochs"])):
    # Validate
    x, x_bar, y, gt = validset
    model.eval()
    f2c = model(y.to(device).unsqueeze(0), x_bar.to(device).unsqueeze(0))
    loss = -ssim3d(f2c[:, ::4].unsqueeze(0), gt[::4].to(device).unsqueeze(0).unsqueeze(0))
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
    fig = plt.figure(figsize=(20, 6))
    pd.Series(model.alphas.cpu().detach().numpy()).plot()
    fig.savefig(weights_dir / f"{epoch}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Train
    for i, (x, x_bar, y, gt) in enumerate(dataloader):
        model.train()
        optimizer.zero_grad()
        f2c = model(y.to(device), x_bar.to(device))

        # loss_ssim = -ssim3d(f2c[:, ::4].unsqueeze(1), gt[:, ::4].to(device).unsqueeze(1))
        loss_l1 = cfg["w_l1"] * l1(f2c, gt.to(device))
        loss_mse = cfg["w_mse"] * mse(f2c, gt.to(device))
        loss_correlation = cfg["w_corr"] * correlation(f2c, y.to(device))
        loss_reg = cfg["w_reg"] * torch.mean((model.alphas - init_alphas) ** 2)
        loss = loss_l1 + loss_mse + loss_reg + loss_correlation

        loss.backward()
        optimizer.step()
        if cfg["alpha_clamp01"]:
            with torch.no_grad():
                model.alphas.clamp_(0.0, 1.0)

        pbar.set_description(
            f"[{i}/{len(dataloader)}] Loss={loss.item():.4f}{'🔼'if loss.item()>last_loss else '🔽'} [L1={loss_l1.item():.4f},L2={loss_mse.item():.4f},CORR={loss_correlation.item():.4f},REG={loss_reg.item():.4f}]"
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

    torch.save(model.state_dict(), pth_dir / f"{epoch}.pt")
    fig, ax = plt.subplots(figsize=(16, 8))
    df.drop(columns="epoch").plot(ax=ax)
    ax.set_yscale("log")
    fig.savefig(loss_trend_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
