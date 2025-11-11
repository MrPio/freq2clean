# %%
from typing import Literal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sys
from scipy.ndimage import uniform_filter1d
from freq2clean import Freq2Clean

sys.path.append("..")
from src import *

# %% Args
denoiser_name: Literal["deepcad", "noise2noise", "noise2void"] = "deepcad"
denoiser_variant = "_150"
dataset = "synthetic"
CUPY_AVAILABLE = False
PATCH_T = 600
AVG_WIN = 2024
BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 0.0075
SAVE_SNAPS = True
METRICS_PATH = Path(f"f2c_{dataset}_metrics_{denoiser_name}.csv")
RES_DIR = mkdir(f"results/{dataset}/")
MAX_FRAMES = 3000
device = "cuda"
snapdir = mkdir("snapshots")
alphadir = mkdir("snapshots/alphas")
checkpointdir = mkdir("snapshots/checkpoints")

# %%
x = Recording(f"dataset/{dataset}/x.tif", max_frames=MAX_FRAMES).normalized
y = Recording(f"dataset/{dataset}/{denoiser_name}{denoiser_variant}.tif", max_frames=MAX_FRAMES).normalized
gt = Recording(f"dataset/{dataset}/gt.tif", max_frames=MAX_FRAMES).normalized
x_bar = uniform_filter1d(x, size=AVG_WIN, axis=0, mode="reflect")

# %%
overlap = 0.7
new_patcht = PATCH_T * (1 - overlap)
discard = math.ceil(1 / (1 - overlap) - 1)
idx = (np.arange(x.shape[0] // new_patcht)[:-discard, None] * new_patcht + np.arange(PATCH_T)).astype(int)
x = torch.from_numpy(x[idx]).float()
x_bar = torch.from_numpy(x_bar[idx]).float()
y = torch.from_numpy(y[idx]).float()
gt = torch.from_numpy(gt[idx]).float()
dataset = TensorDataset(x[1:], x_bar[1:], y[1:], gt[1:])
validset = (x[0], x_bar[0], y[0], gt[0])
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
del x, x_bar, y, gt
cprint(f"Dataset has", len(idx), "samples.")

# %%
model = Freq2Clean(frames=PATCH_T).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
ssim3d = SSIM3D()
l1 = nn.L1Loss().cuda()
mse = nn.MSELoss().cuda()
lambda_reg = 1e-4
init_alphas = model.alphas.detach().clone()

for dir in [snapdir, alphadir, checkpointdir]:
    for file in dir.glob("*.png"):
        file.unlink()

pbar = trange(EPOCHS)
for epoch in pbar:
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
        path=snapdir / f"{epoch}.png",
        size=5,
        vrange=(0, 1),
    )
    fig = plt.figure(figsize=(20, 6))
    pd.Series(model.alphas.cpu().detach().numpy()).plot()
    fig.savefig(alphadir / f"{epoch}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    torch.save(model.state_dict(), checkpointdir / f"{epoch}.pt")

    # Train
    for i, (x, x_bar, y, gt) in enumerate(dataloader):
        model.train()
        optimizer.zero_grad()
        f2c = model(y.to(device), x_bar.to(device))

        # loss_ssim = -ssim3d(f2c[:, ::4].unsqueeze(1), gt[:, ::4].to(device).unsqueeze(1))
        loss_l1 = l1(f2c, gt.to(device))
        loss_mse = mse(f2c, gt.to(device))
        loss_reg = lambda_reg * torch.mean((model.alphas - init_alphas) ** 2)
        loss = 0.5 * loss_l1 + 0.5 * loss_mse + loss_reg

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.alphas.clamp_(0.0, 1.0)

        pbar.set_description(f"[{i}/{len(dataloader)}] Loss={loss.item():.5f}")
