from typing import Literal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sys
from csbdeep.utils import normalize
from scipy.ndimage import uniform_filter1d

sys.path.append("..")
from src import *


def match(src, trg):
    low = torch.quantile(src.flatten(), 0.25)
    high = torch.quantile(src.flatten(), 0.75)
    t_low = torch.quantile(trg.flatten(), 0.25)
    t_high = torch.quantile(trg.flatten(), 0.75)
    denom = (high - low).clamp(min=1e-6)
    a = (t_high - t_low) / denom
    b = t_low - a * low
    return torch.clamp(a * src + b, 0, 1)


class Freq2Clean(nn.Module):
    def __init__(self, frames, alphas=None):
        super().__init__()
        self.frames = frames
        if not alphas:
            alphas = [0.85] + [0.0] * (frames - 1)
        self.alphas = nn.Parameter(torch.tensor(alphas))

    def forward(self, y_hat: torch.Tensor, y_bar: torch.Tensor) -> torch.Tensor:
        Y_hat = torch.fft.fft(y_hat, dim=1)
        Y_bar = torch.fft.fft(y_bar, dim=1)

        Y_hat_abs = torch.abs(Y_hat)
        Y_bar_abs = torch.abs(Y_bar)
        Y_hat_angle = torch.angle(Y_hat)
        # alpha_factor = torch.clamp(self.alphas, 0.0, 1.0) # Clamping for stability

        freq0 = match(torch.mean(y_bar, axis=1), Y_hat_abs[:, 0] / self.frames) * self.frames
        Y_bar_abs[:, 0] = freq0

        Y_abs = Y_bar_abs * self.alphas.view(1, -1, 1, 1) + Y_hat_abs * (1 - self.alphas).view(1, -1, 1, 1)
        Y = torch.polar(Y_abs, Y_hat_angle)
        return torch.fft.ifft(Y, dim=1).real


# Args
denoiser_name: Literal["deepcad", "noise2noise", "noise2void"] = "deepcad"
denoiser_variant = "_15"
dataset = "synthetic"
CUPY_AVAILABLE = False
PATCH_T = 600
AVG_WIN = 2048
BATCH_SIZE = 2
EPOCHS = 100
LEARNING_RATE = 0.0005
SAVE_SNAPS = True
METRICS_PATH = Path(f"f2c_{dataset}_metrics_{denoiser_name}.csv")
RES_DIR = mkdir(f"results/{dataset}/")
MAX_FRAMES = 3000
device = "cuda"
snapdir = mkdir("snapshots")
alphadir = mkdir("snapshots/alphas")


x = Recording(f"dataset/{dataset}/x.tif", max_frames=MAX_FRAMES).normalized
y = Recording(f"dataset/{dataset}/{denoiser_name}{denoiser_variant}.tif", max_frames=MAX_FRAMES).normalized
gt = Recording(f"dataset/{dataset}/gt.tif", max_frames=MAX_FRAMES).normalized

x_bar = uniform_filter1d(x, size=AVG_WIN, axis=0, mode="reflect")

overlap = 0.5
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

model = Freq2Clean(frames=PATCH_T).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
criterion = SSIM3D()  # nn.L1Loss()
mse = nn.MSELoss()
lambda_mse = 0.2
lambda_reg = 1e-4
init_alphas = torch.tensor([0.85] + [0.0] * (PATCH_T - 1)).to(device)

for dir in [snapdir, alphadir]:
    for file in dir.glob("*.png"):
        file.unlink()

pbar = trange(EPOCHS)
for epoch in pbar:
    # Validate
    x, x_bar, y, gt = validset
    model.eval()
    f2c = model(y.to(device).unsqueeze(0), x_bar.to(device).unsqueeze(0))
    loss = -criterion(f2c.unsqueeze(0), gt.to(device).unsqueeze(0).unsqueeze(0))
    imshow(
        {
            "Noisy": x[-1].numpy(),
            "DeepCAD-RT": y[-1].cpu().numpy(),
            "AVG": x_bar[-1].cpu().numpy(),
            f"Freq2Clean ({loss.item():.5f})": f2c[0][-1].cpu().detach().numpy(),
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

    # Train
    for i, (x, x_bar, y, gt) in enumerate(dataloader):
        model.train()
        optimizer.zero_grad()
        f2c = model(y.to(device), x_bar.to(device))

        # loss = -psnr(f2c, gt.to(device), data_range=1.0)
        # loss = -criterion(f2c.unsqueeze(1), gt.to(device).unsqueeze(1))
        loss_ssim = -criterion(f2c.unsqueeze(1), gt.to(device).unsqueeze(1))
        loss_mse = mse(f2c, gt.to(device))
        loss_reg = lambda_reg * torch.mean((model.alphas - init_alphas) ** 2)
        loss = loss_ssim + lambda_mse * loss_mse + loss_reg

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.alphas.clamp_(0.0, 1.0)

        pbar.set_description(f"[{i}/{len(dataloader)}] Loss={loss.item():.5f}")
