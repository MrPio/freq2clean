from typing import Literal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sys
from csbdeep.utils import normalize

sys.path.append("..")
from src import *


class Freq2Clean(nn.Module):
    def __init__(self, frames, alphas=None):
        super().__init__()

        if not alphas:
            alphas = [1] + [0.01] * (frames - 1)
        self.alphas = nn.Parameter(torch.tensor(alphas))

    def forward(self, y_hat: torch.Tensor, y_bar: torch.Tensor) -> torch.Tensor:
        Y_hat = torch.fft.fft(y_hat, dim=1)
        Y_bar = torch.fft.fft(y_bar, dim=1)

        Y_hat_abs = torch.abs(Y_hat)
        Y_bar_abs = torch.abs(Y_bar)
        Y_hat_angle = torch.angle(Y_hat)
        # alpha_factor = torch.clamp(self.alphas, 0.0, 1.0) # Clamping for stability

        Y_abs = Y_bar_abs * self.alphas.view(1, -1, 1, 1) + Y_hat_abs * (1 - self.alphas).view(1, -1, 1, 1)
        Y = torch.polar(Y_abs, Y_hat_angle)
        return torch.fft.ifft(Y, dim=1).real

# TODO: OVERLAP Avg l1
# Args 

denoiser_name: Literal["deepcad", "noise2noise", "noise2void"] = "deepcad"
denoiser_variant = "_15"
dataset = "synthetic"
CUPY_AVAILABLE = False
PATCH_T = 600
BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 0.01
SAVE_SNAPS = True
device = "cuda"
snapdir = mkdir("snapshots")


def normalize(t):
    t_min = t.min()
    t_max = t.max()
    print(t_min, t_max)
    return 2 * (t - t_min) / (t_max - t_min) - 1


def psnr(y_pred: torch.Tensor, y_true: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = torch.mean((y_pred - y_true) ** 2)
    if mse < 1e-10:
        return torch.tensor(999.0, dtype=y_pred.dtype, device=y_pred.device)
    return 10.0 * torch.log10((data_range**2) / mse)


if __name__ == "__main__":
    METRICS_PATH = Path(f"f2c_{dataset}_metrics_{denoiser_name}.csv")
    RES_DIR = mkdir(f"results/{dataset}/")

    clog("blue:Loading Freq2Clean...")
    model = Freq2Clean(frames=PATCH_T).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion=nn.MSELoss()

    clog("red:Loading Normalized dataset...")
    x = Recording(f"dataset/{dataset}/x.tif", max_frames=None).normalized
    y = Recording(f"dataset/{dataset}/{denoiser_name}{denoiser_variant}.tif", max_frames=None).normalized
    gt = Recording(f"dataset/{dataset}/gt.tif", max_frames=None).normalized

    clog("green:Splitting data in batches")
    idx = np.arange(x.shape[0] // PATCH_T)[:, None] * PATCH_T + np.arange(PATCH_T)
    x = torch.from_numpy(x[idx]).float()
    y = torch.from_numpy(y[idx]).float()
    gt = torch.from_numpy(gt[idx]).float()

    clog("green:Loading DataLoader...")
    dataset = TensorDataset(x, y, gt)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    del x, y, gt

    clog("yellow:Running training...")
    for epoch in trange(EPOCHS):
        model.train()
        for i, (x, y, gt) in enumerate(dataloader):
            optimizer.zero_grad()
            f2c = model(y.to(device), x.to(device))
            # loss = -psnr(f2c, gt.to(device), data_range=1.0)
            loss=criterion(f2c, gt.to(device))
            
            loss.backward()
            optimizer.step()

            cprint(f"[{i}/{len(dataloader)}] Loss=", f"cyan:{loss.item():.3f}")
            if SAVE_SNAPS:
                imshow(
                    {
                        "Noisy": x[0][-1].cpu().numpy(),
                        "DeepCAD-RT": y[0][-1].cpu().numpy(),
                        "Freq2Clean": f2c[0][-1].cpu().detach().numpy(),
                        "Ground Truth": gt[0][-1].cpu().numpy(),
                    },
                    path=snapdir / f"{epoch}-{i}.png",
                    size=5,
                    vrange=(0, 1),
                )
        cprint("Alphas=", model.alphas.cpu().detach().numpy())
