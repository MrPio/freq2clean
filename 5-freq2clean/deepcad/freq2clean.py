from typing import Literal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import normalize
from torch.utils.data import TensorDataset, DataLoader


class Freq2Clean(nn.Module):
    def __init__(self, length, alphas=None):
        super().__init__()

        if not alphas:
            alphas = [0.5] * length
        self.alphas = nn.Parameter(torch.tensor(alphas))

    def forward(self, y_hat: torch.Tensor, y_bar: torch.Tensor) -> torch.Tensor:
        Y_hat = torch.fft.fft(y_hat, dim=1)
        Y_bar = torch.fft.fft(y_bar, dim=1)

        Y_hat_abs = torch.abs(Y_hat)
        Y_bar_abs = torch.abs(Y_bar)
        Y_hat_angle = torch.angle(Y_hat)
        # alpha_factor = torch.clamp(self.alphas, 0.0, 1.0) # Clamping for stability

        # alphas (n_freqs,) will be broadcast to match the shape of Y_bar_abs and Y_hat_abs (..., n_freqs).
        Y_abs = self.alphas * Y_bar_abs + (1 - self.alphas) * Y_hat_abs
        Y = torch.polar(Y_abs, Y_hat_angle)
        return torch.fft.ifft(Y, dim=1).real


# Args
denoiser_name: Literal["deepcad", "noise2noise", "noise2void"] = "deepcad"
denoiser_suffx = "theirs"
dataset = "oabf_vpm"
y_path = "../dataset/oabf/vpm/y.tiff"
max_frames = 2_000
CUPY_AVAILABLE = False
PATCH_T = 600
BATCH_SIZE = 4
EPOCHS = 200
LEARNING_RATE = 0.01


def psnr(y_pred: torch.Tensor, y_true: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = torch.mean((y_pred - y_true) ** 2)
    if mse < 1e-10:
        return torch.tensor(999.0, dtype=y_pred.dtype, device=y_pred.device)
    return 10.0 * torch.log10((data_range**2) / mse)


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    from src import *

    METRICS_PATH = Path(f"fft_{dataset}_metrics_{denoiser_name}_{denoiser_suffx}.csv")
    clog("red:Loading Dataset...")
    meta = DATASETS[dataset]
    x, y = (Recording(_, max_frames=max_frames) for _ in [meta.x, y_path])
    x.np = x.np[: y.frames, : y.np.shape[1], : y.np.shape[2]]
    gt = Recording(meta.gt, max_frames=max_frames)
    gt.np = gt.np[: y.frames, : y.np.shape[1], : y.np.shape[2]]
    RES_DIR = mkdir("results/{dataset}/")

    model = Freq2Clean(length=PATCH_T)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dataset = []

    for i in trange(x.frames // PATCH_T, desc="FFT fusion...", colour="cyan"):
        start = i * PATCH_T
        end = start + PATCH_T
        dataset.append((x.np[start:end], y.np[start:end], gt.np[start:end]))

    X_bar = torch.tensor(np.stack([d[0] for d in dataset]), dtype=torch.float32)
    X_hat = torch.tensor(np.stack([d[1] for d in dataset]), dtype=torch.float32)
    Y = torch.tensor(np.stack([d[2] for d in dataset]), dtype=torch.float32)

    X_bar = normalize(X_bar)
    X_hat = normalize(X_hat)
    Y = normalize(Y)

    dataset = TensorDataset(X_bar, X_hat, Y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()

        for X_hat, X_bar, Y in dataloader:
            optimizer.zero_grad()
            Y_pred = model(X_hat, X_bar)
            loss = -psnr(Y_pred, Y, data_range=2.0)  # Data range is [-1, 1], so MAX is 2.0
            loss.backward()
            optimizer.step()
