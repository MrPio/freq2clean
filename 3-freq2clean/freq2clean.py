from typing import Literal
import torch
import sys
import torch.nn as nn
from torch_dct3d import DCT3D
from torch_idct3d import IDCT3D
import torch.nn.init as init

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
    def __init__(self, shape: tuple[int, int, int], mode: Literal["dft1d", "dct3d"]):
        super().__init__()
        self.shape = shape
        self.mode = mode
        # init_alphas = [1] + [0.0] * (num_frames // 2)
        self.initialize_mask()
        if mode == "dct3d":
            self.dct3d = DCT3D()
            self.idct3d = IDCT3D()

    def initialize_mask(self, device=None):
        if self.mode == "dft1d":
            self.mask = nn.Parameter(torch.tensor([0.85] + [0] * (self.shape[0] // 2)))
        elif self.mode == "dct3d":
            s0, δs, t0, δt = 36, 72, -6, 16
            W_s = np.clip(
                (np.sqrt(np.arange(self.shape[1])[None, :] ** 2 + np.arange(self.shape[2])[:, None] ** 2) - (s0 - δs))
                / (2 * δs),
                0,
                1,
            )
            W_t = 1 - np.clip((np.arange(self.shape[0]) - (t0 - δt)) / (2 * δt), 0, 1)
            W = (W_s[:, :, None] * W_t).transpose(2, 0, 1)
            self.mask = nn.Parameter(torch.tensor(W))
        if device:
            self.to(device)
        # init.uniform_(self.mask, a=0.0, b=1.0)

    def forward(self, y_hat: torch.Tensor, y_bar: torch.Tensor) -> torch.Tensor:
        if self.mode == "dft1d":
            fwd = self.fft1d_forward
        elif self.mode == "dct3d":
            fwd = self.dct3d_forward
        return fwd(y_hat, y_bar)

    def fft1d_forward(self, y_hat, y_bar):
        Y_hat = torch.fft.rfft(y_hat, dim=1)
        Y_bar = torch.fft.rfft(y_bar, dim=1)

        Y_hat_abs = torch.abs(Y_hat)
        Y_bar_abs = torch.abs(Y_bar)
        Y_hat_angle = torch.angle(Y_hat)
        # alpha_factor = torch.clamp(self.mask, 0.0, 1.0) # Clamping for stability

        # freq0 = match(self.avg_frame.repeat(y_hat.shape[0], 1, 1), Y_hat_abs[:, 0] / self.frames) * self.frames
        # The mean of means is not the mean of all the values! The smaller the AVG_WIN/PATCH_T the better the approximation is
        Y_bar_abs[:, 0] = (
            match(
                src=Y_bar_abs[:, 0],
                trg=Y_hat_abs[:, 0] / self.shape[0],
            )
            * self.shape[0]
        )

        M = self.mask.view(1, -1, 1, 1)
        Y_abs = Y_bar_abs * M + Y_hat_abs * (1 - M)
        Y = torch.polar(Y_abs, Y_hat_angle)
        return torch.fft.irfft(Y, dim=1).real

    def dct3d_forward(self, y_hat, y_bar):
        Y_hat = self.dct3d(y_hat)
        Y_bar = self.dct3d(y_bar)

        mask = self.mask.unsqueeze(0)
        Y = Y_bar * mask + Y_hat * (1 - mask)
        return self.idct3d(Y)
