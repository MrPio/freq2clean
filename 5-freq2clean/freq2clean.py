import torch
import sys
import torch.nn as nn

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
    def __init__(self, num_frames, avg_frame: np.array, alphas=None):
        super().__init__()
        self.frames = num_frames
        self.avg_frame = torch.tensor(avg_frame)
        if not alphas:
            alphas = [0.85] + [0.001] * (num_frames // 2)
        self.alphas = nn.Parameter(torch.tensor(alphas))

    def forward(self, y_hat: torch.Tensor, y_bar: torch.Tensor) -> torch.Tensor:
        Y_hat = torch.fft.rfft(y_hat, dim=1)
        Y_bar = torch.fft.rfft(y_bar, dim=1)

        Y_hat_abs = torch.abs(Y_hat)
        Y_bar_abs = torch.abs(Y_bar)
        Y_hat_angle = torch.angle(Y_hat)
        # alpha_factor = torch.clamp(self.alphas, 0.0, 1.0) # Clamping for stability

        freq0 = match(self.avg_frame.repeat(y_hat.shape[0], 1, 1), Y_hat_abs[:, 0] / self.frames) * self.frames
        Y_bar_abs[:, 0] = freq0

        Y_abs = Y_bar_abs * (self.alphas).view(1, -1, 1, 1) + Y_hat_abs * (1 - self.alphas).view(1, -1, 1, 1)
        Y = torch.polar(Y_abs, Y_hat_angle)
        return torch.fft.irfft(Y, dim=1).real

    # SSIM WIN ++, DCAD loss
