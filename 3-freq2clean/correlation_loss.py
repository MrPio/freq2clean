import torch
import torch.nn as nn


class CorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        B, T, H, W = x.shape
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
