import torch
import torch.nn as nn


class DCT3D(nn.Module):
    def __init__(self, norm="ortho"):
        super().__init__()
        self.norm = norm

    def dct_1d(self, x):
        N = x.size(-1)
        x_ext = torch.cat([x, x.flip(dims=[-1])], dim=-1)   # length 2N
        X = torch.fft.rfft(x_ext, dim=-1)                   # length N+1 complex
        k = torch.arange(N, device=x.device)
        factor = torch.exp(-1j * torch.pi * k / (2 * N))
        return 2 * torch.real(X[..., :N] * factor)

    def _dct_along_dim(self, x, dim):
        # bring target dim to last, apply 1D dct, restore dims
        x = x.transpose(dim, -1)
        y = self.dct_1d(x)
        return y.transpose(dim, -1)

    def forward(self, x):
        # input shape assumed (B, T, H, W)
        X = self._dct_along_dim(x, -1)   # W
        X = self._dct_along_dim(X, -2)   # H
        X = self._dct_along_dim(X, -3)   # T
        return X