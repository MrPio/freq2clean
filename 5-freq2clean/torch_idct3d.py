import torch
import torch.nn as nn

# class IDCT3D(nn.Module):
#     def __init__(self, norm='ortho'):
#         super().__init__()
#         self.norm = norm

#     def idct_1d(self, X, dim=-1):
#         N = X.size(dim)

#         k = torch.arange(N, device=X.device)
#         factor = torch.exp(1j * torch.pi * k / (2 * N))
#         V = X * factor

#         # Reconstruct rFFT spectrum of length 2N
#         spec_shape = list(X.shape)
#         spec_shape[dim] = N + 1
#         spec = torch.zeros(*spec_shape, dtype=torch.cfloat, device=X.device)
#         # Put V in the first N entries along the correct axis
#         indices = [slice(None)] * X.dim()
#         indices[dim] = slice(0, N)
#         spec[tuple(indices)] = V

#         # Inverse FFT along the same dimension
#         v = torch.fft.irfft(spec, n=2*N, dim=dim)

#         # First N entries give the IDCT-III
#         indices[dim] = slice(0, N)
#         result = v[tuple(indices)]

#         if self.norm == 'ortho':
#             result = result / 2
#             result[..., 0] *= 1/torch.sqrt(torch.tensor(2.0))
#         return result

#     def forward(self, x):
#         # Apply along each axis explicitly
#         x = self.idct_1d(x, dim=-1)   # W
#         x = self.idct_1d(x, dim=-2)   # H
#         x = self.idct_1d(x, dim=-3)   # T
#         return x


class IDCT3D(nn.Module):
    def __init__(self, norm="ortho"):
        super().__init__()
        self.norm = norm

    def idct_1d(self, X):
        N = X.size(-1)
        k = torch.arange(N, device=X.device)
        factor = torch.exp(1j * torch.pi * k / (2 * N))

        # Pad to length N+1 for inverse
        X_pad = torch.zeros(X.shape[:-1] + (N ,), dtype=torch.cfloat, device=X.device)
        X_pad[..., :N] = X * 0.5
        # X_pad[..., N] = 0.0  # The extra term for rfft symmetry

        # Apply factor and perform inverse rfft
        x_ext = torch.fft.irfft(X_pad * factor, n=2*N, dim=-1)
        x = x_ext[..., :N]  # crop to original length
        return x

    def _idct_along_dim(self, X, dim):
        X = X.transpose(dim, -1)
        x = self.idct_1d(X)
        return x.transpose(dim, -1)

    def forward(self, X):
        # reverse order of 3D DCT
        x = self._idct_along_dim(X, -3)  # T
        x = self._idct_along_dim(x, -2)  # H
        x = self._idct_along_dim(x, -1)  # W
        return x
