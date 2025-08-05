import torch.nn.functional as F
import torch


def __make_gaussian_kernel(channels, kernel_size=5, sigma=1.0, device="cuda"):
    # create 1D gaussian
    coords = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    # outer product to 2D kernel
    kernel2d = g[:, None] * g[None, :]
    kernel2d = kernel2d.expand(channels, 1, kernel_size, kernel_size)
    return kernel2d  # (channels, 1, kernel_size, kernel_size)


def lf_hf_tv_loss(I, I_c, I_n, lambda_lf=1.0, lambda_hf=0.5, lambda_tv=1e-4, kernel_size=5, sigma=1.0):
    B, C, H, W = I.shape
    device = I.device

    # low-pass filter
    gauss_k = __make_gaussian_kernel(C, kernel_size, sigma, device=device)
    lpf_I = F.conv2d(I, gauss_k, padding=kernel_size // 2, groups=C)
    lpf_Ic = F.conv2d(I_c, gauss_k, padding=kernel_size // 2, groups=C)
    # high-pass = residual
    hpf_I = I - lpf_I
    hpf_In = I_n - F.conv2d(I_n, gauss_k, padding=kernel_size // 2, groups=C)

    # LF & HF MSE
    loss_lf = F.mse_loss(lpf_I, lpf_Ic)
    loss_hf = F.mse_loss(hpf_I, hpf_In)

    # anisotropic total variation
    tv_h = torch.abs(I[:, :, :, :-1] - I[:, :, :, 1:]).mean()
    tv_v = torch.abs(I[:, :, :-1, :] - I[:, :, 1:, :]).mean()
    loss_tv = tv_h + tv_v

    return lambda_lf * loss_lf + lambda_hf * loss_hf + lambda_tv * loss_tv
