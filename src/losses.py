import torch.nn.functional as F
import torch


def __make_gaussian_kernel(channels, kernel_size=5, sigma=1.0, device="cuda"):
    # create 1D gaussian
    coords = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    # outer product to 2D kernel
    kernel3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
    kernel3d = kernel3d.expand(channels, 1, kernel_size, kernel_size, kernel_size)
    return kernel3d  # (channels, 1, kernel_size, kernel_size, kernel_size)


def lf_hf_tv(pred, gt, lambda_lf=1.0, lambda_hf=0.5, lambda_tv=1e-4, kernel_size=5, sigma=1.0):
    B, C, D, H, W = pred.shape
    device = pred.device

    # low-pass filter
    gauss_k = __make_gaussian_kernel(C, kernel_size, sigma, device=device)
    lpf_pred = F.conv3d(pred, gauss_k, padding=kernel_size // 2, groups=C)
    lpf_gt = F.conv3d(gt, gauss_k, padding=kernel_size // 2, groups=C)

    # high-pass = residual
    hpf_pred = pred - lpf_pred
    hpf_gt = gt - F.conv3d(gt, gauss_k, padding=kernel_size // 2, groups=C)

    # LF & HF MSE
    loss_lf = F.mse_loss(lpf_pred, lpf_gt)
    loss_hf = F.mse_loss(hpf_pred, hpf_gt)

    # anisotropic total variation
    tv_h = torch.abs(pred[:, :, :, :, :-1] - pred[:, :, :, :, 1:]).mean()
    tv_v = torch.abs(pred[:, :, :, :-1, :] - pred[:, :, :, 1:, :]).mean()
    tv_d = torch.abs(pred[:, :, :-1, :, :] - pred[:, :, 1:, :, :]).mean()
    loss_tv = tv_h + tv_v + tv_d

    return lambda_lf * loss_lf + lambda_hf * loss_hf + lambda_tv * loss_tv
