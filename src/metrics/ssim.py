"""
SSIM3D implementation by CNNT: https://github.com/AzR919/CNNT_Microscopy/blob/9fafe48ffa2da8b665cc28cb9d1b5dd3cd151f8f/models/pytorch_ssim.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

from src.video.recording import Recording
from skimage.metrics import structural_similarity


def __ssim_3d(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1).mean(1)


def __create_window_3D(window_size, channel):
    _1D_window = __gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = (
        _1D_window.mm(_2D_window.reshape(1, -1))
        .reshape(window_size, window_size, window_size)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def __gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def ssim3d(vox1, vox2, window_size=11, size_average=True, device="cuda"):
    """Data must be normalized [0,1]"""
    if isinstance(vox1, Recording):
        vox1 = vox1.np
    if isinstance(vox2, Recording):
        vox2 = vox2.np

    if isinstance(vox1, np.ndarray) and isinstance(vox2, np.ndarray):
        vox1 = torch.from_numpy(vox1).unsqueeze(0).unsqueeze(0).float().to(device)  # -> (N=1,C=1,D,H,W)
        vox2 = torch.from_numpy(vox2).unsqueeze(0).unsqueeze(0).float().to(device)  # -> (N=1,C=1,D,H,W)

    with torch.no_grad():
        (_, channel, _, _, _) = vox1.size()
        window = __create_window_3D(window_size, channel)

        if vox1.is_cuda:
            window = window.cuda(vox1.get_device())
        window = window.type_as(vox1)

        return __ssim_3d(vox1, vox2, window, window_size, channel, size_average).item()


def ssim(img1: np.ndarray | Recording, img2: np.ndarray | Recording, data_range=1):
    """Peak Signal to Noise Ratio.
    data_range: distance between minimum and maximum possible values.
    """
    if isinstance(img1, Recording):
        img1 = img1.np
    if isinstance(img2, Recording):
        img2 = img2.np
    return structural_similarity(img1, img2, data_range=data_range)
