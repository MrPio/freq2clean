import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from src.video.recording import Recording


def psnr3d(vox1: np.ndarray | Recording, vox2: np.ndarray | Recording, data_range=1):
    return psnr(vox1, vox2, data_range)


def psnr(img1: np.ndarray | Recording, img2: np.ndarray | Recording, data_range=1):
    """Peak Signal to Noise Ratio.
    data_range: distance between minimum and maximum possible values.
    """
    if isinstance(img1, Recording):
        img1 = img1.np
    if isinstance(img2, Recording):
        img2 = img2.np
    return peak_signal_noise_ratio(img1, img2, data_range=data_range)
