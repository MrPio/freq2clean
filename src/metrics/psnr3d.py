from skimage.metrics import peak_signal_noise_ratio


def psnr3d(vox1, vox2, data_range=1):
    """Peak Signal to Noise Ratio.
    data_range: distance between minimum and maximum possible values.
    """
    return peak_signal_noise_ratio(vox1, vox2, data_range=data_range)
