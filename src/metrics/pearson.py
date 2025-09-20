import numpy as np

from src.video.recording import Recording


def pearson3d(vox1: np.ndarray | Recording, vox2: np.ndarray | Recording):
    if isinstance(vox1, Recording):
        vox1 = vox1.np
    if isinstance(vox2, Recording):
        vox2 = vox2.np

    D, W, H = vox1.shape
    v1 = vox1.reshape(D, -1)
    v2 = vox2.reshape(D, -1)
    # return np.nanmean([np.corrcoef(v1[:, i], v2[:, i])[0, 1] for i in range(W * H)])


    # Mean center
    v1_mean = v1 - np.mean(v1, axis=0, keepdims=True)
    v2_mean = v2 - np.mean(v2, axis=0, keepdims=True)

    # Compute correlation
    numerator = np.sum(v1_mean * v2_mean, axis=0)
    denominator = np.sqrt(np.sum(v1_mean**2, axis=0) * np.sum(v2_mean**2, axis=0))
    corr = numerator / denominator

    return np.nanmean(corr)
