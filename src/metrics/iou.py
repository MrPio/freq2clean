import numpy as np


def iou(mask1: np.array, mask2: np.array):
    """IoU for two boolean masks."""
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    if union == 0:
        return 0.0
    return intersection / union
