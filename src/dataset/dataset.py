import torch
from torchvision import transforms
import random
from pathlib import Path

_ROOT_DIR = Path(__file__).parents[2]


class DatasetMetadata:
    def __init__(self, dir, data_range=2**16, max_value_x=None, max_value_y=None):
        self.dir = _ROOT_DIR / dir
        self.path_x = self.dir / "x.tiff"
        self.path_y = self.dir / "y.tiff"
        self.max_val_x = max_value_x
        self.max_val_y = max_value_y
        self.data_range = data_range


DATASETS = {
    "oabf_astro": DatasetMetadata(
        dir="dataset/oabf/astro",
        max_value_x=14_207,
        max_value_y=6_521,
    ),
    "oabf_vpm": DatasetMetadata(dir="dataset/oabf/vpm"),
    "oabf_resonant_neuro": DatasetMetadata(dir="dataset/oabf/resonant_neuro"),
    "synthetic": DatasetMetadata(
        dir="dataset/zenodo/synthetic", data_range=1_520
    ),  # 1_520 is the 99.9% Quantile of GT
    "zebrafish": DatasetMetadata(dir="dataset/zenodo/zebrafish", data_range=32_767),
}


def random_transform(input, target):
    """
    (From DeepCAD-RT)
    The function for data augmentation. Randomly select one method among five
    transformation methods (including rotation and flip) or do not use data
    augmentation.

    Args:
        input, target (C,W,H) : the input and target patch before data augmentation
    Return:
        input, target (C,W,H) : the input and target patch after data augmentation
    """
    p_trans = random.randrange(0, 8)
    if p_trans == 0:
        pass
    elif p_trans == 1:  # left rotate 90
        input = torch.rot90(input, k=1, dims=(1, 2))
        target = torch.rot90(target, k=1, dims=(1, 2))
    elif p_trans == 2:  # left rotate 180
        input = torch.rot90(input, k=2, dims=(1, 2))
        target = torch.rot90(target, k=2, dims=(1, 2))
    elif p_trans == 3:  # left rotate 270
        input = torch.rot90(input, k=3, dims=(1, 2))
        target = torch.rot90(target, k=3, dims=(1, 2))
    elif p_trans == 4:  # horizontal flip
        input = torch.flip(input, dims=(2,))
        target = torch.flip(target, dims=(2,))
    elif p_trans == 5:  # horizontal flip & left rotate 90
        input = torch.flip(input, dims=(2,))
        input = torch.rot90(input, k=1, dims=(1, 2))
        target = torch.flip(target, dims=(2,))
        target = torch.rot90(target, k=1, dims=(1, 2))
    elif p_trans == 6:  # horizontal flip & left rotate 180
        input = torch.flip(input, dims=(2,))
        input = torch.rot90(input, k=2, dims=(1, 2))
        target = torch.flip(target, dims=(2,))
        target = torch.rot90(target, k=2, dims=(1, 2))
    elif p_trans == 7:  # horizontal flip & left rotate 270
        input = torch.flip(input, dims=(2,))
        input = torch.rot90(input, k=3, dims=(1, 2))
        target = torch.flip(target, dims=(2,))
        target = torch.rot90(target, k=3, dims=(1, 2))
    return input, target


def npy2tensor(max_value):
    return transforms.Compose(
        [
            # numpy uint16 H×W -> torch.Tensor 1×H×W [0,1]
            transforms.Lambda(lambda x: torch.from_numpy(x).float() / float(max_value)),
            transforms.Lambda(lambda x: torch.clip(x, min=0, max=1)),
            transforms.Lambda(lambda t: t.unsqueeze(0) if t.ndim == 2 else t),
            # normalize  [0,1] -> [–1,1]
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    )
