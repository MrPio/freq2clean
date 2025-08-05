import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import *


def random_transform(input, target):
    """
    (From DeepCAD-RT)
    The function for data augmentation. Randomly select one method among five
    transformation methods (including rotation and flip) or do not use data
    augmentation.

    Args:
        input, target : the input and target patch before data augmentation
    Return:
        input, target : the input and target patch after data augmentation
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


class Dataset2PM(Dataset):
    def __init__(self, dir: Path | str, noisy_transforms, clean_transforms, augument=True):
        self.dir = Path(dir)
        self.noisy_transforms = noisy_transforms
        self.clean_transforms = clean_transforms
        self.noisy_paths = sorted((self.dir / "noisy").glob("*.png"))
        self.clean_paths = sorted((self.dir / "clean").glob("*.png"))
        self.augument = augument

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        noisy = np.array(Image.open(self.noisy_paths[idx]))
        clean = np.array(Image.open(self.clean_paths[idx]))
        noisy = self.noisy_transforms(noisy)
        clean = self.clean_transforms(clean)
        if self.augument:
            noisy, clean = random_transform(noisy, clean)
        return noisy, clean


class AstroDataset(Dataset2PM):
    def __init__(self, augument=True):
        noisy_transforms = transforms.Compose(
            [
                # numpy uint16 H×W -> torch.Tensor 1×H×W [0,1]
                transforms.Lambda(lambda x: torch.from_numpy(x).float() / float(14_207)),
                transforms.Lambda(lambda x: torch.clip(x, min=0, max=1)),
                transforms.Lambda(lambda t: t.unsqueeze(0) if t.ndim == 2 else t),
                # normalize  [0,1] -> [–1,1]
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )
        clean_transforms = transforms.Compose(
            [
                # numpy uint16 H×W -> torch.Tensor 1×H×W [0,1]
                transforms.Lambda(lambda x: torch.from_numpy(x).float() / float(6_521)),
                transforms.Lambda(lambda x: torch.clip(x, min=0, max=1)),
                transforms.Lambda(lambda t: t.unsqueeze(0) if t.ndim == 2 else t),
                # normalize  [0,1] -> [–1,1]
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )
        super().__init__("dataset/oabf_astro", noisy_transforms, clean_transforms, augument)


class DatasetNextFrame(Dataset):
    def __init__(self, tiff, frames_per_patch=1):
        self.rec = Recording(tiff, max_frames=None).normalized
        self.frames_per_patch = frames_per_patch

    def __len__(self):
        return len(self.rec.frames - self.frames_per_patch * 2)

    def __getitem__(self, idx):
        even = self.rec[idx : idx + self.frames_per_patch * 2 : 2]
        odd = self.rec[idx + 1 : idx + self.frames_per_patch * 2 : 2]
        return torch.tensor(even), torch.tensor(odd)
