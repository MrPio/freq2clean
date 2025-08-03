import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random


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
    p_trans = random.randrange(8)
    if p_trans == 0:  # no transformation
        input = input
        target = target
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
    def __init__(self, dir, augument=True):
        self.root = dir
        self.transforms = transforms.Compose(
            [
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.ToTensor(),  # [0,1]
                transforms.Normalize(0.5, 0.5),  # [–1,1]
            ]
        )
        # assume structure root/noisy/*.png and root/cond/*.png
        self.noisy_paths = sorted(os.listdir(os.path.join(dir, "noisy")))
        self.cond_paths = sorted(os.listdir(os.path.join(dir, "cond")))
        self.augument = augument

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        noisy = Image.open(os.path.join(self.root, "noisy", self.noisy_paths[idx])).convert("L")
        cond = Image.open(os.path.join(self.root, "cond", self.cond_paths[idx])).convert("L")
        noisy = self.transforms(noisy)
        cond = self.transforms(cond)
        if self.augument:
            noisy, cond = random_transform(noisy, cond)
        return noisy, cond
