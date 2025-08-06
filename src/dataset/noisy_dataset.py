from torch.utils.data import Dataset
from src.dataset.dataset import DATASETS, npy2tensor
from src.recording import Recording
from random import randrange

from src.utils import cprint


class NoisyDataset(Dataset):
    def __init__(
        self, name, patch_xy=64, frames_per_patch=2, augument=True, max_frames=None, overlap=0.5, verbose=False
    ):
        metadata = DATASETS[name]
        self.patch_xy = patch_xy
        self.transforms = npy2tensor(metadata.max_val_x)
        self.frames_per_patch = frames_per_patch
        self.augument = augument
        self.x = Recording(metadata.path_x, max_frames=max_frames)
        self.overlap = overlap

        D, H, W = self.x.np.shape
        h = w = self.patch_xy
        d = self.frames_per_patch * 2
        o = self.overlap
        self.Xs = int((W - w) // (w * (1 - o))) + 1
        self.Ys = int((H - h) // (h * (1 - o))) + 1
        self.Zs = int((D - d) // (d * (1 - o))) + 1

        self.verbose = verbose
        if verbose:
            cprint(
                "The dataset has",
                f"green:{self.Zs}z",
                "x",
                f"red:{self.Ys}y",
                "x",
                f"yellow:{self.Xs}x",
                "=",
                len(self),
                "samples",
            )

    def __len__(self):
        return self.Xs * self.Ys * self.Zs

    def idx2pos(self, idx):
        x = idx % self.Xs
        y = (idx % (self.Xs * self.Ys)) // self.Xs
        z = idx // (self.Xs * self.Ys)

        x1 = x * int(self.patch_xy * (1 - self.overlap))
        y1 = y * int(self.patch_xy * (1 - self.overlap))
        z1 = z * int(self.frames_per_patch * 2 * (1 - self.overlap))
        x2 = x1 + self.patch_xy
        y2 = y1 + self.patch_xy
        z2 = z1 + self.frames_per_patch * 2

        return (z1, z2), (y1, y2), (x1, x2)

    def __getitem__(self, idx):
        z, y, x = self.idx2pos(idx)
        if self.verbose:
            cprint(
                idx,
                "-->",
                f"green:[{z[0]}:{z[1]} z,",
                f"red:{y[0]}:{y[1]} y,",
                f"yellow:{x[0]}:{x[1]} x]",
            )
        even = self.x.np[z[0] : z[1] : 2, y[0] : y[1], x[0] : x[1]]
        odd = self.x.np[z[0] + 1 : z[1] : 2, y[0] : y[1], x[0] : x[1]]
        return self.transforms(even).unsqueeze(0), self.transforms(odd).unsqueeze(0)
