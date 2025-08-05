from torch.utils.data import Dataset
from src.dataset.dataset import DATASETS, npy2tensor
from src.recording import Recording


class NoisyDataset(Dataset):
    def __init__(self, name, frames_per_patch=2, augument=True, max_frames=None):
        metadata = DATASETS[name]
        self.transforms = npy2tensor(metadata.max_val_x)
        self.frames_per_patch = frames_per_patch
        self.augument = augument
        self.x = Recording(metadata.path_x, max_frames=max_frames)

    def __len__(self):
        return len(self.x.frames - self.frames_per_patch * 2)

    def __getitem__(self, idx):
        even = self.x.np[idx : idx + self.frames_per_patch * 2 : 2]
        odd = self.x.np[idx + 1 : idx + self.frames_per_patch * 2 : 2]
        return self.transforms(even), self.transforms(odd)
