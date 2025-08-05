from torch.utils.data import Dataset
from src.dataset.dataset import DATASETS, npy2tensor, random_transform
from src.recording import Recording


class NoisyCleanDataset(Dataset):
    def __init__(self, name, augument=True, max_frames=None):
        metadata = DATASETS[name]
        self.noisy_transforms = npy2tensor(metadata.max_val_x)
        self.clean_transforms = npy2tensor(metadata.max_val_y)
        self.augument = augument
        self.x = Recording(metadata.path_x, max_frames=max_frames)
        self.y = Recording(metadata.path_y, max_frames=max_frames)

    def __len__(self):
        return self.rec.frames

    def __getitem__(self, idx):
        noisy = self.noisy_transforms(self.x.np[idx])
        clean = self.clean_transforms(self.y.np[idx])
        if self.augument:
            noisy, clean = random_transform(noisy, clean)
        return noisy, clean
