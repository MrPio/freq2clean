import glob
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from .data_process import load3DImages2Tensor

class ReadDatasets(Dataset):
    """
    Dataset class for loading 3D images for training, validation, and testing.
    """

    def __init__(self, dataPath: str, mode: str, dataType: str, denoising_strategy: str, dataExtension: str , trainFrames=-1):
        super(ReadDatasets, self).__init__()
        assert mode in ['train', 'val', 'test'], "Mode must be 'train', 'val', or 'test'."
        assert dataType == '3D', "DataType must be '3D'."
        assert denoising_strategy == 'FAST', "Denoising strategy must be 'FAST'."
        assert dataExtension in ['tif', 'tiff'], "Data extension must be 'tif' or 'tiff'."
        self.denoising_strategy = denoising_strategy
        self.dataPath = dataPath
        self.mode = mode
        self.dataExtension = dataExtension
        self.inputFileNames = glob.glob(os.path.join(dataPath, '*.tif')) + glob.glob(os.path.join(dataPath, '*.tiff'))
        self.trainFrames = trainFrames
        if mode in ['train', 'test']:
            self.imageAll = load3DImages2Tensor(dataPath=dataPath, dataExtension=dataExtension, trainFrames=self.trainFrames)
            self.inputsNum = len(self.imageAll)
        elif mode == 'val':
            self.imageAll_gt = load3DImages2Tensor(dataPath=dataPath, dataExtension=dataExtension)
            path_raw = dataPath.replace("gt", "train")
            self.imageAll_raw = load3DImages2Tensor(dataPath=path_raw, dataExtension=dataExtension)
            self.inputsNum = len(self.imageAll_raw)

    def __getitem__(self, item):
        if self.mode == 'train':
            return self._get_train_item(item)
        elif self.mode == 'val':
            return self._get_val_item(item)
        else:
            return self._get_test_item(item)

    def _get_train_item(self, item):
        image = self.imageAll[item]
        t, w, h = image.shape
        input, label = [], []
        new_w = w - (w % 2)
        new_h = h - (h % 2)
        image = image[:, :new_w, :new_h]
        # Data augmentation
        image = torch.rot90(image, k=random.randint(0, 3), dims=[1, 2])
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[2])
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[1])
        for i in range(t - 1):
            if random.random() < 0.5:
                input.append(image[i, ...])
                label.append(image[i + 1, ...])
            else:
                input.append(image[i + 1, ...])
                label.append(image[i, ...])

        input_tensor = torch.stack(input, dim=0).unsqueeze(0)
        label_tensor = torch.stack(label, dim=0).unsqueeze(0)
        return input_tensor, label_tensor

    def _get_val_item(self, item):
        image_raw = self.imageAll_raw[item]
        t = image_raw.shape[0]
        image_gt = self.imageAll_gt[item]

        image_raw = torch.cat((image_raw, image_raw.flip(0)[:100, :, :]), dim=0)
        image_gt = torch.cat((image_gt, image_gt.flip(0)[:100, :, :]), dim=0)

        input_tensor = torch.stack([image_raw[i, ...] for i in range(image_raw.shape[0])], dim=0).unsqueeze(0)
        label_tensor = torch.stack([image_gt[i, ...] for i in range(image_gt.shape[0])], dim=0).unsqueeze(0)

        return input_tensor, t, label_tensor

    def _get_test_item(self, item):
        image = self.imageAll[item]
        t = image.shape[0]
        image = torch.cat((image, image.flip(0)[:100, :, :]), dim=0)

        input_tensor = torch.stack([image[i, ...] for i in range(image.shape[0])], dim=0).unsqueeze(0)
        return input_tensor, t

    def __len__(self):
        return self.inputsNum

if __name__ == '__main__':
    path = r"path to data"
    train_set = ReadDatasets(dataPath=path, mode='train', dataType='3D', denoising_strategy='FAST', dataExtension='tif')
    train_loader = DataLoader(dataset=train_set, batch_size=1)
