from skimage import io
import torch
import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size,
                           w // block_size)


def generate_subimages(img, masks):
    n, c, h, w = img.shape
    subimages = torch.zeros(n, c, h // 2, w // 2, dtype=img.dtype, layout=img.layout, device=img.device)
    for i in range(c):
        mask = masks[i]
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimages[:, i:i + 1, :, :] = img_per_channel[mask].reshape(n, h // 2, w // 2, 1).permute(0, 3, 1, 2)

    return subimages


def sampler(img: torch.Tensor, operation_seed_counter):
    """
        Based on the sampler from
        `"Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images"
            <https://github.com/TaoHuang2018/Neighbor2Neighbor>`.
    """
    n, c, h, w = img.shape
    device = img.device
    masks1 = []
    masks2 = []
    for i in range(c):
        mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4,), dtype=torch.bool, device=device)
        mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4,), dtype=torch.bool, device=device)
        idx_pair = torch.tensor(
            [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
            dtype=torch.int64,
            device=device)
        g_cpu_generator = torch.Generator()
        g_cpu_generator.manual_seed(operation_seed_counter + i)
        rd_idx = torch.randint(low=0, high=8, size=(n * h // 2 * w // 2,), generator=g_cpu_generator)

        rd_idx = rd_idx.to(device)

        rd_pair_idx = idx_pair[rd_idx]
        rd_pair_idx += torch.arange(start=0, end=n * h // 2 * w // 2 * 4, step=4, dtype=torch.int64,
                                    device=device).reshape(-1, 1)

        mask1[rd_pair_idx[:, 0]] = 1
        mask2[rd_pair_idx[:, 1]] = 1

        masks1.append(mask1)
        masks2.append(mask2)

    return masks1, masks2


def load3Dnpy2Tensor(dataPath: str, dataExtension: str) -> torch.Tensor:
    imageAll = []
    for imName in list(os.walk(dataPath, topdown=False))[-1][-1]:
        print('load image name -----> ', imName)
        imDir = dataPath + '//' + imName
        if dataExtension == 'tif':
            image = io.imread(imDir)
            os.remove(imDir)
        elif dataExtension == 'npy':
            image = np.load(imDir)
            os.remove(imDir)
        print(image.shape)
        imageTensor = torch.from_numpy(image / 1.0).float()
        imageAll.append(imageTensor)
    return imageAll


def load3DImages2Tensor(dataPath: str, dataExtension: str, trainFrames: int = -1) -> torch.Tensor:
    imageAll = []

    for imName in list(os.walk(dataPath, topdown=False))[-1][-1]:
        print('load image name -----> ', imName)
        imDir = dataPath + '//' + imName
        try:
            if dataExtension in ['tif', 'tiff']:
                image = io.imread(imDir)

            print('image shape:', image.shape)

            if trainFrames != -1 and image.shape[0] > trainFrames:
                image = image[:trainFrames, ...]

            imageTensor = torch.from_numpy(image / 1.0).float()
            imageAll.append(imageTensor)
        except:
            pass
    return imageAll


def split_into_patches(x, patchw, patchh, overlap):
    t, c, w, h = x.shape
    stepw = int(patchw * (1 - overlap))
    steph = int(patchh * (1 - overlap))
    patches = []
    ranges = []
    for i in range(0, w - patchw + 1, stepw):
        for j in range(0, h - patchh + 1, steph):
            patch = x[:, :, i:i + patchw, j:j + patchh]
            patch_range = {
                't_start': 0,
                't_end': t,
                'w_start': i,
                'w_end': i + patchw,
                'h_start': j,
                'h_end': j + patchh
            }
            patches.append(patch)
            ranges.append(patch_range)
    return patches, ranges


def process_patch(patch):
    processed_patch = patch * 2
    return processed_patch


def reconstruct_data(patches, shape):
    t, c, w, h = shape
    result = torch.zeros(shape)

    for patch, patch_range in patches:
        result[:, :, patch_range['w_start']:patch_range['w_end'], patch_range['h_start']:patch_range['h_end']] = patch

    return result


def get_gaussian(s, sigma=1.0 / 8) -> np.ndarray:
    temp = np.zeros(s)
    coords = [i // 2 for i in s]
    sigmas = [i * sigma for i in s]
    temp[tuple(coords)] = 1
    gaussian_map = gaussian_filter(temp, sigmas, 0, mode='constant', cval=0)
    gaussian_map /= np.max(gaussian_map)
    return gaussian_map
