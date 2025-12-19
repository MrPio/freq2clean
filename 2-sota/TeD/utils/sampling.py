import torch
from einops import rearrange
import numpy as np
"""Referenced by SRDTrans version of sampling
https://github.com/cabooster/SRDTrans
"""
operation_seed_counter = 0

def generate_mask_pair(img):
    n, c, h, w = img.shape

    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)

    # prepare random mask pairs
    idx_pair = torch.tensor([
        [0, 1], [1, 0],
        [1, 3], [3, 1],
        [0, 2], [2, 0],
        [2, 3], [3, 2]],
        dtype=torch.int64, device=img.device)

    n_i, _ = idx_pair.shape


    rd_idx = torch.zeros(size=(n * h // 2 * w // 2,),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=n_i,
                  size=(n * h // 2 * w // 2,),
                  out=rd_idx, device=img.device)

    # [n * h // 2 * w // 2, ]
    rd_pair_idx = idx_pair[rd_idx]
    # [n * t * h // 2 * w // 2, 2]

    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1

    return mask1, mask2


def generate_subimages(img, mask):
    n, c, h, w = img.shape

    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)

    return subimage


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    # cuda
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


if __name__ == "__main__":
    batch_dim = 3
    channel_dim = 1
    spatial_dim = 8

    pixel_num = batch_dim * channel_dim * (spatial_dim ** 2)
    img = torch.linspace(0, pixel_num-1, steps=pixel_num, dtype=torch.float)
    img = torch.reshape(img, (batch_dim, channel_dim, spatial_dim, spatial_dim))

    T = img[:,0,:].shape
    print(T)

    mask1, mask2 = generate_mask_pair(img)
    sub1 = generate_subimages(img, mask1)
    sub2 = generate_subimages(img, mask2)

    print(img)
    print(sub1)
    print(sub2)

    sub3 = generate_subimages(img[:,0,:].unsqueeze(dim=1), mask2)
    print(sub3)





