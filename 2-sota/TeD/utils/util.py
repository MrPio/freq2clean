import math
import argparse
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
from kornia.filters.kernels import get_binary_kernel2d
import torch.nn as nn
import torch.nn.functional as F
import logging

"""Referenced by SUPPORT version of utils
https://github.com/NICALab/SUPPORT
"""
def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument("--random_seed", type=int, default=0, help="random seed for rng")
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from (need epoch-1 model)")
    parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
    parser.add_argument("--save_name", type=str, default="TeD_250121_2pIntraVital", help="name of the experiment")

    parser.add_argument("--results_dir", type=str, default="./results", help="root directory to save results")
    parser.add_argument("--input_frames", type=int, default=41, help="# of input frames")
    parser.add_argument("--output_frames", type=int, default=1, help="# of input frames")

    # dataset
    parser.add_argument("--is_folder", action="store_true", help="noisy_data is folder")
    parser.add_argument("--noisy_data", type=str, nargs="+", help="List of path to the noisy data")
    parser.add_argument("--image_size", type=int, default=[41, 128, 128], nargs="+", help="size of the patches")
    parser.add_argument("--batch_num", type=int, default=50, help="size of the batches")
    parser.add_argument("--root", type=str, default="./data/synthetic", help="dataset folder")

    # model
    parser.add_argument("--patch_size", type=int, default=4, nargs="+", help="patch size")
    parser.add_argument("--window_size", type=int, default=8, nargs="+", help="window size")
    parser.add_argument("--rstb_depths", type=int, default=[2,2,2,2,2], nargs="+", help="depth of each Swin Transformer layer")

    parser.add_argument("--embed_dim", type=int, default=60, nargs="+", help="patch embedding dimension")
    parser.add_argument("--num_heads", type=int, default=6, nargs="+", help="number of attention heads in different layers")
    parser.add_argument("--attn_drop_rate", type=int, default=0.1, nargs="+", help="attention dropout rate")

    # training
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: min betas")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: max betas")
    parser.add_argument("--loss_coef", type=float, default=[0.5, 0.5, 1e-3], nargs="+", help="L1/L2/Reg loss coefficients")

    # util
    parser.add_argument("--use_CPU", action="store_true", help="use CPU")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--logging_interval_batch", type=int, default=50, help="interval between logging info (in batches)")
    parser.add_argument("--logging_interval", type=int, default=1, help="interval between logging info (in epochs)")
    parser.add_argument("--sample_interval", type=int, default=10, help="interval between saving denoised samples")
    parser.add_argument("--sample_max_t", type=int, default=600, help="maximum time step of saving sample")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving trained models (in epochs)")
    opt = parser.parse_args()

    # argument checking
    if (opt.input_frames) != opt.image_size[0]:
        raise Exception("input frames must be equal to z-frames of patch_size")
    if len(opt.loss_coef) != 3:
        raise Exception("loss_coef must be length-3 array")

    return opt


def get_coordinate(img_size, patch_size, patch_interval):
    """DeepCAD version of stitching
    https://github.com/cabooster/DeepCAD
    """
    whole_s, whole_h, whole_w = img_size
    img_s, img_h, img_w = patch_size
    gap_s, gap_h, gap_w = patch_interval

    cut_w = (img_w - gap_w)/2
    cut_h = (img_h - gap_h)/2
    cut_s = (img_s - gap_s)/2

    num_w = math.ceil((whole_w-img_w+gap_w)/gap_w)
    num_h = math.ceil((whole_h-img_h+gap_h)/gap_h)
    num_s = math.ceil((whole_s-img_s+gap_s)/gap_s)

    coordinate_list = []
    for x in range(0,num_h):
        for y in range(0,num_w):
            for z in range(0,num_s):
                single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                if x != (num_h-1):
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                elif x == (num_h-1):
                    init_h = whole_h - img_h
                    end_h = whole_h

                if y != (num_w-1):
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                elif y == (num_w-1):
                    init_w = whole_w - img_w
                    end_w = whole_w

                if z != (num_s-1):
                    init_s = gap_s*z
                    end_s = gap_s*z + img_s
                elif z == (num_s-1):
                    init_s = whole_s - img_s
                    end_s = whole_s
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if y == 0:
                    if num_w > 1:
                        single_coordinate['stack_start_w'] = y*gap_w
                        single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                        single_coordinate['patch_start_w'] = 0
                        single_coordinate['patch_end_w'] = img_w-cut_w
                    else:
                        single_coordinate['stack_start_w'] = 0
                        single_coordinate['stack_end_w'] = img_w
                        single_coordinate['patch_start_w'] = 0
                        single_coordinate['patch_end_w'] = img_w
                elif y == num_w-1:
                    single_coordinate['stack_start_w'] = whole_w-img_w+cut_w
                    single_coordinate['stack_end_w'] = whole_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = img_w
                else:
                    single_coordinate['stack_start_w'] = y*gap_w+cut_w
                    single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = img_w-cut_w

                if x == 0:
                    if num_h > 1:
                        single_coordinate['stack_start_h'] = x*gap_h
                        single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                        single_coordinate['patch_start_h'] = 0
                        single_coordinate['patch_end_h'] = img_h-cut_h
                    else:
                        single_coordinate['stack_start_h'] = 0
                        single_coordinate['stack_end_h'] = x*gap_h+img_h
                        single_coordinate['patch_start_h'] = 0
                        single_coordinate['patch_end_h'] = img_h
                elif x == num_h-1:
                    single_coordinate['stack_start_h'] = whole_h-img_h+cut_h
                    single_coordinate['stack_end_h'] = whole_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = img_h
                else:
                    single_coordinate['stack_start_h'] = x*gap_h+cut_h
                    single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = img_h-cut_h

                if z == 0:
                    if num_s > 1:
                        single_coordinate['stack_start_s'] = z*gap_s
                        single_coordinate['stack_end_s'] = z*gap_s+img_s-cut_s
                        single_coordinate['patch_start_s'] = 0
                        single_coordinate['patch_end_s'] = img_s-cut_s
                    else:
                        single_coordinate['stack_start_s'] = z*gap_s
                        single_coordinate['stack_end_s'] = z*gap_s+img_s
                        single_coordinate['patch_start_s'] = 0
                        single_coordinate['patch_end_s'] = img_s
                elif z == num_s-1:
                    single_coordinate['stack_start_s'] = whole_s-img_s+cut_s
                    single_coordinate['stack_end_s'] = whole_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = img_s
                else:
                    single_coordinate['stack_start_s'] = z*gap_s+cut_s
                    single_coordinate['stack_end_s'] = z*gap_s+img_s-cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = img_s-cut_s

                coordinate_list.append(single_coordinate)

    return coordinate_list

def setup_logger(name, log_file, level=logging.INFO):
    logging.getLogger().handlers.clear()
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if not logger.hasHandlers():
        # File handler (logs to file)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream handler (logs to console)
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    logger.propagate = False
    return logger

def imshow(inp, title = None, cmin=0, cmax=255):
    # Set caxis
    cmin = cmin
    cmax = cmax
    inp = inp.detach().numpy().transpose((0,1))

    tpef_colors = [(0,0,0), (0, 0.5, 0), (0, 1.0, 0)]
    tpef_cmap = colors.LinearSegmentedColormap.from_list('test', tpef_colors, N=512)
    plt.imshow(inp, cmap=tpef_cmap)
    plt.clim(cmin, cmax)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def np_imshow(inp, title = None, cmin=0, cmax=255):
    # Set caxis
    cmin = cmin
    cmax = cmax
    inp = inp.transpose((0,1))

    tpef_colors = [(0,0,0), (0, 0.5, 0), (0, 1.0, 0)]
    tpef_cmap = colors.LinearSegmentedColormap.from_list('test', tpef_colors, N=512)
    plt.imshow(inp, cmap=tpef_cmap)
    plt.clim(cmin, cmax)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def rTG_imshow(inp, title = None):
    # Set caxis
    cmin = 0
    cmax = 1.0
    inp = inp.detach().numpy().transpose((0,1))

    tpef_colors = [(0,0,0), (0, 0.5, 0), (0, 1.0, 0)]
    tpef_cmap = colors.LinearSegmentedColormap.from_list('test', tpef_colors, N=512)
    plt.imshow(inp, 'jet')
    plt.clim(cmin, cmax)
    plt.colorbar()

    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def median_filter_2d(input, kernel_size):
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))
    input = input.unsqueeze(dim=0)
    padding_size = [(k - 1) // 2 for k in kernel_size]

    padding = nn.ReplicationPad2d(padding_size[0])

    input = padding(input)

    # prepare kernel
    kernel = get_binary_kernel2d(kernel_size).to(input)

    b, c, h, w = input.shape

    # map the local window to single vector
    features = F.conv2d(input.reshape(b * c, 1, h, w), kernel, padding=(0,0), stride=1)
    features = features.view(b, c, -1, h-padding_size[0]*2, w-padding_size[0]*2)  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    median = torch.median(features, dim=2)[0]

    return median.squeeze()

def mean_filter_2d(input, kernel_size):
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))
    input = input.unsqueeze(dim=0)
    padding_size = [(k - 1) // 2 for k in kernel_size]

    padding = nn.ReplicationPad2d(padding_size[0])

    input = padding(input)

    # prepare kernel
    kernel = get_binary_kernel2d(kernel_size).to(input)

    b, c, h, w = input.shape

    # map the local window to single vector
    features = F.conv2d(input.reshape(b * c, 1, h, w), kernel, padding=(0,0), stride=1)
    features = features.view(b, c, -1, h-padding_size[0]*2, w-padding_size[0]*2)  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    mean = torch.mean(features, dim=2)[0]

    return mean.squeeze()
