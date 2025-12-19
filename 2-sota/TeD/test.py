import numpy as np
import torch

from tqdm import tqdm
from utils.data_loader import DataFolder_test_stitch
from model.TeD import TeD
from utils.util import parse_arguments, np_imshow
from utils.data_loader import load_image, save_processed_image
import torch.nn as nn
import os

def validate(test_dataloader, model, device):
    """
    Validate a model with a test data

    Arguments:
        test_dataloader: (Pytorch DataLoader)
            Should be DatasetFRECTAL_test_stitch!
        model: (Pytorch nn.Module)

    Returns:
        denoised_stack: denoised image stack (Numpy array with dimension [T, X, Y])
    """
    with torch.no_grad():
        model.eval()
        # initialize denoised stack to NaN array.
        denoised_stack = np.zeros(test_dataloader.dataset.noisy_image.shape, dtype=np.float32)

        # stitching denoised stack
        for _, (noisy_image, rTG, _, single_coordinate) in enumerate(tqdm(test_dataloader, desc="validate")):
            noisy_image = noisy_image.to(device)  # [b, z, y, x]
            rTG = rTG.to(device)
            noisy_image_denoised = model(noisy_image, rTG)

            T = noisy_image.size(1)
            for bi in range(noisy_image.size(0)):
                stack_start_w = int(single_coordinate['stack_start_w'][bi])
                stack_end_w = int(single_coordinate['stack_end_w'][bi])
                patch_start_w = int(single_coordinate['patch_start_w'][bi])
                patch_end_w = int(single_coordinate['patch_end_w'][bi])

                stack_start_h = int(single_coordinate['stack_start_h'][bi])
                stack_end_h = int(single_coordinate['stack_end_h'][bi])
                patch_start_h = int(single_coordinate['patch_start_h'][bi])
                patch_end_h = int(single_coordinate['patch_end_h'][bi])

                stack_start_s = int(single_coordinate['init_s'][bi])
                denoised_stack[stack_start_s + (T // 2), stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = noisy_image_denoised[bi].squeeze()[patch_start_h:patch_end_h, patch_start_w:patch_end_w].cpu()

        # substitute nan values to 0 and denormalize
        denoised_stack = denoised_stack * test_dataloader.dataset.std_image.numpy() + test_dataloader.dataset.mean_image.numpy()
        return denoised_stack

if __name__ == '__main__':

    mode = "synthetic"  # Choose from: "in_vivo", "intravital", "naomi"

    if mode == "in_vivo":
        save_name = "TeD_invivo_vascular"
        data_folder = "invivo_vascular"
    elif mode == "intravital":
        save_name = "TeD_2p_intravital"
        data_folder = "2p_intravital"
    elif mode == "naomi":
        save_name = "TeD_naomi"
        data_folder = "naomi"
    elif mode == "synthetic":
        save_name = "synthetic"
        data_folder = "synthetic"
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose from 'in_vivo', 'intravital', 'naomi'")

    print(f"Mode: {mode}\nSave Name: {save_name}\nData Folder: {data_folder}")

    save_epoch = 149
    data_path = './data'
    batch_size = 100  # lower it if memory exceeds.

    model_file = os.path.join("./results/saved_models", save_name, "model_%d.pth" % (save_epoch))

    opt = parse_arguments()

    image_size = [opt.input_frames, 128, 128]
    image_interval = [1, image_size[1]-20, image_size[2]-20]

    model = TeD(img_size=(opt.image_size[1]//2, opt.image_size[2]//2),
                patch_size=opt.patch_size,
                in_channels=opt.input_frames,
                out_channels=opt.output_frames,
                window_size=opt.window_size, depths=opt.rstb_depths,
                embed_dim=opt.embed_dim,
                num_heads=opt.num_heads,
                attn_drop_rate=opt.attn_drop_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0], output_device=0)
    model.to(device)

    model.load_state_dict(
        torch.load(model_file, map_location="cuda:0"))

    print('Loaded trained model and optimizer weights of epoch {}'.format(save_epoch))

    file_paths = os.listdir(os.path.join(data_path, data_folder))
    file_paths.sort()

    for i in range(len(file_paths)):
        data_file = (os.path.join(os.path.join(data_path, data_folder), file_paths[i]))
        save_folder = os.path.join("./results/images", save_name,
                                   "%s_epoch_%d" % (data_folder, save_epoch))
        os.makedirs(save_folder, exist_ok=True)

        output_file = os.path.join(save_folder, file_paths[i])

        demo_tif, bit_depth = load_image(data_file)
        demo_tif = demo_tif[:, :, :]

        testset = DataFolder_test_stitch(demo_tif, patch_size=image_size, patch_interval=image_interval)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
        denoised_stack = validate(testloader, model, device)

        demo_tif = demo_tif[(opt.input_frames - 1) // 2:-(opt.input_frames - 1) // 2, :, :]
        denoised_stack = denoised_stack[(opt.input_frames - 1) // 2:-(opt.input_frames - 1) // 2, :, :]

        T, X, Y = denoised_stack.shape

        np_imshow(np.concatenate((demo_tif[int(T/2),:,:].squeeze().detach().cpu(),denoised_stack[int(T / 2), :, :].squeeze()), axis=1), cmin=0, cmax=255)
        save_processed_image(output_file, denoised_stack, bit_depth)
