import datetime
import os
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from models.Unet_Lite import Unet_Lite
from datasets.dataset import ReadDatasets
from tqdm import tqdm
from skimage import io

def goTesting(args):
    logging.info('Testing Start!!!')

    # Create test dataset loader
    test_set = ReadDatasets(
        dataPath=args.test_path,
        dataType=args.data_type,
        dataExtension=args.data_extension,
        mode='test',
        denoising_strategy=args.denoising_strategy
    )

    num_gpu = (len(args.gpu_ids.split(",")) + 1) // 2
    inputFileNames = [f for f in os.listdir(args.test_path) if os.path.isfile(os.path.join(args.test_path, f))]

    filename = f'models_{os.path.basename(args.train_folder)}'
    testSave_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'result', filename)
    os.makedirs(testSave_dir, exist_ok=True)

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logging.info(f'Samples for testing = {len(test_set)}')
    output_channels = args.miniBatch_size

    # Load model
    model = Unet_Lite(
        in_channels=args.miniBatch_size,
        out_channels=output_channels,
        f_maps=[64, 64, 64],
        num_groups=32,
        final_sigmoid=True
    ).cuda()

    model.cuda(args.local_rank)

    if args.local_rank == 0:
        model_path = args.checkpoint_path
        if not os.path.exists(model_path):
            logging.error(f"Model checkpoint '{model_path}' not found.")
            return

        checkpoint = torch.load(model_path,weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    logging.info(('%15s' * 2) % ('GPU_mem', 'total_loss'))
    pbar = tqdm(total=len(test_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input, t = data
            mean = torch.mean(input)
            input = input - mean
            input = input[0].cuda(args.local_rank, non_blocking=True)
            output = torch.zeros(input.shape)

            miniBatch = args.miniBatch_size
            timeFrame = t
            assert miniBatch <= timeFrame, "miniBatch size must <= time frame length !!!"

            for j in range(0, timeFrame):
                if j + miniBatch <= timeFrame:
                    input_batch = input[:, j:j + miniBatch, ...]
                    output_p = model(input_batch).cpu().detach()
                    if j == 0:
                        output[:, j:j + output_channels, ...] = output_p
                    else:
                        output[:, j + output_channels - 1, ...] = output_p[:, -1, ...]
                        output[:, j:j + output_channels - 1, ...] = 0.5 * (output_p[:, :-1, ...] + output[:, j:j + output_channels - 1, ...])

            output = output[:, :t, ...] + mean
            output_image = np.squeeze(output.numpy()) * 1.0
            result_name = os.path.join(testSave_dir, f"{datetime.datetime.now().strftime('%Y%m%d%H%M')}_{inputFileNames[i]}")

            output_image = np.clip(output_image, 0, 65535)
            io.imsave(result_name, output_image.astype(np.uint16), check_contrast=False)

            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            pbar.set_description(f'Testing...\t\t{mem}')
            pbar.update(1)

    pbar.close()
    print("Saved: " + testSave_dir)
    print("Test End")

