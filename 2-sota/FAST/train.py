import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.dataset import ReadDatasets
from tqdm import tqdm
from models.Unet_Lite import Unet_Lite
from models.loss.loss import Projection_Loss
import datetime
from skimage import io
from datasets.data_process import generate_subimages, sampler
from utils.config import args2json


def goTraining(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Model and Loss Definition
    model = Unet_Lite(in_channels=args.miniBatch_size,
                      out_channels=args.miniBatch_size,
                      f_maps=[64, 64, 64],
                      num_groups=32,
                      final_sigmoid=True).cuda()

    projection_loss = Projection_Loss()
    model.cuda(args.local_rank)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    # Prepare data
    train_set = ReadDatasets(dataPath=args.train_folder, dataType=args.data_type, dataExtension=args.data_extension,
                             mode='train', denoising_strategy=args.denoising_strategy ,trainFrames=args.train_frames)
    val_set = ReadDatasets(dataPath=args.val_folder if args.withGT else args.train_folder, dataType=args.data_type,
                           dataExtension=args.data_extension, mode='val', denoising_strategy=args.denoising_strategy)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, drop_last=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,
                            pin_memory=True)

    # Define checkpoint directory
    checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint',
                                  os.path.basename(args.train_folder) + datetime.datetime.now().strftime("%Y%m%d%H%M"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Get list of input file names
    inputFileNames = [f for f in os.listdir(args.train_folder) if os.path.isfile(os.path.join(args.train_folder, f))]

    # Training Loop
    pbar = tqdm(range(args.epochs), total=args.epochs, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    seed = 1
    for epoch in pbar:
        for i, data in enumerate(train_loader):
            input, target = data
            input = input - torch.mean(input)
            target = target - torch.mean(target)
            input = input[0].cuda(args.local_rank, non_blocking=True)
            target = target[0].cuda(args.local_rank, non_blocking=True)

            miniBatch = args.miniBatch_size
            timeFrame = input.shape[1]
            assert miniBatch <= timeFrame, "miniBatch size must <= time frame length"

            for j in range(0, timeFrame - miniBatch, 1):
                input_batch = input[:, j:j + miniBatch, ...]
                target_batch = target[:, j:j + miniBatch, ...]

                mask1, mask2 = sampler(input_batch, operation_seed_counter=seed)
                seed += 1

                input_sub1 = generate_subimages(input_batch, mask1)
                target_sub2 = generate_subimages(target_batch, mask2)

                with torch.no_grad():
                    output_denoised = model(input_batch)

                output_denoised_sub1 = generate_subimages(output_denoised, mask1)
                input_sub1_denoised = model(input_sub1)

                loss1 = torch.nn.functional.mse_loss(input_sub1_denoised, target_sub2)
                loss2 = projection_loss(input_sub1_denoised, target_sub2)
                loss_ST = loss1 + loss2
                loss_SC = torch.nn.functional.mse_loss(input_sub1_denoised, output_denoised_sub1)

                loss_all = loss_ST + loss_SC
                optimizer.zero_grad()
                loss_all.backward()
                optimizer.step()

        # Validation and Model Saving
        if (epoch + 1) % int(args.save_freq) == 0:
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    input, t, _ = data
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
                                output[:, j:j + miniBatch, ...] = output_p
                            else:
                                output[:, j + miniBatch - 1, ...] = output_p[:, -1, ...]
                                output[:, j:j + miniBatch - 1, ...] = 0.5 * (
                                            output_p[:, :-1, ...] + output[:, j:j + miniBatch - 1, ...])

                    output = output[:, 0:t, ::]
                    output = output + mean
                    output_image = np.squeeze(output.numpy())
                    result_name = os.path.join(checkpoint_dir, f"Epoch_{epoch + 1}" + '_' + inputFileNames[i])
                    root, ext = os.path.splitext(result_name)
                    tif_name = root + ".tif"
                    io.imsave(tif_name, output_image[0:10, ::].astype(np.int32), check_contrast=False)

            filename = os.path.basename(args.train_folder) + "_Epoch_" + str(epoch + 1) + '.pth'
            final_name = os.path.join(checkpoint_dir, filename)
            torch.save({'state_dict': model.state_dict()}, final_name)
            args.checkpoint_path = final_name
            args.mode = 'test'
            args2json(args, checkpoint_dir + "//config.json")
