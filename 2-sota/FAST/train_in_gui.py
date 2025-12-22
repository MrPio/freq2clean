import os
import random
import time
import torch
import torch.optim
from tqdm import tqdm
from utils import *
from models.loss.loss import Projection_Loss
import datetime
from skimage import io
from datasets.data_process import generate_subimages, sampler
from utils.config import args2json


def goTraining(self):
    args = self.parameters.args
    if args.local_rank == 0:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log')
        log_file = log_dir + '.txt'
        log_args(log_file)
        logging.info('------This is all argsurations------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('-------This is a halving line------')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = self.parameters.model
    projection_loss = Projection_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay,
                                 amsgrad = args.amsgrad)
    inputFileNames = list(os.walk(args.train_folder, topdown = False))[-1][-1]
    checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint',
                                  os.path.basename(args.train_folder) + datetime.datetime.now().strftime(
                                      "%Y%m%d%H%M"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    logging.info('Training Start!!!')
    train_loader = self.parameters.train_loader
    val_loader = self.parameters.val_loader
    start_time = time.time()
    torch.set_grad_enabled(True)
    logging.info(('\n' + '%15s' * 3) % ('Epoch', 'GPU_mem', 'total_loss'))
    pbar = range(args.epochs)
    pbar = tqdm(pbar, total = args.epochs, bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    seed = 0
    flag = 0
    for epoch in pbar:
        for i, data in enumerate(train_loader):
            adjust_learning_rate(optimizer, epoch, args.epochs, args.lr)
            #  ###############################################Training-Part#################################################
            input, target = data
            input = input[0].cuda(args.local_rank, non_blocking = True)
            if flag == 0:
                input_image = np.squeeze(input.cpu().detach().numpy()) * 1.0
                input_array = input_image[20, ::].astype(np.uint16)

                self.imageReady.emit(np.squeeze(input_array), 'raw')
                self.imageReady.emit(np.squeeze(input_array), 'denoised')
                flag = 1
            input = input - torch.mean(input)
            target = target - torch.mean(target)
            target = target[0].cuda(args.local_rank, non_blocking = True)
            miniBatch = args.miniBatch_size
            timeFrame = input.shape[1]
            assert miniBatch <= timeFrame, "miniBatch size must <= time frame length !!!"
            for j in range(0, timeFrame - miniBatch, 1):  # sub-batch
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
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'


                #  ####################################################End######################################################

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = args.clip_gradients)  # clip gradients

                loss_all.backward()
                optimizer.step()
        pbar.set_postfix(
            epoch = f'{epoch + 1}/{args.epochs}',
            GPU_mem = mem,
            loss = f'{loss_all:.2f}'
        )

        if (epoch + 1) % 1 == 0:
            model.eval()  # Set the model to evaluation mode

            with torch.no_grad():  # Disable gradient calculation during validation
                for i, data in enumerate(val_loader):
                    input, t, _ = data
                    mean = torch.mean(input)
                    input = input - mean
                    input = input[0].cuda(args.local_rank, non_blocking = True)
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

                    input = input[:, 0:t, ::]
                    input = input + mean
                    output = output[:, 0:t, ::]
                    output = output + mean

                    output_image = np.squeeze(output.cpu().detach().numpy()) * 1.0
                    input_image = np.squeeze(input.cpu().detach().numpy()) * 1.0
                    ####
                    image_array = output_image[20, ::].astype(np.uint16)
                    self.imageReady.emit(np.squeeze(image_array), 'denoised')
                    image_array = input_image[20, ::].astype(np.uint16)
                    self.imageReady.emit(np.squeeze(image_array), 'raw')

                    result_name = os.path.join(checkpoint_dir,
                                               f"Epoch_{epoch + 1}" + '_' + inputFileNames[i])

                    root, ext = os.path.splitext(result_name)

                    tif_name = root + ".tif"
                    io.imsave(tif_name, output_image[20, ::].astype(np.int16), check_contrast = False)

            filename = os.path.basename(args.train_folder) + "_Epoch_" + str(epoch + 1) + '.pth'
            final_name = os.path.join(checkpoint_dir, filename)
            torch.save({
                'state_dict': model.state_dict(),
            },
                final_name)
            # #  ########################################################################################################
            print('\nModel is saved in ' + filename)
            args.checkpoint_path = final_name
            args.mode = 'test'
            # args.data_extension = 'npy'
            args2json(args, checkpoint_dir + "//config.json")
            # Clear unnecessary variables and tensors to release memory
            del input, output, input_batch, output_image
            torch.cuda.empty_cache()  # Clear GPU memory
            model.train()  # Set the model back to training mode

    end_time = time.time()
    total_time = (end_time - start_time) / 3600

    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('------The training process finished!------')
