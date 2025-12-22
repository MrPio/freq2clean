import argparse
import datetime
import os
import logging
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.Unet_Lite import Unet_Lite
from datasets.dataset import ReadDatasets
from utils import *
from tqdm import tqdm
from skimage import io
from torchvision.models import resnet50



def goTesting(args):
    device = 'cuda:0'
    model1 = resnet50().to(device)
    dummy_input = torch.rand(1, 3, 256, 256).to(device)

    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model1(dummy_input)

    logging.info('Testing Start!!!')

    print('test set')
    test_set = ReadDatasets(dataPath = args.test_path,
                            dataType = args.data_type,
                            dataExtension = args.data_extension,
                            mode = 'test',
                            denoising_strategy = args.denoising_strategy)

    num_gpu = (len(args.gpu_ids.split(",")) + 1) // 2
    inputFileNames = list(os.walk(args.test_path, topdown = False))[-1][-1]
    filename = 'models_' + os.path.basename(args.train_folder)
    testSave_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'result',
                                filename)
    # testSave_dir = os.path.join(args.results_dir, filename, )
    print('testSave_dir = {}'.format(testSave_dir))
    if not os.path.exists(testSave_dir):
        os.makedirs(testSave_dir)
    test_loader = DataLoader(dataset = test_set, batch_size = args.batch_size,
                             drop_last = True, num_workers = args.num_workers, pin_memory = True)

    logging.info('Samples for testing = {}'.format(len(test_set)))
    output_channels = args.miniBatch_size
    #  #################################################Model&Loss-Define#######################################################
    model = Unet_Lite(in_channels = args.miniBatch_size,
                      out_channels = output_channels,
                      f_maps = [64, 64, 64],
                      num_groups = 32,
                      final_sigmoid = True).cuda()

    # model = TemporalShift(model, n_segment = 10, n_div = 3, inplace = False)
    # model = basicVSR(spynet_pretrained = args.spynet_pretrained).cuda()
    #  #######################################################End###############################################################
    model.cuda(args.local_rank)
    # model = nn.parallel.DataParallel(model)

    if args.local_rank == 0:
        model_path = args.checkpoint_path
        if not os.path.exists(model_path):
            logging.error(f"Model checkpoint '{args.checkpoint}' not found.")
            return

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        # model.load_state_dict(checkpoint)
    model.cuda(args.local_rank)

    model.eval()
    logging.info(('\n' + '%15s' * 2) % ('GPU_mem', 'total_loss'))
    pbar = tqdm(total = len(test_loader), bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    # print(model)
    # input = Variable(torch.randn(1, 16, 512, 512)).cuda()
    # flops, params = profile(model, (input,))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
    start = time.perf_counter()
    with torch.no_grad():  # Disable gradient calculation during validation
        for i, data in enumerate(test_loader):
            input, t = data
            mean = torch.mean(input)
            input = input - mean
            del data
            input = input[0].cuda(args.local_rank, non_blocking = True)

            output = torch.zeros(input.shape)
            output_patch = []
            miniBatch = args.miniBatch_size
            timeFrame = t

            assert miniBatch <= timeFrame, "miniBatch size must <= time frame length !!!"
            timings = np.zeros((timeFrame, 1))
            # for j in range(0, timeFrame, timeFrame // 2):  # sub-batch
            for j in range(0, timeFrame, 1):  # sub-batch
                input_batch = input[:, j:j + miniBatch, ...]

                output_p = model(input_batch)
                output_p = output_p.cpu().detach()

                if j == 0:
                    output[:, j:j + output_channels, ...] = output_p  # + input_batch_mean
                else:
                    output[:, j + output_channels - 1, ...] = output_p[:, -1, ...]
                    output[:, j:j + output_channels - 1, ...] = 0.5 * (
                            output_p[:, 0:-1, ...] + output[:, j:j + output_channels - 1, ...])

                mode = 1
            avg = timings.sum() / timeFrame
            print('\navg={}\n'.format(avg))
            fps = avg / 1000
            print('\nfps={}\n'.format(fps))

            end = time.perf_counter()
            print('Processing time(s): ', end - start)
            output = output[:, 0:t, ::]
            output = output + mean
            output_image = np.squeeze(output.numpy()) * 1.0
            result_name = os.path.join(testSave_dir,
                                       datetime.datetime.now().strftime("%Y%m%d%H%M") + '_' + inputFileNames[i])
            print(result_name)
            root, ext = os.path.splitext(result_name)

            npy_file_name = root + ".npy"
            # np.save(npy_file_name, output_image.astype(np.uint16))
            print(npy_file_name)
            output_image = np.clip(output_image, 0, 65535)
            io.imsave(result_name, output_image.astype(np.uint16), check_contrast = False)
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            pbar.set_description(f'Testing...\t\t{mem}')
    print("Test End")
    pbar.close()
    return npy_file_name
