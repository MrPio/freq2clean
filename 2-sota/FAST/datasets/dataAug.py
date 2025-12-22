import numpy as np
import random
import torch
import time
from skimage import io




def randomTransform(input,label):
    pass

def add_sCMOS_noise(image, readout_noise_std=1.5, fpn_std=0.01, dark_current_rate=0.02,
                    electron_gain=0.5, adc_bits=16, exposure_time=1.0):
    """
    Add sCMOS noise to an image based on Zyla 4.2 sCMOS parameters.

    Parameters:
    - image: The input image as a PyTorch tensor. Assumed to be in [0, 1] range.
    - readout_noise_std: The standard deviation for the readout noise.
    - fpn_std: The standard deviation for the fixed pattern noise.
    - dark_current_rate: The rate of dark current noise.
    - electron_gain: Gain factor for electrons.
    - adc_bits: Bit-depth of ADC.
    - exposure_time: Exposure time for the image.

    Returns:
    - Noisy image as a PyTorch tensor. Values might be outside [0, 1] due to noise.
    """

    # Convert PyTorch tensor to NumPy array
    image = image.cpu().numpy() / 255.0

    # Fixed Pattern Noise (FPN)
    fpn = np.random.normal(0, fpn_std, image.shape)

    # Readout Noise
    readout_noise = np.random.normal(0, readout_noise_std / 255.0, image.shape)

    # Dark Current Noise
    dark_noise = np.random.poisson(dark_current_rate * exposure_time, image.shape) / 255.0

    # Electron Gain Noise (Assuming it's multiplicative. This might need adjustment.)
    image = image * electron_gain

    # ADC Quantization Noise
    adc_quantization_noise = (1.0 / (2 ** adc_bits)) * np.random.rand(*image.shape)

    # Add noise components
    # noisy_image = image + fpn + readout_noise + dark_noise + adc_quantization_noise
    noisy_image = image + dark_noise + adc_quantization_noise
    # # Ensure values are within [0, 1] range
    # noisy_image = np.clip(noisy_image, 0.0, 1.0)

    # Convert back to PyTorch tensor
    noisy_image = torch.from_numpy(noisy_image / 1.0).float() * 255.0

    return noisy_image


def randomAddnoise(input, mode='train'):
    """
    function：add noise
    """
    inputAug = torch.zeros(input.shape[0], input.shape[1], input.shape[2])
    targetAug = torch.zeros(input.shape[0], input.shape[1], input.shape[2])
    for i in range(input.shape[0]):
        if mode == 'train':
            inputFrame = add_sCMOS_noise(input[i, ::])
            targetFrame = add_sCMOS_noise(input[i, ::])
            inputAug[i, :, :] = inputFrame
            targetAug[i, :, :] = targetFrame
        else:
            inputAug = input
            targetAug = input
    if mode == 'train':
        io.imsave(r'E:\Denoising\Deep3D\data\train\1-aug\input_aug.tif', inputAug.numpy(), check_contrast=False)
        io.imsave(r'E:\Denoising\Deep3D\data\train\1-aug\target_aug.tif', targetAug.numpy(), check_contrast=False)
    return inputAug, targetAug


if __name__ == '__main__':
    input = np.zeros([512, 512])
    input = torch.tensor(input)
    t = 10
    T1 = time.time()
    for i in range(10):
        inputAug, targetAug = randomTransform(input, t)
        print(inputAug.shape)
        # print(np.argwhere(inputAug == 1))
    T2 = time.time()
    print((T2 - T1))
