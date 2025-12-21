# Simulated calcium imaging data sampled at 0.3 Hz
cd SRDTrans
python test.py --datasets_folder synthetic_1hz --denoise_model cad_03hz --GPU 0 --patch_x 160 --patch_t 128

# --datasets_folder: the folder containing the data to be processed (one or more *.tif stacks)
# --denoise_model: the subfolder (under pth/) containing pre-trained models (e.g., ad_03hz).
# --GPU: specify the GPU(s) used for inference. (e.g., '0', '0,1', '0,1,2')
# --patch_x, --patch_t: patch size in three dimensions (xy and t), should be divisible by 8.