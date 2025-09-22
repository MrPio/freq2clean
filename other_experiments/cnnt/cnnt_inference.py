import h5py
import tifffile
from tqdm import tqdm
from CNNT_Microscopy.running_inference import *
from CNNT_Microscopy.utils import *
from CNNT_Microscopy.models.enhancement_model import *
from CNNT_Microscopy.microscopy_dataset import *

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import *


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


dataset = DATASETS["oabf_astro"]
config = DotDict(
    {
        "time": 16,
        "width": [128, 160],
        "height": [128, 160],
        "test_case": "dataset/oabf_astro.h5",
        "batch_size": 2,
        "load_path": "logs/check/08-16-2025_T22-31-02_epoch-29.pth",
        # Defaults
        "blocks": [32, 64, 96],
        "blocks_per_set": 4,
        "n_head": 4,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dropout_p": 0.1,
        "norm_mode": "instance",
        "with_mixer": 1,
        "use_conv_3D": False,
        "loss": ["mse", "ssim"],
        "loss_weights": [0.1, 1.0],
        "optim": "adamw",
        "global_lr": 5e-4,
        "weight_decay": 0.1,
        "beta1": 0.90,
        "beta2": 0.95,
        "no_w_decay": False,
        "clip_grad_norm": 1.0,
        "scheduler": "ReduceLROnPlateau",
        "per_scaling": False,
        "im_value_scale": [0, 65536],
        "valu_thres": 0.002,
        "area_thres": 0.25,
        "run_name": None,
        "run_notes": None,
        "skip_LSUV": False,
        "no_residual": False,
        "train_only": False,
        "fine_samples": -1,
        "time_scale": 0,
    }
)
cutout = (config.time, config.height[-1], config.width[-1])
overlap = (config.time // 4, config.height[-1] // 4, config.width[-1] // 4)

h5file = h5py.File(config.test_case, libver="latest", mode="r")
cprint("The available footages are:\n", *[f"rand:{_}" for _ in h5file.keys()])
test_set = MicroscopyDataset(
    h5files=[h5file],
    keys=list(h5file.keys()),
    time_cutout=config.time,
    cutout_shape=(config.width[0], config.height[0]) * 2,
    im_value_scale=config.im_value_scale,
)
model = CNNT_enhanced_denoising_runtime(config, None)

for key in tqdm(list(h5file.keys())):
    raw, gt = (np.array(h5file[key][_]).astype(np.float32) for _ in ["noisy_im", "clean_im"])

    pred, _ = running_inference(model, raw, cutout=cutout, overlap=overlap, batch_size=2, device="cuda")
    tifffile.imwrite(f"{key}.tif", np.stack([raw, pred, gt]), imagej=True)
