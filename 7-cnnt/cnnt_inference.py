import h5py
from CNNT_Microscopy.running_inference import *
from CNNT_Microscopy.utils import *
from CNNT_Microscopy.models.enhancement_model import *
from CNNT_Microscopy.microscopy_dataset import *
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import *
 
dataset=DATASETS["dataset/oabf/astro"]
config={
    "time":16,
    "height":[128, 160],
    "width":[128, 160],
    "test_case":"logs/check/08-10-2025_T19-47-09_epoch-60.pth",
    "batch_size":2,
    "load_path":""
}

h5file = h5py.File(file,libver='latest',mode='r')
keys = list(h5file.keys())

test_set = MicroscopyDataset(h5files=[h5file], keys=keys,
                                time_cutout=config.time,
                                cutout_shape=[config.height[0], config.width[0]],
                                rng = None,
                                per_scaling = config.per_scaling,
                                im_value_scale = config.im_value_scale,
                                valu_thres=config.valu_thres,
                                area_thres=config.area_thres,
                                time_scale = config.time_scale)

model = CNNT_enhanced_denoising_runtime(config, None)

output, _ = running_inference(model, x,  batch_size=2)
output = normalize_image(output, values=(0, 1), clip=True)
