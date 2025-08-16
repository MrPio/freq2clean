import h5py
from CNNT_Microscopy.running_inference import *
from CNNT_Microscopy.utils import *
from CNNT_Microscopy.models.enhancement_model import *
from CNNT_Microscopy.microscopy_dataset import *

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import *

dataset = DATASETS["oabf_astro"]
config = {
    "time": 16,
    "size": 128,
    "test_case": "dataset/oabf_astro.h5",
    "batch_size": 2,
    "load_path": "logs/check/08-10-2025_T19-47-09_epoch-60.pth",
}

h5file = h5py.File(config["test_case"], libver="latest", mode="r")
test_set = MicroscopyDataset(
    h5files=[h5file],
    keys=list(h5file.keys()),
    time_cutout=config.time,
    cutout_shape=(config.size,) * 2,
    im_value_scale=[0, 2**16 - 1],
)

model = CNNT_enhanced_denoising_runtime(config, None)

output, _ = running_inference(model, x, batch_size=2)
output = normalize_image(output, values=(0, 1), clip=True)
