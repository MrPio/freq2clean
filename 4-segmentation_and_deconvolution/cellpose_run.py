import sys
import torch
from cellpose import models, plot
from joblib import Parallel, delayed

sys.path.append("..")
from src import clog, Recording, imshow, mkdir, np, tqdm

# Args
dataset = sys.argv[1]
folders = [
    "gt",
    "deepcad_15",
    "fft_15",
    "deepcad_30",
    "fft_30",
    "deepcad_75",
    "fft_75",
    "deepcad_150",
    "fft_150",
    "deepcad_300",
    "fft_300",
]
num_frames = 50
model_names = ["cpsam"]  # "cyto3", "nuclei"] They perform the same

for folder in folders:
    # Init
    clog(f"Processing folder", f"red:{folder}...")
    data_path = f"dataset/{dataset}/{folder}/data.tiff"
    vid = Recording(data_path, max_frames=None)
    frames = list(range(0, vid.frames, vid.frames // num_frames))

    for model_name in model_names:
        clog(f"Running segmentation with", f"green:{model_name}...")
        model = models.CellposeModel(pretrained_model=model_name, gpu=torch.cuda.is_available())

        masks = []
        for frame in tqdm(frames, leave=False):
            mask, _, _ = model.eval(vid.np[frame], diameter=None)
            masks.append(mask)
        masks = np.stack(masks, axis=0)

        clog(f"Segmentation complete. Found", f"blue:{masks.max()}", "neurons.")
        out_dir = mkdir(f"cellpose_results/{dataset}/{folder}")
        suffx = f"{model_name}"
        imshow(
            {"Original Image": vid.np[-1], "Segmented Neurons": plot.mask_rgb(masks[-1])},
            size=16,
            path=out_dir / f"{suffx}.png",
        )
        np.save(out_dir / f"{suffx}_mask.npy", masks)
