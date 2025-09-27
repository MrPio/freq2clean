import sys
import torch
from cellpose import models, plot

sys.path.append("..")
from src import clog, Recording, imshow, mkdir, np

# Args
dataset = sys.argv[1]
folder = sys.argv[2]
num_frames = 20
model_names = ["cpsam", "cyto3", "nuclei"]

# Init
data_path = f"dataset/{dataset}/{folder}/data.tiff"
img = Recording(data_path, max_frames=None)
frames = list(range(0, img.frames, img.frames // num_frames))
img = img.np[frames]

for model_name in model_names:
    clog(f"Running segmentation with", f"green:{model_name}...")
    model = models.CellposeModel(pretrained_model=model_name, gpu=torch.cuda.is_available())
    masks, flows, styles = model.eval(img, diameter=None)

    clog(f"Segmentation complete. Found", masks.max(), "neurons.")
    out_dir = mkdir(f"cellpose_results/{dataset}/{folder}")
    suffx = f"{num_frames}_{model_name}"
    imshow(
        {"Original Image": img, "Segmented Neurons": plot.mask_rgb(masks)},
        size=16,
        path=out_dir / f"{suffx}.png",
    )
    np.save(out_dir / f"{suffx}_mask.npy", masks)
