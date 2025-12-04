import sys
from cellSAM import segment_cellular_image, get_model
from dotenv import load_dotenv

sys.path.append("..")
from src import clog, Recording, imshow, mkdir, np, tqdm

# Args
dataset = sys.argv[1]
folders = [
    "gt",
    "x",
    "deepcad_15",
    "fft_15",
    "deepcad_150",
    "fft_150",
]
num_frames = 100
load_dotenv()
model = get_model()

# Init
for folder in folders:
    clog(f"Processing folder", f"red:{folder}...")
    data_path = f"dataset/{dataset}/{folder}/data.tiff"
    vid = Recording(data_path, max_frames=None)
    frames = list(range(0, vid.frames, vid.frames // num_frames))

    masks = []
    for frame in tqdm(frames, leave=False):
        x = vid.np[frame]
        try:
            mask, _, _ = segment_cellular_image(x, model, device="cuda")
        except:
            mask = np.zeros_like(x)
        masks.append(mask)
    masks = np.stack(masks, axis=0)

    clog(f"Segmentation complete. Found", f"blue:{masks.max()}", "neurons.")
    out_dir = mkdir(f"cellsam_results/{dataset}/{folder}")
    imshow(
        {"Original Image": vid.np[-1], "Segmented Neurons": masks[-1]},
        size=16,
        path=out_dir / f"snap.png",
    )
    np.save(out_dir / f"mask.npy", masks)
