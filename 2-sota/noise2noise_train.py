import sys
from pathlib import Path
from careamics import Configuration, CAREamist
import torch
from PIL import Image
from csbdeep.utils import normalize

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(str(FILE_DIR.parent))
from src import *

dataset = "synthetic"
max_frames = 3000
patch_ts = [8, 16, 32, 64, 128]
epochs = 10
bs = 64
test_frame = 100

# Init
torch.set_float32_matmul_precision("high")
WORK_DIR = Path("n2n")
OUT_DIR = mkdir(WORK_DIR / "output")
METRICS_PATH = OUT_DIR / "metrics.csv"

clog("red:Loading Dataset...")
metadata = DATASETS[dataset]
x, gt = (Recording(_, max_frames=max_frames).normalized for _ in [metadata.x, metadata.gt])
x_mean, gt_mean = (np.mean(_) for _ in [x, gt])
x_std, gt_std = (np.std(_) for _ in [x, gt])
df = (
    pd.read_csv(METRICS_PATH, index_col="suffx")
    if METRICS_PATH.exists()
    else pd.DataFrame(columns=["suffx", "PSNR", "SSIM"]).set_index("suffx")
)


def save(suffx, pred, gt):
    clog("yellow:Rendering and evaluating...")
    df.loc[suffx] = [psnr(pred, gt), s := ssim(pred, gt)]
    df.to_csv(METRICS_PATH)
    Image.fromarray(np.clip(normalize(pred, 0.25, 99.9), 0, 1) * 255).convert("RGB").save(OUT_DIR / f"{suffx}.png")


for patch_t in tqdm(patch_ts, desc="patch_t"):
    suffx = f"n2n_{dataset}_frames{max_frames}_t{patch_t}_ep{epochs}"
    if suffx in df.index:
        continue
    config_dict = {
        "experiment_name": suffx,
        "algorithm_config": {
            "algorithm": "n2n",
            "loss": "mse",
            "model": {
                "architecture": "UNet",
            },
        },
        "training_config": {
            "batch_size": bs,
            "num_epochs": epochs,
        },
        "data_config": {
            "data_type": "array",
            "axes": "ZYX",
            "patch_size": [patch_t, 128, 128],
            "image_means": [gt_mean],
            "image_stds": [gt_std],
        },
    }
    cfg = Configuration(**config_dict)

    engine = CAREamist(cfg, work_dir=WORK_DIR)
    engine.train(train_source=x, train_target=gt)
    start, end = test_frame - patch_t // 2, test_frame + patch_t // 2
    pred = engine.predict(x[start:end, :488, :488])[0][0, 0, patch_t // 2]
    save(suffx, pred, gt[test_frame, :488, :488])
