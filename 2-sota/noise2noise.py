import sys
from pathlib import Path
from careamics import Configuration, CAREamist

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(str(FILE_DIR.parent))
from src import *

dataset = "synthetic"
max_frames = 300
patch_t = 128
epochs = 10

# Init
WORK_DIR = Path("n2n")
OUT_DIR = WORK_DIR / "output"
OUT_DIR.mkdir(exist_ok=True, parents=True)
SUFFX = f"{dataset}_frames{max_frames}_t{patch_t}_ep{epochs}"

clog("red:Loading Dataset...")
metadata = DATASETS[dataset]
x, gt = (Recording(_, max_frames=max_frames) for _ in [metadata.x, metadata.gt])
x_mean, gt_mean = (np.mean(_.np) for _ in [x, gt])
x_std, gt_std = (np.std(_.np) for _ in [x, gt])
config_dict = {
    "experiment_name": "N2N_synthetic",
    "algorithm_config": {
        "algorithm": "n2n",
        "loss": "mse",
        "model": {
            "architecture": "UNet",
        },
    },
    "training_config": {
        "batch_size": 32,
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

clog("cyan:Starting N2V training...")
engine = CAREamist(cfg, work_dir=WORK_DIR)
history = engine.train(train_source=x.np, train_target=gt.np)

clog("green:Predictiong...")
y = engine.predict(x.np[:, :488, :488])
y = y[0].squeeze().squeeze()
np.save(OUT_DIR / f"{SUFFX}.npy")

clog("yellow:Rendering...")
for zoom in [1, 3]:
    imshow([y[i] for i in [0, x.frames // 2, -1]], zoom=zoom, size=8, path=OUT_DIR / f"{SUFFX}_{zoom}x.png")
    Recording(y[:300]).render(OUT_DIR / f"{SUFFX}_{zoom}x.mp4", codec="libx264")
