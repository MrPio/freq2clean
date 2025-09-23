import sys
from pathlib import Path
from careamics import Configuration, CAREamist
import torch

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(str(FILE_DIR.parent))
from src import Recording, clog, cprint, DATASETS

dataset = "synthetic"
max_frames = 6000
patch_t = 32
epochs = 10
bs = 32

# Init
torch.set_float32_matmul_precision("high")
WORK_DIR = Path("n2v")
OUT_DIR = WORK_DIR / "output"
OUT_DIR.mkdir(exist_ok=True, parents=True)
SUFFX = f"n2v_{dataset}_frames{max_frames}_t{patch_t}_ep{epochs}"
cprint(f"magenta:{SUFFX}")

clog("red:Loading Dataset...")
metadata = DATASETS[dataset]
x, gt = (Recording(_, max_frames=max_frames) for _ in [metadata.x, metadata.gt])
config_dict = {
    "experiment_name": SUFFX,
    "algorithm_config": {
        "algorithm": "n2v",
        "loss": "n2v",
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
    },
}
cfg = Configuration(**config_dict)

if (WORK_DIR / f"checkpoints/{SUFFX}.ckpt").exists():
    raise f"Experiment exists! [{SUFFX}]"
else:
    clog("cyan:Starting training...")
    CAREamist(cfg, work_dir=WORK_DIR).train(train_source=x.np)
