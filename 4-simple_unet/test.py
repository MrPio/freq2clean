"""
Usage:
python tester.py \
    --checkpoint="pth/202508041042/9.pt" \
"""

import argparse
from datetime import datetime, timedelta
from time import time_ns
import numpy as np
import torch
from diffusers import DDPMScheduler
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from pathlib import Path
import pandas as pd
from torchvision.transforms import ToPILImage
import torch
import pytorch_msssim
from torch.nn.functional import l1_loss, mse_loss

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the .pt file")
    return parser.parse_args()


BS = 16
args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = Path("results") / args.checkpoint.parts[-2]
out_dir.mkdir(parents=True, exist_ok=True)

cprint("red:Loading model...")
train_name = datetime.now().strftime("%Y%m%d%H%M")
model = DeepCADImprovementUNet()
state_dict = torch.load(args.checkpoint, map_location="cpu")
model.load_state_dict(state_dict)
model.to(device).eval()

cprint("green:Loading dataset...")
recording = Recording(DATASETS["oabf_astro"] / "y.tiff")

cprint("yellow:Starting training...")
start_time = time_ns()
preds = np.empty_like(recording.np)
# TODO: batch x
for i in tqdm(range(recording.frames), desc="Frames"):
    x = recording[i]
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = x * 2 - 1
    with torch.no_grad():
        pred = model(x, torch.zeros(x.shape[0]).to(model.device)).sample
        preds[i] = pred[0].numpy()

Recording(preds).render(out_dir / f"pred.mp4")
