"""
Usage: python test.py --checkpoint="pth/202508051638/4.pt"
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
from csbdeep.utils import normalize
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the .pt file")
    return parser.parse_args()


BS = 8
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
ds = NoisyCleanDataset(name="oabf_astro")
rec = Recording(DATASETS["oabf_astro"] / "y.tiff", max_frames=300)
frames = rec.frames
rec = rec.np

cprint("yellow:Starting training...")
start_time = time_ns()
preds = np.empty_like(rec, dtype=np.float32)
for i in tqdm(range(0, frames, BS), desc="Frames"):
    x = rec[i : i + BS]
    x = ds.clean_transforms(x).unsqueeze(1).to(model.device)
    with torch.no_grad():
        pred = model(x, torch.zeros(x.shape[0]).to(model.device)).sample
        preds[i : i + BS] = pred.cpu().squeeze(1).numpy()

pred = Recording(preds)
# pred.save_sample(out_dir / f"pred.tiff")
pred.render(out_dir / f"pred.mp4")
