"""
Usage: python test_stack.py --checkpoint="pth/202508061211/11.pt" --steps=60
"""

import argparse
from datetime import datetime, timedelta
from time import time_ns
import numpy as np
import torch
from diffusers import DDIMScheduler
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
    parser.add_argument("--dataset", type=str, default="oabf_astro", help="Testset folder")
    parser.add_argument("--steps", type=int, default=50, help="Number of DDIM timesteps (must match training)")
    return parser.parse_args()


def predict(even, odd, i):
    # even_enc = encoder(even)
    even_enc = torch.zeros(even.size(0), 1, model.config.cross_attention_dim, device=odd.device, dtype=odd.dtype)
    for j, t in enumerate(tqdm(scheduler.timesteps, leave=False)):
        with torch.no_grad():
            model_input = torch.cat([odd, even], dim=1)
            eps_pred = model(sample=model_input, timestep=t, encoder_hidden_states=even_enc).sample
            odd = scheduler.step(model_output=eps_pred, timestep=t, sample=odd).prev_sample
        if (j + 1) % 10 == 0:
            pil_stack(
                [
                    pil_stack(map(tensor2pil, [odd[0, :, d] for d in range(odd.size(2))])),
                    pil_stack(map(tensor2pil, [even[0, :, d] for d in range(even.size(2))])),
                ],
                horizontally=False,
            ).save(out_dir / "pred" / f"{i}_{j+1}.png")

    return odd


BS = 1
FRAMES_PER_PATCH = 64
PATCH_XY = 64
EMBED_DIM = 512
DDPM_STEPS = 1_000

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = Path("results") / args.checkpoint.parts[-2]
(out_dir / "pred").mkdir(parents=True, exist_ok=True)
encoder_checkpoint = args.checkpoint.parent / f"enc_{args.checkpoint.stem}.pt"

cprint("red:Loading model...")
train_name = datetime.now().strftime("%Y%m%d%H%M")
model = NextFramesUNetStacked(patch_xy=PATCH_XY)
# encoder = VideoEncoder(embed_dim=EMBED_DIM)
model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
# encoder.load_state_dict(torch.load(encoder_checkpoint, map_location="cpu"))
scheduler = DDIMScheduler(num_train_timesteps=DDPM_STEPS, beta_schedule="linear", clip_sample=False)
scheduler.set_timesteps(args.steps)

cprint("green:Loading dataset...")
dataset = NoisyDataset(
    name=args.dataset,
    patch_xy=PATCH_XY,
    frames_per_patch=FRAMES_PER_PATCH,
    max_frames=128,
    augument=False,
    overlap=0,
)
dataloader = DataLoader(dataset, batch_size=BS, shuffle=False, num_workers=1)

cprint("blue:Loading accelerator...")
accelerator = Accelerator()
print(f"🚀 Accelerator launching on {accelerator.num_processes} GPU(s)")
# model, encoder, dataloader = accelerator.prepare(model, encoder, dataloader)
model, dataloader = accelerator.prepare(model, dataloader)
model.eval()
# encoder.eval()

cprint("yellow:Starting training...")
start_time = time_ns()
preds = np.empty_like(dataset.x.np, dtype=np.float32)
patch = 0
with tqdm(dataloader, desc="Patches") as pbar:
    for i, (even, odd) in enumerate(pbar):
        pbar.set_postfix({"VRAM": f"{get_gpu_memory()[0] / 1024:.1f}GiB"})

        odd = scheduler.add_noise(odd, torch.randn_like(odd), torch.tensor([DDPM_STEPS - 1], device=device))
        pred = predict(even, odd, i)  # (B, 1, D, H, W)
        pred = pred.cpu().squeeze(1).numpy()

        for p in range(pred.shape[0]):
            z, y, x = dataset.idx2pos(patch)
            patch += 1
            preds[z[0] : z[1] : 2, y[0] : y[1], x[0] : x[1]] = pred[p]
            preds[z[0] + 1 : z[1] : 2, y[0] : y[1], x[0] : x[1]] = pred[p]

pred = Recording(preds)
pred.render(out_dir / f"pred.mp4")
