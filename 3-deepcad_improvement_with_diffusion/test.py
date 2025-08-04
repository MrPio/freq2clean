"""
Usage:
python tester.py \
    --checkpoint="pth/202508041042/9.pt" \
    --dataset="dataset/astro_1" \
    --steps=120
"""

import argparse
from pathlib import Path
import sys
import torch
from diffusers import DDPMScheduler, DDIMScheduler
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import *


def parse_args():
    parser = argparse.ArgumentParser(description="Load a DDPM checkpoint and run one denoising step")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to the .pt file (e.g. pth/202508030044/99.pt)"
    )
    parser.add_argument("--dataset", type=Path, default="dataset/astro", help="Testset folder")
    parser.add_argument("--ddpm_steps", type=int, default=1000, help="Number of DDPM timesteps (must match training)")
    parser.add_argument("--steps", type=int, default=50, help="Number of DDIM timesteps (must match training)")
    args = parser.parse_args()
    return args


args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = Path("results") / args.checkpoint.parts[-2]
for folder in ["noisy", "cond", "pred"]:
    (out_dir / folder).mkdir(parents=True, exist_ok=True)

cprint("red:Loading model...")
model = DiffDenoiseUNet()
# model = torch.nn.DataParallel(model)
# scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_steps)
scheduler = DDIMScheduler(num_train_timesteps=args.ddpm_steps, beta_schedule="linear", clip_sample=False)
scheduler.set_timesteps(args.steps)

state_dict = torch.load(args.checkpoint, map_location="cpu")
model.load_state_dict(state_dict)
model.to(device).eval()

cprint("green:Loading dataset...")
dataset = Dataset2PM(str(args.dataset), augument=False)
# indexes = np.random.choice(range(len(dataset)), size=args.samples)
indexes = [0, 1_000]


def predict(sample, cond):
    for j, t in enumerate(tqdm(scheduler.timesteps, leave=False)):
        model_input = torch.cat([sample, cond], dim=1)
        with torch.no_grad():
            noise_pred = model(model_input, t.unsqueeze(0).to(device)).sample

        # obtain the denoised image at t−1
        # step_out = scheduler.step(noise_pred, t, sample)
        step_out = scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=sample,
            eta=0.0,
        )
        sample = step_out.prev_sample
        # if j % 10 == 0:
        #     tensor2pil(sample[0]).save(out_dir / "pred" / f"{i}_{j}.png")

    return sample


cprint("yellow:Starting testing...")
for i in tqdm(indexes, desc="Processing images"):
    noisy, cond = dataset[i]
    noisy = noisy.unsqueeze(0).to(device)  # [1,1,512,512]
    cond = cond.unsqueeze(0).to(device)  # [1,1,512,512]

    eps = torch.randn_like(noisy)
    more_noisy = scheduler.add_noise(noisy, eps, torch.tensor([args.ddpm_steps - 1], device=device))

    pred1 = predict(more_noisy, cond)
    pred2 = predict(-more_noisy, cond)
    pred = (pred1 + pred2) / 2

    np.save(out_dir / "pred" / f"{i}.npy", pred[0].cpu().numpy())
    np.save(out_dir / "pred" / f"{i}_1.npy", pred1[0].cpu().numpy())
    tensor2pil(more_noisy[0]).save(out_dir / "noisy" / f"{i}.png")
    tensor2pil(pred[0]).save(out_dir / "pred" / f"{i}.png")
