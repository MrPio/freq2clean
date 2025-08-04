from datetime import datetime, timedelta
from time import time_ns
import torch
from diffusers import DDPMScheduler
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from pathlib import Path
import pandas as pd
from torchvision.transforms import ToPILImage
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import *

EPOCHS = 10
BS = 14
TRAINSET = "dataset/astro_192"
DDPM_STEPS = 1_000


def remove_noise(noisy_t: torch.Tensor, 
                 eps: torch.Tensor, 
                 t: torch.LongTensor, 
                 scheduler) -> torch.Tensor:
    """
    Invert scheduler.add_noise: recover the clean `noisy` from `noisy_t`.
    
    Args:
        noisy_t:  Tensor of shape (B, C, H, W), the noisy image at time t.
        eps:      Tensor of shape (B, C, H, W), the noise that was added.
        t:        LongTensor of shape (B,) or scalar, the timesteps.
        scheduler: an instance of DDPMScheduler (or similar) with attributes:
                   - alphas_cumprod: 1D torch.Tensor of length num_train_timesteps
    Returns:
        Tensor of same shape as noisy_t, the estimated original image.
    """
    # gather the scalars for each batch element
    # alphas_cumprod is a 1D tensor of shape (num_train_timesteps,)
    alpha_prod = scheduler.alphas_cumprod.to(noisy_t.device)
    
    # alpha_prod_t and one_minus_alpha_t have shape (B,)
    alpha_prod_t      = alpha_prod[t]
    one_minus_alpha_t = 1.0 - alpha_prod_t
    
    # take square roots
    sqrt_alpha_prod      = torch.sqrt(alpha_prod_t)       # (B,)
    sqrt_one_minus_alpha = torch.sqrt(one_minus_alpha_t)  # (B,)
    
    # reshape to broadcast over (C, H, W)
    for _ in range(noisy_t.ndim - 1):
        sqrt_alpha_prod      = sqrt_alpha_prod.unsqueeze(-1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
    
    # invert the noising
    noisy = (noisy_t - sqrt_one_minus_alpha * eps) / sqrt_alpha_prod
    return noisy


cprint("red:Loading model...")
train_name = datetime.now().strftime("%Y%m%d%H%M")
model = ConditionedUNet(
    sample_size=512,  # 512×512
    block_out_channels=(32, 64, 128, 256),
    # block_out_channels=(64, 128, 256, 512),
    layers_per_block=3,
    down_block_types=("DownBlock2D",) * 4,
    up_block_types=("UpBlock2D",) * 4,
)
noise_scheduler = DDPMScheduler(num_train_timesteps=DDPM_STEPS)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

cprint("green:Loading dataset...")
dataset = Dataset2PM(TRAINSET)
dataloader = DataLoader(dataset, batch_size=BS, shuffle=True, num_workers=1)

cprint("blue:Loading accelerator...")
accelerator = Accelerator()
print(f"🚀 Accelerator launching on {accelerator.num_processes} GPU(s)")
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

metrics = pd.DataFrame(columns=["MSE", "Free VRAM"])
start_time = time_ns()
pth_dir = Path(f"pth/{train_name}")
(pth_dir / "snaps").mkdir(parents=True, exist_ok=True)

cprint("yellow:Starting training...")
for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    with tqdm(dataloader, leave=False, disable=not accelerator.is_main_process) as pbar:
        for noisy, cond in pbar:
            # clean, cond: [B,1,512,512]
            eps = torch.randn_like(noisy)
            t = torch.randint(0, DDPM_STEPS, (noisy.size(0),), device=noisy.device)
            noisy_t = noise_scheduler.add_noise(noisy, eps, t)

            # input: [B,2,512,512]
            model_input = torch.cat([noisy_t, cond], dim=1)
            noise_pred = model(model_input, t).sample

            loss = torch.nn.functional.l1_loss(noise_pred, eps)
            # loss = []
            # for i in range(noisy.size(0)):
            #     sample = noise_scheduler.step(noise_pred, t[i], noisy_t).prev_sample
            #     loss.append(torch.nn.functional.l1_loss(noisy, sample))
            # loss = sum(loss)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # TQDM logging
            vram_free = get_gpu_memory()[0] / 1024
            delta = (time_ns() - start_time) // 1e9
            h, m, s = delta // 3600, (delta % 3600) // 60, delta % 60

            pbar.set_postfix(
                {
                    f"Loss": f"{loss.item():.3f}",
                    "Free VRAM": f"{vram_free:.1f}GiB",
                    "Time": f"{h:.0f}h {m:.0f}m {s:.0f}s",
                }
            )
            metrics.loc[len(metrics)] = [loss.item(), vram_free]

            # Preview images
            if (len(metrics) - 1) % 50 == 0 and accelerator.is_main_process:
                recovered = remove_noise(noisy_t, noise_pred, t, noise_scheduler)
                tensor2pil(recovered[0]).save(f"pth/{train_name}/snaps/pred_{len(metrics)}.png")
                tensor2pil(noisy[0]).save(f"pth/{train_name}/snaps/noisy_{len(metrics)}.png")
                tensor2pil(noisy_t[0]).save(f"pth/{train_name}/snaps/noisy_t_{len(metrics)}_{t.cpu()[0].item()}.png")
                tensor2pil(cond[0]).save(f"pth/{train_name}/snaps/cond_{len(metrics)}.png")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            metrics.to_parquet(pth_dir / f"metrics.parquet")
            cprint(f"cyan:Saving checkpoint [{epoch+1}/{EPOCHS}]")
            torch.save(model.state_dict(), pth_dir / f"{epoch}.pt")
