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
import torch
from torch.nn.functional import l1_loss, mse_loss

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import *

EPOCHS = 10
SAVE_EVERY = 100
BS = 1
FRAMES_PER_PATCH = 64
PATCH_XY = 64
TRAINSET = "oabf_astro"
EMBED_DIM = 512
DDPM_STEPS = 1_000
OVERLAP = 0.4

cprint("red:Loading model...")
train_name = datetime.now().strftime("%Y%m%d%H%M")
# model = NextFramesUNet(patch_xy=PATCH_XY, cross_attention_dim=EMBED_DIM)
model = NextFramesUNetStacked(patch_xy=PATCH_XY)
# encoder = VideoEncoder(embed_dim=EMBED_DIM)
scheduler = DDPMScheduler(num_train_timesteps=DDPM_STEPS)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

cprint("green:Loading dataset...")
dataset = NoisyDataset(
    name=TRAINSET,
    patch_xy=PATCH_XY,
    frames_per_patch=FRAMES_PER_PATCH,
    max_frames=None,
    augument=False,
    overlap=OVERLAP,
)
dataloader = DataLoader(dataset, batch_size=BS, shuffle=True, num_workers=1)
cprint("The dataset has", len(dataset), "samples!")
cprint("The augumentation is", f"green:{'ON' if dataset.augument else 'OFF'}")

cprint("blue:Loading accelerator...")
accelerator = Accelerator()
print(f"🚀 Accelerator launching on {accelerator.num_processes} GPU(s)")
# model, encoder, optimizer, dataloader = accelerator.prepare(model, encoder, optimizer, dataloader)
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

metrics = pd.DataFrame(columns=["Total Loss", "L1", "MSE"])
start_time = time_ns()
pth_dir = Path(f"pth/{train_name}")
(pth_dir / "snaps").mkdir(parents=True, exist_ok=True)


def remove_noise(noisy_t: torch.Tensor, eps: torch.Tensor, t: torch.LongTensor, scheduler) -> torch.Tensor:
    # gather the scalars for each batch element
    # alphas_cumprod is a 1D tensor of shape (num_train_timesteps,)
    alpha_prod = scheduler.alphas_cumprod.to(noisy_t.device)

    # alpha_prod_t and one_minus_alpha_t have shape (B,)
    alpha_prod_t = alpha_prod[t]
    one_minus_alpha_t = 1.0 - alpha_prod_t

    # take square roots
    sqrt_alpha_prod = torch.sqrt(alpha_prod_t)  # (B,)
    sqrt_one_minus_alpha = torch.sqrt(one_minus_alpha_t)  # (B,)

    # reshape to broadcast over (C, H, W)
    for _ in range(noisy_t.ndim - 1):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

    # invert the noising
    noisy = (noisy_t - sqrt_one_minus_alpha * eps) / sqrt_alpha_prod
    return noisy


cprint("yellow:Starting training...")
checkpoint = 0
for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    with tqdm(dataloader, leave=False, disable=not accelerator.is_main_process) as pbar:
        for even, odd in pbar:
            # even, odd: (BS, 1, FRAMES_PER_PATCH, 64, 64)
            eps = torch.randn_like(odd)
            t = torch.randint(0, DDPM_STEPS, (even.size(0),), device=odd.device)
            noisy_odd = scheduler.add_noise(odd, eps, t)

            # even_enc = encoder(even)
            even_enc = torch.zeros(
                even.size(0), 1, model.config.cross_attention_dim, device=odd.device, dtype=odd.dtype
            )
            model_input = torch.cat([noisy_odd, even], dim=1)
            eps_pred = model(
                sample=model_input,
                timestep=t,
                encoder_hidden_states=even_enc,
            ).sample

            mse = mse_loss(eps_pred, eps)
            l1 = l1_loss(eps_pred, eps)
            loss = mse

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # TQDM logging
            vram_free = get_gpu_memory()[0] / 1024
            delta = (time_ns() - start_time) // 1e9
            h, m, s = delta // 3600, (delta % 3600) // 60, delta % 60
            pbar.set_postfix(
                {
                    f"Loss": f"{loss.item():.3f} (L1:{l1.item():.2f}, MSE:{mse.item():.2f})",
                    "Free VRAM": f"{vram_free:.1f}GiB",
                    "Time": f"{h:.0f}h {m:.0f}m {s:.0f}s",
                }
            )
            metrics.loc[len(metrics)] = [loss.item(), l1.item(), mse.item()]

            if (len(metrics) - 1) % 50 == 0 and accelerator.is_main_process:
                metrics.to_parquet(pth_dir / f"metrics.parquet")
                recovered = remove_noise(noisy_odd, eps_pred, t, scheduler)
                pil_stack(map(tensor2pil, [even[0, :, 0], noisy_odd[0, :, 0], recovered[0, :, 0]])).save(
                    f"pth/{train_name}/snaps/{len(metrics)}_({t[0].item()}).png"
                )

            # Save checkpoint
            if (len(metrics)) % SAVE_EVERY == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoint += 1
                    cprint(f"cyan:\nSaving checkpoint [{checkpoint+1}]")
                    torch.save(model.state_dict(), pth_dir / f"{checkpoint}.pt")
                    # torch.save(encoder.state_dict(), pth_dir / f"enc_{checkpoint}.pt")
