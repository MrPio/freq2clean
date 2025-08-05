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
import pytorch_msssim
from torch.nn.functional import l1_loss, mse_loss
from loss import lf_hf_tv_loss

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import *

EPOCHS = 10
SAVE_EVERY = 250
BS = 16


def gradient_loss(pred, gt):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    return l1_loss(pred_dx, gt_dx) + l1_loss(pred_dy, gt_dy)


cprint("red:Loading model...")
train_name = datetime.now().strftime("%Y%m%d%H%M")
model = DeepCADImprovementUNet()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

cprint("green:Loading dataset...")
dataset = NoisyCleanDataset(name="oabf_astro", augument=False)
dataloader = DataLoader(dataset, batch_size=BS, shuffle=True, num_workers=1)

cprint("blue:Loading accelerator...")
accelerator = Accelerator()
print(f"🚀 Accelerator launching on {accelerator.num_processes} GPU(s)")
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

metrics = pd.DataFrame(columns=["TotalLoss", "L1", "SSIM", "Gradient Loss", "MSE"])
start_time = time_ns()
pth_dir = Path(f"pth/{train_name}")
(pth_dir / "snaps").mkdir(parents=True, exist_ok=True)

cprint("yellow:Starting training...")
checkpoints = 0
for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    with tqdm(dataloader, leave=False, disable=not accelerator.is_main_process) as pbar:
        for gt, x in pbar:
            pred = model(x, torch.zeros(x.shape[0]).to(model.device)).sample

            l1 = l1_loss(pred, gt)
            mse = mse_loss(pred, gt)
            ssim = pytorch_msssim.ssim(pred, gt, data_range=2.0, size_average=True)
            grad = gradient_loss(pred, gt)
            # loss = 1 * mse + 0.1 * l1 + 0.55 * (1 - ssim) + 0.1 * grad
            loss = lf_hf_tv_loss(pred, x, gt, lambda_hf=1.85, lambda_tv=1.25e-4)

            # ssim_x = pytorch_msssim.ssim(pred, x, data_range=2.0, size_average=True)
            # loss = mse_loss(pred, x) + (1 - ssim_x) + 0.5 * l1_loss(pred, gt)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # TQDM logging
            vram_free = get_gpu_memory()[0] / 1024
            delta = (time_ns() - start_time) // 1e9
            h, m, s = delta // 3600, (delta % 3600) // 60, delta % 60
            pbar.set_postfix(
                {
                    f"Loss": f"{loss.item():.3f} (L1:{l1.item():.2f}, SSIM:{ssim.item():.2f}, Grad:{grad.item():.2f}, MSE:{mse.item():.2f})",
                    "Free VRAM": f"{vram_free:.1f}GiB",
                    "Time": f"{h:.0f}h {m:.0f}m {s:.0f}s",
                }
            )
            metrics.loc[len(metrics)] = [loss.item(), l1.item(), ssim.item(), grad.item(), mse.item()]

            # Save preview image and metrics
            if (len(metrics) - 1) % 50 == 0 and accelerator.is_main_process:
                metrics.to_parquet(pth_dir / f"metrics.parquet")
                pil_stack(map(tensor2pil, [gt[0], x[0], pred[0]])).save(f"pth/{train_name}/snaps/{len(metrics)}.png")

            # Save checkpoint
            if (len(metrics)) % SAVE_EVERY == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoints += 1
                    cprint(f"cyan:\nSaving checkpoint [{checkpoints+1}]")
                    torch.save(model.state_dict(), pth_dir / f"{checkpoints}.pt")
