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

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import *

EPOCHS = 10
BS = 16
TRAINSET = "oabf_astro"


def gradient_loss(pred, gt):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    return l1_loss(pred_dx, gt_dx) + l1_loss(pred_dy, gt_dy)


cprint("red:Loading model...")
train_name = datetime.now().strftime("%Y%m%d%H%M")
model = NextFrameUNet(frames=1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

cprint("green:Loading dataset...")
dataset = DatasetNextFrame(DATASETS[TRAINSET], frames_per_patch=1)
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
# TODO:run
for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    with tqdm(dataloader, leave=False, disable=not accelerator.is_main_process) as pbar:
        for x, gt in pbar:
            print(x.shape, gt.shape)
            pred = model(x, torch.zeros(x.shape[0]).to(model.device)).sample

            l1 = l1_loss(pred, gt)
            mse = mse_loss(pred, gt)
            ssim = pytorch_msssim.ssim(pred, gt, data_range=2.0, size_average=True)
            grad = gradient_loss(pred, gt)
            loss = 1 * l1 + 0.5 * (1 - ssim) + 0.25 * grad

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # TQDM logging
            vram_free = get_gpu_memory()[0] / 1024
            delta = (time_ns() - start_time) // 1e9
            h, m, s = delta // 3600, (delta % 3600) // 60, delta % 60
            pbar.set_postfix(
                {
                    f"Loss": f"{loss.item():.2f} (L1:{l1.item():.2f}, SSIM:{ssim.item():.2f}, Grad:{grad.item():.2f}, MSE:{mse.item():.2f})",
                    "Free VRAM": f"{vram_free:.1f}GiB",
                    "Time": f"{h:.0f}h {m:.0f}m {s:.0f}s",
                }
            )
            metrics.loc[len(metrics)] = [loss.item(), l1.item(), ssim.item(), grad.item(), mse.item()]

            # Preview images
            if (len(metrics) - 1) % 50 == 0 and accelerator.is_main_process:
                pil_stack(map(tensor2pil, [x[0], gt[0], pred[0]])).save(f"pth/{train_name}/snaps/{len(metrics)}.png")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            metrics.to_parquet(pth_dir / f"metrics.parquet")
            cprint(f"cyan:Saving checkpoint [{epoch+1}/{EPOCHS}]")
            torch.save(model.state_dict(), pth_dir / f"{epoch}.pt")
