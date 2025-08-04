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

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import *

EPOCHS = 10
BS = 16
TRAINSET = "dataset/astro_192"

cprint("red:Loading model...")
train_name = datetime.now().strftime("%Y%m%d%H%M")
model = DeepCADImprovementUNet()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

cprint("green:Loading dataset...")
dataset = Dataset2PM(TRAINSET, augument=False)
dataloader = DataLoader(dataset, batch_size=BS, shuffle=True, num_workers=1)

cprint("blue:Loading accelerator...")
accelerator = Accelerator()
print(f"🚀 Accelerator launching on {accelerator.num_processes} GPU(s)")
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

metrics = pd.DataFrame(columns=["L1", "SSIM"])
start_time = time_ns()
pth_dir = Path(f"pth/{train_name}")
(pth_dir / "snaps").mkdir(parents=True, exist_ok=True)

cprint("yellow:Starting training...")
for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    with tqdm(dataloader, leave=False, disable=not accelerator.is_main_process) as pbar:
        for gt, x in pbar:
            pred = model(x, torch.zeros(BS)).sample

            l1 = torch.nn.functional.l1_loss(pred, gt)
            ssim = pytorch_msssim.ssim(pred, gt, data_range=2.0, size_average=True)
            loss = l1 + 0.5 * (1 - ssim)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # TQDM logging
            vram_free = get_gpu_memory()[0] / 1024
            delta = (time_ns() - start_time) // 1e9
            h, m, s = delta // 3600, (delta % 3600) // 60, delta % 60
            pbar.set_postfix(
                {
                    f"Loss": f"{loss.item():.2f} ({l1.item():.2f}, {ssim.item():.2f})",
                    "Free VRAM": f"{vram_free:.1f}GiB",
                    "Time": f"{h:.0f}h {m:.0f}m {s:.0f}s",
                }
            )
            metrics.loc[len(metrics)] = [loss.item(), ssim.item()]

            # Preview images
            if (len(metrics) - 1) % 50 == 0 and accelerator.is_main_process:
                pil_stack(map(tensor2pil, [x[0], gt[0], pred[0]])).save(f"pth/{train_name}/snaps/{len(metrics)}.png")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            metrics.to_parquet(pth_dir / f"metrics.parquet")
            cprint(f"cyan:Saving checkpoint [{epoch+1}/{EPOCHS}]")
            torch.save(model.state_dict(), pth_dir / f"{epoch}.pt")
