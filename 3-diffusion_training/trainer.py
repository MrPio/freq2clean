from datetime import datetime
import torch
from diffusers import DDPMScheduler
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import *

EPOCHS = 100
BS = 2
TRAINSET = "dataset/astro"
DDPM_STEPS = 1_000

cprint("red:Loading model...")
train_name = datetime.now().strftime("%Y%m%d%H%M")
model = ConditionedUNet(
    sample_size=512,  # 512×512
    block_out_channels=(64, 128, 256, 512),
    layers_per_block=2,
    down_block_types=("DownBlock2D",) * 4,
    up_block_types=("UpBlock2D",) * 4,
)
noise_scheduler = DDPMScheduler(num_train_timesteps=DDPM_STEPS)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

cprint("green:Loading dataset...")
dataset = Dataset2PM(TRAINSET)
dataloader = DataLoader(dataset, batch_size=BS, shuffle=True, num_workers=4)

cprint("blue:Loading accelerator...")
accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

cprint("yellow:Starting training...")
for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    for noisy, cond in tqdm(dataloader, leave=False):
        # clean, cond: [B,1,512,512]
        t = torch.randint(0, DDPM_STEPS, (noisy.size(0),), device=noisy.device)
        more_noisy = noise_scheduler.add_noise(noisy, torch.randn_like(noisy), t)

        # input: [B,2,512,512]
        model_input = torch.cat([noisy, cond], dim=1)

        noise_pred = model(model_input, t).sample
        loss = torch.nn.functional.mse_loss(more_noisy - noise_pred, noisy)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        cprint(f"cyan:Saving checkpoint [{epoch+1}/{EPOCHS}]")
        torch.save(model.state_dict(), f"pth/{train_name}_{epoch}.pt")
