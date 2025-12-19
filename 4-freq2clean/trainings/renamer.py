import torch
from pathlib import Path

name_old="alphas"
name_new="mask"

for checkpoint_path in Path(".").rglob("*.pt"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if name_old in state_dict:
        state_dict[name_new] = state_dict.pop(name_old)

    if "state_dict" in checkpoint:
        checkpoint["state_dict"] = state_dict
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved new checkpoint to {checkpoint_path}")
