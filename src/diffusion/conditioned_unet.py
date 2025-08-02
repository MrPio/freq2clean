from diffusers.models import UNet2DModel
import torch.nn as nn


class ConditionedUNet(UNet2DModel):
    def __init__(self, **kwargs):
        super().__init__(in_channels=2, out_channels=1, **kwargs)
