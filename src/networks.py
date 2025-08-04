from diffusers.models import UNet2DModel, UNet2DConditionModel, UNet3DConditionModel
import torch.nn as nn


# TODO: Cerca Conditioned da diffusers
class DiffDenoiseUNet(UNet2DModel):
    """Has ≈12.8 million params."""

    def __init__(self):
        super().__init__(
            in_channels=2,
            out_channels=1,
            sample_size=512,
            block_out_channels=(32, 64, 128, 256),
            layers_per_block=3,
            down_block_types=("DownBlock2D",) * 4,
            up_block_types=("UpBlock2D",) * 4,
        )


class DeepCADImprovementUNet(UNet2DModel):
    def __init__(self):
        super().__init__(
            in_channels=1,
            out_channels=1,
            sample_size=512,
            block_out_channels=(32, 64, 128, 256),
            layers_per_block=2,
            down_block_types=("DownBlock2D",) * 4,
            up_block_types=("UpBlock2D",) * 4,
            add_attention=False,
            # norm_num_groups=16,
        )


class NextFrameUNet(UNet2DModel):
    def __init__(self, frames=1):
        super().__init__(
            in_channels=frames,
            out_channels=frames,
            sample_size=512,
            block_out_channels=(16, 32, 64, 128),  # (32, 64, 128, 256),
            layers_per_block=2,
            down_block_types=("DownBlock2D",) * 4,
            up_block_types=("UpBlock2D",) * 4,
            add_attention=False,
            norm_num_groups=16,
        )
