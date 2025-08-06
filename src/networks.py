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


class NextFramesUNet(UNet3DConditionModel):
    def __init__(self, patch_xy, cross_attention_dim: int = 128):
        super().__init__(
            in_channels=1,
            out_channels=1,
            sample_size=patch_xy,
            block_out_channels=(16, 32, 64, 128),
            layers_per_block=2,
            down_block_types=(
                "DownBlock3D",
                "DownBlock3D",
                "DownBlock3D",
                "CrossAttnDownBlock3D",
            ),
            up_block_types=(
                "CrossAttnUpBlock3D",
                "UpBlock3D",
                "UpBlock3D",
                "UpBlock3D",
            ),
            cross_attention_dim=cross_attention_dim,
            norm_num_groups=8,
        )


class VideoEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (B, 1, Frames, H, W)
        x = self.net(x)  # (B, Embed_dim, Frames, H, W)
        B, E, F, H, W = x.shape
        x = x.reshape(B, E, F * H * W)  # (B, D, Seq)
        x = x.permute(0, 2, 1)  # (B, Seq, D)
        return x
