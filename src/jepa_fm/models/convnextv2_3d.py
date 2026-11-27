import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from torch.utils.checkpoint import checkpoint

from .utils import GRN3D, LayerNorm3D


class Block3D(nn.Module):
    """ConvNeXtV2 3D block.

    Args:
        dim (int): Number of input channels.
        kernel_size (int): Depthwise kernel size, applied to all three axes.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, kernel_size=3, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv3d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )  # depthwise conv
        self.norm = LayerNorm3D(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1x1 convs
        self.act = nn.GELU()
        self.grn = GRN3D(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, D, H, W) -> (N, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)  # (N, D, H, W, C) -> (N, C, D, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2_3D(nn.Module):
    """ConvNeXt V2 3D

    Args:
        in_chans (int): Number of input channels. Default: 1
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 27, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        kernel_sizes (tuple(int)): Depthwise kernel size at each stage. Default: [3, 5, 5, 3]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
    """

    def __init__(
        self,
        in_chans=1,
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        kernel_sizes=[3, 5, 5, 3],
        drop_path_rate=0.0,
        checkpoint_blocks=False,
    ):
        super().__init__()
        self.depths = depths
        self.checkpoint_blocks = checkpoint_blocks
        self.downsample_layers = (
            nn.ModuleList()
        )  # 2x2x2 patchify stem and 3 downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=2, stride=2),
            LayerNorm3D(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm3D(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block3D(
                        dim=dims[i],
                        kernel_size=kernel_sizes[i],
                        drop_path=dp_rates[cur + j],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            for block in self.stages[i]:
                x = (
                    checkpoint(block, x, use_reentrant=False)
                    if self.checkpoint_blocks
                    else block(x)
                )
        return x  # (N, C, D / 16, H / 16, W / 16)


def convnextv2_tiny_3d(**kwargs):
    model = ConvNeXtV2_3D(dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextv2_base_3d(**kwargs):
    model = ConvNeXtV2_3D(dims=[128, 256, 512, 1024], **kwargs)
    return model


def convnextv2_large_3d(**kwargs):
    model = ConvNeXtV2_3D(dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnextv2_huge_3d(**kwargs):
    model = ConvNeXtV2_3D(dims=[352, 704, 1408, 2816], **kwargs)
    return model
