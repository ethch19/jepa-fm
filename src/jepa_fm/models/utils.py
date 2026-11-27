import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm3D(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class LayerNorm2D(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN3D(nn.Module):
    """3D GRN (Global Response Normalisation) layer. Normalised over channel mean"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        gx = torch.linalg.vector_norm(
            x, ord=2, dim=(1, 2, 3), keepdim=True, dtype=torch.float32
        )
        nx = (gx / (gx.mean(dim=-1, keepdim=True) + self.eps)).to(dtype)
        return self.gamma * (x * nx) + self.beta + x


class GRN2D(nn.Module):
    """2D GRN (Global Response Normalisation) layer. Normalised over channel mean"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        gx = torch.linalg.vector_norm(
            x, ord=2, dim=(1, 2), keepdim=True, dtype=torch.float32
        )
        nx = (gx / (gx.mean(dim=-1, keepdim=True) + self.eps)).to(dtype)
        return self.gamma * (x * nx) + self.beta + x


def global_average_pooling(net: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """(B, V, *crop_size) crops -> (B, V, C) latents. Uses global average pooling"""
    b, v = imgs.shape[:2]
    features = net(imgs.flatten(0, 1).unsqueeze(1))  # (B*V, C, *crop_size)
    return features.mean(dim=tuple(range(2, features.dim()))).view(b, v, -1)
