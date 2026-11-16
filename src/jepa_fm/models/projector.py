import torch.nn as nn


class Projector(nn.Module):
    """Projector for loss calculations only: Linear -> BatchNorm1d -> GELU -> Linear"""

    def __init__(self, in_dim: int, out_dim: int = 128, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
