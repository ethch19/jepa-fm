import torch.nn as nn


class AdaLNBlock(nn.Module):
    """Residual MLP block with adaLN-Zero conditioning"""

    def __init__(self, dim: int, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.mod = nn.Linear(cond_dim, 3 * dim)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, cond):
        """(B, T, dim), (B, cond_dim) -> (B, T, dim)"""
        gamma, beta, gate = self.mod(cond).chunk(3, dim=-1)
        gamma, beta, gate = gamma[:, None], beta[:, None], gate[:, None]
        return x + gate * self.mlp(self.norm(x) * (1 + gamma) + beta)


class ConditioningEncoder(nn.Module):
    def __init__(self, cond_dim: int, cond_hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, cond_hidden_dim),
            nn.GELU(),
            nn.Linear(cond_hidden_dim, cond_hidden_dim),
        )

    def forward(self, cond):
        return self.net(cond)


class ConditionedPredictor(nn.Module):
    """``out = x + predictor(x, cond)``. ``x`` is (B, dim) or (B, T, dim)."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        depth: int = 3,
        hidden_dim: int | None = None,
        cond_hidden_dim: int = 1024,
    ):
        super().__init__()
        self.cond_enc = ConditioningEncoder(cond_dim, cond_hidden_dim)
        self.blocks = nn.ModuleList(
            AdaLNBlock(dim, hidden_dim or dim, cond_hidden_dim) for _ in range(depth)
        )

    def forward(self, x, cond):
        """(B, dim), (B, cond_dim) -> (B, dim); (B, T, dim) keeps its T axis."""
        flat = x.dim() == 2
        if flat:
            x = x.unsqueeze(1)
        cond = self.cond_enc(cond)
        for block in self.blocks:
            x = block(x, cond)
        return x.squeeze(1) if flat else x
