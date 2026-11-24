import torch

from jepa_fm.losses.composite import LossTerm


class SIGReg(LossTerm):
    def __init__(self, knots: int = 17, num_proj: int = 256, weight: float = 1.0):
        super().__init__(weight)
        self.num_proj = num_proj

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z: torch.Tensor, **ctx) -> torch.Tensor:
        # (V, B, C): the sample axis is B, so V becomes a stack of statistics
        proj = z.transpose(0, 1).float()  # fp32
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        a = a.div_(a.norm(p=2, dim=0))

        x_t = (proj @ a).unsqueeze(-1) * self.t  # (V, B, P, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)  # (V, P)
        return statistic.mean()  # average over views and projections
