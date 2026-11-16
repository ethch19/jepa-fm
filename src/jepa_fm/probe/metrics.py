from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor


class EmbeddingMetric(ABC):
    """Base for metrics computed on an (N, D) matrix of embeddings"""

    def __call__(self, z: Tensor) -> dict[str, float]:
        if z.ndim != 2:
            raise ValueError(f"expected (N, D) embeddings, got shape {tuple(z.shape)}")
        z = z.detach()
        if z.dtype not in (torch.float32, torch.float64):
            z = z.float()
        return self.compute(z)

    @abstractmethod
    def compute(self, z: Tensor) -> dict[str, float]:
        """Return this metric's scalars for a validated (N, D) matrix."""

    @staticmethod
    def singular_values(z: Tensor) -> Tensor:
        """Descending singular values of z"""
        return torch.linalg.svdvals(z)


class RankMe(EmbeddingMetric):
    """Smooth effective rank of the embeddings

    exp of the Shannon entropy (nats) of the L1-normalised singular values, so
    an isotropic spectrum scores min(N, D) and a rank-1 spectrum scores 1.
    """

    def __init__(self, eps: float = 1e-7, clip_dim: int | None = None):
        self.eps = eps
        self.clip_dim = clip_dim

    def compute(self, z: Tensor) -> dict[str, float]:
        sigma = self.singular_values(z)
        n, k = z.shape
        total = sigma.sum()
        if total <= 0:  # total collapse
            return {"rankme": 0.0, "rank": 0}

        p = sigma / total + self.eps
        score = torch.exp(-(p * p.log()).sum())

        if self.clip_dim is not None:
            score = score.clamp(max=self.clip_dim)

        threshold = sigma[0] * max(n, k) * self.eps
        return {"rankme": score.item(), "rank": int((sigma > threshold).sum())}


class PerDimVariance(EmbeddingMetric):
    """Spread of the embeddings per dimension, in whatever basis the encoder produced.

    `top_share` is the variance carried by the `top_k` widest dimensions as a
    fraction of the total: it stays near 1 when a few directions carry the signal
    no matter what the rank reports. `dead_frac` counts dimensions whose standard
    deviation never rises above `dead_std`, i.e. wasted channel capacity.
    """

    def __init__(self, top_k: int = 64, dead_std: float = 1e-3):
        self.top_k = top_k
        self.dead_std = dead_std

    def compute(self, z: Tensor) -> dict[str, float]:
        var = z.var(dim=0, unbiased=False)
        total = var.sum()
        if total <= 0:  # total collapse
            return {"mean_std": 0.0, "min_std": 0.0, "dead_frac": 1.0, "top_share": 0.0}

        std = var.clamp_min(0).sqrt()
        top = var.topk(min(self.top_k, var.numel())).values.sum()
        return {
            "mean_std": std.mean().item(),
            "min_std": std.min().item(),
            "dead_frac": (std < self.dead_std).float().mean().item(),
            "top_share": (top / total).item(),
        }


class Uniformity(EmbeddingMetric):
    """How evenly the latents spread over the unit sphere.

    ``log mean exp(-2 * ||x_i - x_j||^2)`` over sampled pairs, after Wang & Isola
    (2020); lower is more uniform. Pairs come from a fixed seed so two checkpoints
    are compared on the same pairs rather than on sampling noise.
    """

    def __init__(self, num_pairs: int = 8192, seed: int = 0):
        self.num_pairs = num_pairs
        self.seed = seed

    def compute(self, z: Tensor) -> dict[str, float]:
        n = z.shape[0]
        if n < 2:
            return {"uniformity": 0.0, "mean_cos": 0.0}

        generator = torch.Generator(device=z.device).manual_seed(self.seed)
        i = torch.randint(n, (self.num_pairs,), generator=generator, device=z.device)
        j = torch.randint(n - 1, (self.num_pairs,), generator=generator, device=z.device)
        j = j + (j >= i)  # skip self-pairs without rejection sampling

        x = torch.nn.functional.normalize(z, dim=1)
        cos = (x[i] * x[j]).sum(dim=1)
        sq_dist = 2 - 2 * cos  # unit vectors: ||a - b||^2 = 2 - 2<a, b>
        return {
            "uniformity": torch.log(torch.exp(-2 * sq_dist).mean()).item(),
            "mean_cos": cos.mean().item(),
        }


def select_by_rankme(scores: Sequence[float]) -> int:
    if not scores:
        raise ValueError("no scores to select from")

    best = 0
    for i in range(1, len(scores)):
        score = scores[i]
        if score > scores[best]:
            best = i
        elif score == scores[best] and (
            score > scores[i - 1] or (i < len(scores) - 1 and score > scores[i + 1])
        ):
            best = i
    return best
