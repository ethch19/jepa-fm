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
