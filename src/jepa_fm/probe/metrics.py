"""Label-free embedding-space metrics: an (N, D) matrix in, scalars out.

No model, no I/O, no labels. Extraction lives in the caller, so the same
functions serve both the online (during training) and offline (from a
checkpoint) paths.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor


class EmbeddingMetric(ABC):
    """Base for metrics computed on an (N, D) matrix of embeddings.

    Subclasses implement `compute`; the base handles input hygiene so every
    metric sees a detached, 2-D, floating-point matrix. float32/float64 are
    passed through untouched -- pass float64 yourself when you want the
    reduction carried out at higher precision.
    """

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
        """Descending singular values of z -- only these, never U or V."""
        return torch.linalg.svdvals(z)


class RankMe(EmbeddingMetric):
    """Smooth effective rank of the embedding spectrum (Garrido et al., ICML 2023).

    exp of the Shannon entropy (nats) of the L1-normalised singular values, so
    an isotropic spectrum scores min(N, D) and a rank-1 spectrum scores 1.
    Comparing runs is valid within one SSL method only -- the absolute number
    means nothing across loss families.
    """

    def __init__(self, eps: float = 1e-7, clip_dim: int | None = None):
        self.eps = eps
        self.clip_dim = clip_dim

    def compute(self, z: Tensor) -> dict[str, float]:
        sigma = self.singular_values(z)
        n, k = z.shape
        total = sigma.sum()
        if total <= 0:  # total collapse; a nan would just hide it
            return {"rankme": 0.0, "rank": 0}

        # epsilon after the division, so p_k stays strictly positive on zeros
        p = sigma / total + self.eps
        score = torch.exp(-(p * p.log()).sum())

        if self.clip_dim is not None:
            score = score.clamp(max=self.clip_dim)

        threshold = sigma[0] * max(n, k) * self.eps
        return {"rankme": score.item(), "rank": int((sigma > threshold).sum())}


def select_by_rankme(scores: Sequence[float]) -> int:
    """Index of the best candidate when the scores are ordered by hyperparameter.

    RankMe's hyperparameter-selection procedure (Algorithm 1): take the highest
    score, and on a plateau of tied scores take the trailing edge -- except when
    the plateau runs to the last candidate, which has no trailing edge to find.
    """
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
