import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class LossTerm(nn.Module, ABC):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = float(weight)

    @abstractmethod
    def forward(self, z: torch.Tensor, **ctx) -> torch.Tensor:
        """z: (B, V, C) latents -> unweighted scalar. ``ctx`` carries the
        encoder-space latents, batch features and modules a term may need."""


class CompositeLoss(nn.Module):
    """Weighted sum of named loss terms"""

    def __init__(self, terms: dict[str, LossTerm]):
        super().__init__()
        self.terms = nn.ModuleDict(terms)

    def forward(
        self, z: torch.Tensor, **ctx
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        values = {name: term(z, **ctx) for name, term in self.terms.items()}
        total = sum(
            (self.terms[name].weight * value for name, value in values.items()),
            start=z.new_zeros(()),
        )
        return total, values
