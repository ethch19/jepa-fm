import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class LossTerm(nn.Module, ABC):
    """One additive loss term.

    ``forward`` returns the raw statistic; the weight is applied by
    ``CompositeLoss`` so that logged values stay comparable across weights.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = float(weight)

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, V, C) latents -> unweighted scalar."""


class CompositeLoss(nn.Module):
    """Weighted sum of named loss terms.

    ``forward`` returns ``(total, {name: value})`` so the loop can log each term
    separately -- a term that collapses to zero is otherwise invisible.
    """

    def __init__(self, terms: dict[str, LossTerm]):
        super().__init__()
        self.terms = nn.ModuleDict(terms)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        values = {name: term(z) for name, term in self.terms.items()}
        total = sum(
            (self.terms[name].weight * value for name, value in values.items()),
            start=z.new_zeros(()),
        )
        return total, values
