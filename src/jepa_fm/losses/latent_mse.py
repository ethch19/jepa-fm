import torch

from jepa_fm.data.utils import PairOrder, PairType
from jepa_fm.losses.composite import LossTerm
from jepa_fm.losses.utils import pair_indices


class LatentMSE(LossTerm):
    def __init__(
        self,
        pairs: PairType | str = PairType.SAME_LESION,
        order: PairOrder | str = PairOrder.COMBINATION,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        self.pairs = PairType(pairs)
        self.order = PairOrder(order)

    def forward(self, z: torch.Tensor, **ctx) -> torch.Tensor:
        n_lesions, n_views, _ = z.shape
        a, b = pair_indices(n_lesions, n_views, self.pairs, self.order, z.device)
        z = z.flatten(0, 1)  # (B * V, C) = (lesions * views, channels)
        if a.numel() == 0:
            raise ValueError(
                f"no pairs for pairs={self.pairs} with B={n_lesions}, V={n_views}: "
                "diff_lesion needs 2+ lesions per batch, same_lesion and all need 2+ views"
            )
        return (z[a] - z[b]).pow(2).mean()
