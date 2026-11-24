import torch
import torch.nn.functional as F

from jepa_fm.data.utils import PairOrder, PairType
from jepa_fm.losses.composite import LossTerm
from jepa_fm.losses.utils import pair_indices


class CosineDistance(LossTerm):
    """1 - cos(z_a, z_b), averaged over pairs. Same pairing knobs as LatentMSE.

    MSE and cosine disagree once latent norms grow: MSE chases magnitude, cosine
    only direction. Pair it with ``pairs: diff_lesion`` to spread different lesions
    apart instead of pulling the same lesion's views together.
    """

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
        if a.numel() == 0:
            raise ValueError(f"no pairs for {self.pairs} with B={n_lesions}, V={n_views}")
        z = z.flatten(0, 1)  # (B * V, C) = (lesions * views, channels)
        return (1.0 - F.cosine_similarity(z[a], z[b], dim=-1)).mean()
