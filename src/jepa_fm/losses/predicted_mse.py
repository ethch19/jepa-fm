import torch

from jepa_fm.data.utils import PairOrder, PairType
from jepa_fm.losses.composite import LossTerm
from jepa_fm.losses.utils import pair_indices


class PredictedMSE(LossTerm):
    """MSE between a predicted latent and the target latent, both projected"""

    def __init__(
        self,
        pairs: PairType | str = PairType.DIFF_LESION,
        order: PairOrder | str = PairOrder.COMBINATION,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        self.pairs = PairType(pairs)
        self.order = PairOrder(order)

    def forward(
        self,
        z: torch.Tensor,
        *,
        latents: torch.Tensor,
        features: torch.Tensor,
        predictor: torch.nn.Module,
        projector: torch.nn.Module,
    ) -> torch.Tensor:
        if predictor is None:
            raise ValueError(
                "predicted_mse needs a predictor: set `predictor` in the config"
            )
        n_lesions, n_views = latents.shape[:2]
        a, b = pair_indices(n_lesions, n_views, self.pairs, self.order, latents.device)
        if a.numel() == 0:
            raise ValueError(
                f"no pairs for {self.pairs} with B={n_lesions}, V={n_views}: "
                "diff_lesion needs 2+ lesions per batch"
            )
        flat = latents.flatten(0, 1).float()  # (lesions * views, encoder width)
        delta = features[b // n_views] - features[a // n_views]  # r_target - r_context
        predicted = predictor(flat[a], delta)
        out = projector(torch.cat([predicted, flat[b]]))
        predicted, target = out.chunk(2)
        return (predicted - target).pow(2).mean()
