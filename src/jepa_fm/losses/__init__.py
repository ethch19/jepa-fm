from jepa_fm.losses.composite import CompositeLoss, LossTerm
from jepa_fm.losses.cosine import CosineDistance
from jepa_fm.losses.latent_mse import LatentMSE
from jepa_fm.losses.sigreg import SIGReg
from jepa_fm.losses.utils import pair_indices

__all__ = [
    "CompositeLoss",
    "CosineDistance",
    "LatentMSE",
    "LossTerm",
    "SIGReg",
    "pair_indices",
]
