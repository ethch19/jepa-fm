import torch

from jepa_fm.data.utils import PairOrder, PairType


def pair_indices(
    n_lesions: int,
    n_views: int,
    mode: PairType | str,
    order: PairOrder | str = PairOrder.COMBINATION,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Index pairs into the flattened (B * V) latent batch.

    Returns ``(a, b)``: ``a`` is the context view, ``b`` the target.
    ``COMBINATION`` = pairs are returned in the order of their index (view 0, view 1)
    ``PERMUTATION`` = pairs including both (a, b) and (b, a)
    """
    mode, order = PairType(mode), PairOrder(order)

    flat = torch.arange(n_lesions * n_views, device=device)
    lesion, view = flat // n_views, flat % n_views

    same_lesion = lesion[:, None] == lesion[None, :]
    same_view = view[:, None] == view[None, :]

    match mode:
        case PairType.ALL:
            mask = ~(same_lesion & same_view)  # exclude i paired with itself
        case PairType.SAME_LESION:
            mask = same_lesion & ~same_view
        case PairType.DIFF_LESION:
            mask = ~same_lesion
        case _:
            raise ValueError(f"unknown pair mode: {mode!r}")

    if order is PairOrder.COMBINATION:
        mask = mask & (flat[:, None] < flat[None, :])  # keep one direction

    return mask.nonzero(as_tuple=True)
