"""Model in, (N, D) embedding matrices out, plus the row labels the metrics need.

Shared by the online path (a probe during training) and the offline path (from a
checkpoint), so the numbers stay comparable between the two.
"""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from jepa_fm.models.utils import forward_latents


@torch.no_grad()
def extract_embeddings(
    net: torch.nn.Module,
    projector: torch.nn.Module,
    loader: DataLoader,
    spaces: Sequence[str],
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> tuple[dict[str, Tensor], list[str]]:
    """One pass over `loader`: a float32 CPU (N, D) matrix per space, plus row labels.

    The loader must yield single views (`views=1`): the same lesion twice would
    silently weigh its directions twice in the spectrum.
    """
    net.eval()
    projector.eval()
    latents: list[Tensor] = []
    labels: list[str] = []
    for imgs, _, dataset in loader:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            latents.append(
                forward_latents(net, imgs.to(device, non_blocking=True))[:, 0]
            )
        labels.extend(dataset)

    z = torch.cat(latents).float().cpu()
    matrices = {"enc": z}
    if "proj" in spaces:
        matrices["proj"] = projector(z.to(device)).float().cpu()

    unknown = set(spaces) - set(matrices)
    if unknown:
        raise ValueError(
            f"unknown probe space(s) {sorted(unknown)}; use 'enc' and/or 'proj'"
        )
    return {space: matrices[space] for space in spaces}, labels


def group_rows(labels: Sequence[str], group_by: str | None) -> dict[str, list[int]]:
    """Row indices per group: one 'all' group, or one group per metadata file."""
    if group_by in (None, "all"):
        return {"all": list(range(len(labels)))}
    if group_by != "dataset":
        raise ValueError(f"group_by must be null, 'all' or 'dataset', got {group_by!r}")

    groups: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)
    return groups
