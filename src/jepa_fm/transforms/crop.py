import numpy as np
import torch
from monai.transforms import CenterSpatialCrop, Resize, Transform


class RandomZoom3D(Transform):
    """Random zoom about the centre, resized back to ``size`` so the output shape is fixed."""
    def __init__(
        self,
        size: tuple[int, int, int],
        scale: tuple[float, float] = (0.08, 1.0),
        p: float = 1.0,
    ) -> None:
        super().__init__()
        if len(size) != 3 or any(s <= 0 for s in size):
            raise ValueError(f"size must be 3 positive ints, got {size}")
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if not 0 < scale[0] <= scale[1] <= 1:
            raise ValueError(f"scale must satisfy 0 < low <= high <= 1, got {scale}")
        self.size = tuple(size)
        self.scale = scale
        self.p = p
        self._resize = Resize(self.size, mode="trilinear")

    def __call__(self, vol: np.ndarray | torch.Tensor) -> torch.Tensor:
        if vol.ndim < 3:
            raise ValueError(f"expected >= 3 dims, got shape {vol.shape}")
        lead = tuple(vol.shape[:-3])

        x = (
            torch.as_tensor(np.ascontiguousarray(vol))
            if isinstance(vol, np.ndarray)
            else vol
        )
        x = x.reshape(-1, *x.shape[-3:]).float()

        if self.p < 1.0 and torch.rand(1).item() >= self.p:
            x = CenterSpatialCrop([min(s, d) for s, d in zip(self.size, x.shape[-3:])])(
                x
            )
        else:
            s = self.scale[0] + torch.rand(1).item() * (self.scale[1] - self.scale[0])
            x = CenterSpatialCrop(
                [max(1, int(round(s * d))) for d in x.shape[-3:]]
            )(x)
        return self._resize(x).reshape(*lead, *self.size)
