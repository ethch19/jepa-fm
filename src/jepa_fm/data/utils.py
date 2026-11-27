import numpy as np
from enum import StrEnum, auto
import pandas as pd
from pathlib import Path


class SplitType(StrEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


class VoxelFormat(StrEnum):
    STANDARD = auto()
    REVERSED = auto()


class PairType(StrEnum):
    ALL = auto()
    SAME_LESION = auto()
    DIFF_LESION = auto()


class PairOrder(StrEnum):
    """Whether each pair is produced once or in both directions.

    Only asymmetric terms can tell the difference; a squared distance gives the
    same value either way. COMBINATION keeps the roles meaningful: pairs are
    (a, b) with a < b, so with views: 2 the first view is always the context and
    the second always the target.
    """

    COMBINATION = auto()  # one pair per pair of images
    PERMUTATION = auto()  # (a, b) and (b, a)


def load_col_file(file_path: Path, cols=None) -> pd.DataFrame:
    match file_path.suffix:
        case ".csv":
            tabular = pd.read_csv(file_path)
        case ".parquet":
            if cols:
                tabular = pd.read_parquet(file_path, columns=cols)
            else:
                raise RuntimeError(
                    f"No column provided for parquet file: {file_path.name}"
                )
        case _:
            raise RuntimeError(f"Did not match any suffix for {file_path.name}")
    return tabular


def centroid_crop(
    vol: np.ndarray,
    centre: tuple[float, ...],
    crop_size: tuple[int, ...],
    fill: float = 0,
) -> np.ndarray:
    if len(centre) != vol.ndim or len(crop_size) != vol.ndim:
        raise ValueError(
            f"centre {centre} and crop_size {crop_size} need {vol.ndim} entries for shape {vol.shape}"
        )
    start = [
        int(round(centre_d)) - size_d // 2
        for centre_d, size_d in zip(centre, crop_size)
    ]
    padding = [
        (max(0, -start_d), max(0, start_d + size_d - acc_size))
        for start_d, size_d, acc_size in zip(start, crop_size, vol.shape)
    ]
    if any(any(p) for p in padding):
        vol = np.pad(vol, padding, mode="constant", constant_values=fill)
        start = [
            start_d + padding_start
            for start_d, (padding_start, _) in zip(start, padding)
        ]
    return vol[
        tuple(
            slice(start_d, start_d + size_d)
            for start_d, size_d in zip(start, crop_size)
        )
    ]


def load_file(file_path: str) -> np.ndarray:
    img_path = Path(file_path)
    if img_path.is_file():
        if img_path.name.endswith("npy"):
            vol_img = np.load(img_path, mmap_mode="r")
        else:
            raise RuntimeError(f"Non .npy found calling _load_file: {img_path}")
        return vol_img
    else:
        raise RuntimeError(f"File not found at: {img_path}")
