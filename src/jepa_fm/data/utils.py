import numpy as np
from enum import StrEnum, auto
import pandas as pd
from pathlib import Path


class SplitType(StrEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


def load_col_file(file_path: Path, cols=None) -> pd.DataFrame:
    match file_path.suffix:
        case "csv":
            tabular = pd.read_csv(file_path)
        case "parquet":
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
    vol: np.ndarray, centre: tuple[float, float], crop_size: tuple[float, float], fill=0
):
    start = [
        int(round(centre_d)) - size_d // 2
        for centre_d, size_d in zip(centre, crop_size)
    ]
    padding = [
        (max(0, -start_d), max(0, start_d + size_d - acc_size))
        for start_d, size_d, acc_size in zip(start, crop_size, vol.shape)
    ]

    if any(x or y for x, y in padding):
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
