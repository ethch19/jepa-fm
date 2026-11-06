import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from typing import NamedTuple

from jepa_fm.data.utils import SplitType, load_col_file, centroid_crop

_EXTENSIONS = ["csv", "parquet"]


class Lesion(NamedTuple):
    uuid: str
    npy_img_path: str
    npy_mask_path: str
    centroid: tuple[float, float, float]  # x, y, z


class PretrainDataset(Dataset):
    def __init__(
        self,
        split: SplitType,
        meta_dir: str,
        views: int,
        crop_size: tuple[float, float, float],
    ):
        self.split = split
        self.meta_dir = Path(meta_dir)
        self.views = views
        self.crop_size = crop_size
        if views > 1:
            self.transform = self._transform()
        else:
            self.transform = self._inf_transform()

        data_dfs = []
        for file in self.meta_dir.rglob("*"):
            if file.is_file and any(file.name.endswith(ext) for ext in _EXTENSIONS):
                tabular = load_col_file(
                    file,
                    cols=[
                        "node_id",
                        "scan_id",
                        "cohort",
                        "npy_img_path",
                        "npy_seg_path",
                        "centroid_x",
                        "centroid_y",
                        "centroid_z",
                    ],
                )

                npy_only = (
                    tabular["npy_img_path"].astype(str).str.endswith(".npy", na=False)
                    & tabular["npy_seg_path"].astype(str).str.endswith(".npy", na=False)
                ).all()

                invalid = tabular[~npy_only]
                print(
                    f"Invalid files found in dataframe: {invalid.to_markdown(index=False)}"
                )

                cohort_only = npy_only.loc[npy_only["cohort"] == self.split]

                data_dfs.append(cohort_only)

        samples = pd.concat(data_dfs, ignore_index=True) if data_dfs else pd.DataFrame()

        samples = self.samples.sort_values(by="uuid", ascending=True).reset_index(
            drop=True
        )

        if len(samples) == 0:
            raise RuntimeError(
                f"No files found under {self.img_dir} with {_EXTENSIONS}"
            )

        self.records = [
            Lesion(
                row.node_id,
                row.npy_img_path,
                row.npy_mask_path,
                (row.centroid_x, row.centroid_y, row.centroid_z),
            )
            for row in samples.itertuples(index=False)
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records[idx]
        img = self._load_file(row.npy_img_path)

        patch = centroid_crop(img, (row.centroid[0], row.centroid[1]), self.crop_size)

        return (
            torch.stack([self.transform(img) for _ in range(self.V)]),
            self.records[idx],
        )

    def _load_file(file_path: str) -> np.ndarray:
        img_path = Path(file_path)
        if img_path.is_file():
            if img_path.name.endswith("npy"):
                vol_img = np.load(img_path, mmap_mode="r")
            else:
                raise RuntimeError(f"Non .npy found calling _load_file: {img_path}")
            return vol_img
        else:
            raise RuntimeError(f"File not found at: {img_path}")

    # TODO: replace with grayscale image transformations
    def _transform():
        return v2.Compose(
            [
                v2.RandomResizedCrop(128, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _inf_transform():
        return v2.Compose(
            [
                v2.Resize(128),
                v2.CenterCrop(128),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
