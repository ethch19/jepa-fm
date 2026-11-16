import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from typing import NamedTuple
import monai.transforms as mai_tf

from jepa_fm.data.utils import (
    VoxelFormat,
    SplitType,
    load_col_file,
    centroid_crop,
    load_file,
)
from jepa_fm.transforms import RandomZoom3D

_EXTENSIONS = [".csv", ".parquet"]


class Lesion(NamedTuple):
    uuid: str
    npy_img_path: str
    npy_mask_path: str
    centroid: tuple[float, float, float]  # stored dependining on voxel_format
    dataset: str  # stem of the metadata file this row came from


class PretrainDataset(Dataset):
    @staticmethod
    def transform(crop_size: tuple[int, int, int]):
        return v2.Compose(
            [
                RandomZoom3D(size=crop_size, scale=(0.6, 1)),
                mai_tf.RandAxisFlip(prob=0.5),
                mai_tf.RandHistogramShift(prob=0.5),
                mai_tf.RandGaussianSmooth(prob=0.5),
                mai_tf.ThresholdIntensity(threshold=-1024, cval=-1024),
                mai_tf.NormalizeIntensity(subtrahend=-1024, divisor=3072),
            ]
        )

    @staticmethod
    def inf_transform():
        return v2.Compose(
            [
                mai_tf.ThresholdIntensity(threshold=-1024, cval=-1024),
                mai_tf.NormalizeIntensity(subtrahend=-1024, divisor=3072),
            ]
        )

    def __init__(
        self,
        split: SplitType,
        meta_dir: str,
        crop_size: tuple[int, int, int],
        views: int = 1,
        inference: bool = False,
        voxel_format: VoxelFormat = VoxelFormat.REVERSED,
    ):
        self.split = split
        self.meta_dir = Path(meta_dir)
        self.views = views
        self.crop_size = tuple(int(round(s)) for s in crop_size)
        self.voxel_format = (
            VoxelFormat[voxel_format.upper()]
            if isinstance(voxel_format, str)
            else voxel_format
        )
        if not inference:
            self.transform = PretrainDataset.transform(self.crop_size)
        else:
            self.transform = PretrainDataset.inf_transform()

        data_dfs = []
        for file in self.meta_dir.rglob("*"):
            if file.is_file() and file.suffix in _EXTENSIONS:
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

                npy_only = tabular["npy_img_path"].str.endswith(
                    ".npy", na=False
                ) & tabular["npy_seg_path"].str.endswith(".npy", na=False)

                if not npy_only.all():
                    print(f"{file}: dropped {(~npy_only).sum()} rows")

                data_dfs.append(
                    tabular.loc[npy_only & (tabular["cohort"] == self.split)].assign(
                        dataset=file.stem
                    )
                )

        samples = pd.concat(data_dfs, ignore_index=True) if data_dfs else pd.DataFrame()

        if samples.empty:
            raise RuntimeError(
                f"No files found under {self.meta_dir} with {_EXTENSIONS}"
            )

        samples = samples.sort_values(by="node_id", ascending=True).reset_index(
            drop=True
        )

        match self.voxel_format:
            case VoxelFormat.STANDARD:
                self.records = [
                    Lesion(
                        row.node_id,
                        row.npy_img_path,
                        row.npy_seg_path,
                        (row.centroid_x, row.centroid_y, row.centroid_z),
                        row.dataset,
                    )
                    for row in samples.itertuples(index=False)
                ]
            case VoxelFormat.REVERSED:
                self.records = [
                    Lesion(
                        row.node_id,
                        row.npy_img_path,
                        row.npy_seg_path,
                        (row.centroid_z, row.centroid_y, row.centroid_x),
                        row.dataset,
                    )
                    for row in samples.itertuples(index=False)
                ]
            case _:
                raise RuntimeError(
                    "voxel_format not specified when initialising dataset"
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        """Returns (V, D, H, W) where views are extracted from the same centroid crop, just different transformations

        plus the lesion and dataset this row belongs to, as plain strings so the
        default collate hands them back as a list rather than mangling a pandas row.
        """
        row = self.records[idx]
        img = load_file(row.npy_img_path)

        patch = centroid_crop(img, row.centroid, self.crop_size)

        return (
            torch.stack([self.transform(patch) for _ in range(self.views)]),
            row.uuid,
            row.dataset,
        )
