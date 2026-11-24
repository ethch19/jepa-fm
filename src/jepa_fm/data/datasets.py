import json
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from typing import NamedTuple
import monai.transforms as mai_tf
import pyarrow.parquet as pq

from jepa_fm.data.utils import (
    VoxelFormat,
    SplitType,
    load_col_file,
    centroid_crop,
    load_file,
)
from jepa_fm.transforms import RandomZoom3D

PARQUET_META_KEY = b"jepa_radiomics"

_EXTENSIONS = [".csv", ".parquet"]


class Lesion(NamedTuple):
    uuid: str
    npy_img_path: str
    npy_mask_path: str
    centroid: tuple[float, float, float]
    dataset: str  # stem of the metadata file


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
        radiomics_path: str | Path | None = None,
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

        duplicated = samples["node_id"].duplicated()
        if duplicated.any():
            raise RuntimeError(
                f"{int(duplicated.sum())} duplicate node_id(s) under {self.meta_dir}, "
                f"first {samples.loc[duplicated, 'node_id'].iloc[0]!r}"
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

        self.features, self.row_of = self._load_radiomics(radiomics_path)

    def __len__(self):
        return len(self.records)

    def _load_radiomics(self, path: str | Path | None):
        if path is None:
            return None, {}
        cache_path = Path(path).with_suffix(".parquet")
        if not cache_path.is_file():
            raise RuntimeError(f"radiomics cache missing: {cache_path}")
        table = pq.read_table(cache_path)
        raw_meta = table.schema.metadata or {}
        if PARQUET_META_KEY not in raw_meta:
            raise RuntimeError(
                f"radiomics cache {cache_path.name}: no {PARQUET_META_KEY!r} metadata"
            )
        cache_meta = json.loads(raw_meta[PARQUET_META_KEY])
        for key, value in (
            ("crop_size", list(self.crop_size)),
            ("voxel_format", str(self.voxel_format)),
        ):
            if cache_meta.get(key) != value:
                raise RuntimeError(
                    f"radiomics cache {cache_path.name}: {key} "
                    f"{cache_meta.get(key)!r} != dataset {value!r}"
                )

        frame = table.to_pandas()
        row_of = {str(uuid): i for i, uuid in enumerate(frame["uuid"])}
        missing = [row.uuid for row in self.records if str(row.uuid) not in row_of]
        if missing:
            raise RuntimeError(
                f"radiomics cache {cache_path.name}: {len(missing)} of "
                f"{len(self.records)} records missing, first {missing[0]!r}"
            )
        return (
            frame.drop(columns="uuid").to_numpy(dtype=np.float32, copy=True),
            row_of,
        )

    def __getitem__(self, idx: int):
        """Returns (V, D, H, W) of the same tumour crop"""
        row = self.records[idx]
        img = load_file(row.npy_img_path)

        patch = centroid_crop(img, row.centroid, self.crop_size)

        views = [self.transform(patch) for _ in range(self.views)]

        if self.features is None:
            features = torch.zeros(0)
        else:
            features = torch.from_numpy(self.features[self.row_of[row.uuid]])

        return torch.stack(views), row.uuid, row.dataset, features
