"""Downstream evaluation cohorts: one labelled metadata file in, features out.

Kept apart from `PretrainDataset` because these rows are a different animal: no
cohort split, an optional mask column and, above all, labels that must stay
row-aligned with the features built from the same rows.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from jepa_fm.data.datasets import Lesion, PretrainDataset
from jepa_fm.data.utils import VoxelFormat, centroid_crop, load_col_file, load_file

REQUIRED_COLUMNS = ("node_id", "npy_img_path", "centroid_x", "centroid_y", "centroid_z")


class EvaluationDataset(Dataset):
    """Rows of one metadata file, cropped around their centroids at inference time.

    `labels` keeps every original column with row i aligned to `records[i]`, so the
    caller never re-reads and re-sorts the file: a sort that silently disagreed with
    the feature order would mislabel every downstream metric. The crop is the
    deterministic `PretrainDataset.inf_transform`, never the training transform.
    """

    def __init__(
        self,
        meta_file: str | Path,
        crop_size: Sequence[int],
        voxel_format: VoxelFormat | str = VoxelFormat.REVERSED,
    ) -> None:
        self.meta_file = Path(meta_file)
        self.name = self.meta_file.stem
        self.crop_size = tuple(int(round(s)) for s in crop_size)
        self.voxel_format = (
            VoxelFormat[voxel_format.upper()]
            if isinstance(voxel_format, str)
            else voxel_format
        )
        self.transform = PretrainDataset.inf_transform()

        # load_col_file refuses parquet without an explicit column list
        columns = (
            pq.ParquetFile(self.meta_file).schema.names
            if self.meta_file.suffix == ".parquet"
            else None
        )
        frame = load_col_file(self.meta_file, cols=columns)

        missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(f"{self.name}: missing required column(s) {missing}")

        npy_only = frame["npy_img_path"].str.endswith(".npy", na=False)
        if not npy_only.all():
            print(f"{self.meta_file.name}: dropped {(~npy_only).sum()} rows")
        frame = frame.loc[npy_only]
        if frame.empty:
            raise ValueError(f"{self.meta_file.name}: no rows with a .npy image")

        duplicated = frame["node_id"].duplicated()
        if duplicated.any():
            example = frame.loc[duplicated, "node_id"].iloc[0]
            raise ValueError(
                f"{self.name}: {int(duplicated.sum())} duplicate node_id values, "
                f"e.g. {example!r}"
            )

        # sorted like PretrainDataset, so features do not depend on file order
        frame = frame.sort_values("node_id", ascending=True).reset_index(drop=True)
        self.labels = frame

        coords = ("centroid_x", "centroid_y", "centroid_z")
        if self.voxel_format is VoxelFormat.REVERSED:
            coords = tuple(reversed(coords))
        masks = (
            frame["npy_seg_path"] if "npy_seg_path" in frame.columns else [""] * len(frame)
        )
        self.records = [
            Lesion(uuid, img, mask, centroid, self.name)
            for uuid, img, mask, centroid in zip(
                frame["node_id"],
                frame["npy_img_path"],
                masks,
                frame[list(coords)].itertuples(index=False, name=None),
            )
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        row = self.records[idx]
        patch = centroid_crop(load_file(row.npy_img_path), row.centroid, self.crop_size)
        # one view: evaluation must not see a randomly transformed image
        return torch.stack([self.transform(patch)]), row.uuid


@dataclass(frozen=True)
class Endpoint:
    """What a cohort predicts, as the config author declared it.

    `binary` labels must already be 0/1; `survival` labels are only checked to be
    numeric, and what the column holds is interpreted by stage 2.
    """

    kind: str  # "binary" | "survival"
    label_column: str


ENDPOINT_KINDS = {"binary", "survival"}


def parse_endpoint(raw: DictConfig | dict) -> Endpoint:
    values = dict(raw)
    unknown = set(values) - {"kind", "label_column"}
    if unknown:
        raise ValueError(
            f"endpoint has unknown key(s) {sorted(unknown)}; use 'kind' and 'label_column'"
        )
    missing = {"kind", "label_column"} - set(values)
    if missing:
        raise ValueError(f"endpoint is missing key(s) {sorted(missing)}")
    kind = values["kind"]
    if kind not in ENDPOINT_KINDS:
        raise ValueError(
            f"unknown endpoint kind {kind!r}; use one of {sorted(ENDPOINT_KINDS)}"
        )
    column = values["label_column"]
    if not isinstance(column, str) or not column:
        raise ValueError(
            f"endpoint label_column must be a column name, got {column!r}"
        )
    return Endpoint(kind, column)


def validate_endpoint(
    labels: pd.DataFrame, endpoint: Endpoint, dataset: str
) -> dict[str, float]:
    """Check the declared label column and summarise it for the MLflow run."""
    column = endpoint.label_column
    if column not in labels.columns:
        raise ValueError(f"{dataset}: endpoint column {column!r} is not in the metadata")
    values = labels[column]
    if values.isna().any():
        raise ValueError(
            f"{dataset}: {column} has {int(values.isna().sum())} null values"
        )

    if endpoint.kind == "binary":
        unique = sorted(values.unique().tolist(), key=str)
        # bool passes is_numeric_dtype and compares equal to 0/1, so exclude it by name
        if (
            pd.api.types.is_bool_dtype(values)
            or not pd.api.types.is_numeric_dtype(values)
            or set(unique) - {0, 1}
        ):
            raise ValueError(
                f"{dataset}: {column} must be 0/1, found {len(unique)} "
                f"distinct value(s) {unique}"
            )
        return {"prevalence": float(values.mean())}

    if not pd.api.types.is_numeric_dtype(values):
        raise ValueError(f"{dataset}: {column} must be numeric")
    return {
        "mean_label": float(values.mean()),
        "min_label": float(values.min()),
        "max_label": float(values.max()),
    }
