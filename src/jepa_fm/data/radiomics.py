import argparse
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from jepa_fm.data.datasets import PARQUET_META_KEY, Lesion, PretrainDataset
from jepa_fm.data.utils import SplitType, VoxelFormat, centroid_crop, load_file

DEFAULT_PARAMS = Path(__file__).parents[3] / "config" / "radiomics.yaml"
ZERO_VARIANCE_EPS = 1e-8


def params_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RadiomicsExtractor:
    def __init__(
        self,
        params_path: str | Path | None = None,
        bin_width: float | None = None,
        label: int = 1,
    ):
        from radiomics import featureextractor, setVerbosity
        import SimpleITK

        setVerbosity(logging.ERROR)
        self.params_path = Path(params_path or DEFAULT_PARAMS)
        self.bin_width = bin_width
        self.label = label
        self.feature_names: list[str] | None = None
        self._sitk = SimpleITK
        self.extractor = featureextractor.RadiomicsFeatureExtractor(
            str(self.params_path),
            **({} if bin_width is None else {"binWidth": bin_width}),
        )

    def __call__(
        self, volume: np.ndarray, mask: np.ndarray, label: int | None = None
    ) -> np.ndarray:
        """((D, H, W) image, (D, H, W) binary mask) -> (F,) float32, sorted-name order."""
        array = np.asarray(mask)
        if array.max(initial=0) > 1:
            raise ValueError(
                f"mask must be binary (0/1), found values up to {array.max()}"
            )
        result = self.extractor.execute(
            self._sitk.GetImageFromArray(np.asarray(volume, dtype=np.float32)),
            self._sitk.GetImageFromArray(array.astype(np.uint8)),
            label=self.label if label is None else label,
        )
        names = [k for k in sorted(result) if not k.startswith("diagnostics")]
        if self.feature_names is None:
            self.feature_names = names
        values = [np.real(result[name]) for name in names]
        return np.array(values, dtype=np.float32)


def get_optimal_jobs() -> int:
    """70% of the cores; workers run one BLAS thread each (see _radiomics_cli)."""
    return max(1, int((os.cpu_count() or 1) * 0.70))


_EXTRACTOR: RadiomicsExtractor | None = None


def _extractor() -> RadiomicsExtractor:
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = RadiomicsExtractor()
    return _EXTRACTOR


def _extract(
    task: tuple[Lesion, tuple[int, int, int]],
) -> tuple[str, list[str], np.ndarray]:
    record, crop_size = task
    image = centroid_crop(load_file(record.npy_img_path), record.centroid, crop_size)
    mask = centroid_crop(load_file(record.npy_mask_path), record.centroid, crop_size)
    extractor = _extractor()
    try:
        values = extractor(image, mask)
    except Exception as err:
        raise RuntimeError(f"{record.uuid}: radiomics failed") from err
    assert extractor.feature_names is not None
    return record.uuid, extractor.feature_names, values


def _drop_all_nan(
    features: np.ndarray, names: list[str]
) -> tuple[np.ndarray, list[str]]:
    keep = np.isfinite(features).any(axis=0)
    if keep.all():
        return features, names
    dropped = [name for name, kept in zip(names, keep, strict=True) if not kept]
    shown = ", ".join(dropped[:5]) + (" ..." if len(dropped) > 5 else "")
    print(f"dropping {len(dropped)} all-NaN features: {shown}")
    return features[:, keep], [name for name, kept in zip(names, keep) if kept]


def _colinearity(corr: np.ndarray, cutoff: float) -> np.ndarray:
    keep = np.ones(corr.shape[0], dtype=bool)
    for j in range(corr.shape[0]):
        if keep[j] and (np.abs(corr[j, :j][keep[:j]]) >= cutoff).any():
            keep[j] = False
    return keep


def select_features(
    features: np.ndarray, names: list[str], cutoff: float = 0.99
) -> tuple[np.ndarray, list[str], list[dict]]:
    """((N, F) raw, names) -> (z-scored (N, K), names, per-stage counts)."""
    x = np.asarray(features, dtype=np.float64)
    stages: list[dict] = []

    def drop(keep: np.ndarray, stage: str) -> None:
        nonlocal x, names
        kept = [name for name, flag in zip(names, keep, strict=True) if flag]
        stages.append(
            {"stage": stage, "removed": x.shape[1] - len(kept), "remaining": len(kept)}
        )
        x, names = x[:, keep], kept

    drop(np.isfinite(x).all(axis=0), "non_finite")
    drop(x.std(axis=0) > ZERO_VARIANCE_EPS, "zero_variance")
    if not names:
        raise ValueError("selection removed every feature")

    x = (x - x.mean(axis=0)) / x.std(axis=0)

    if cutoff > 0 and len(names) > 1:
        drop(
            _colinearity(np.corrcoef(x, rowvar=False), cutoff),
            f"collinear_{cutoff}",
        )
    return x.astype(np.float32), names, stages


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta-path", required=True, help="directory of csv/parquet metadata"
    )
    parser.add_argument("--crop-size", required=True, type=int, nargs=3)
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="parquet path, may live inside --meta-path; the loader skips "
        "files carrying the radiomics metadata",
    )
    parser.add_argument(
        "--voxel-format",
        default=VoxelFormat.REVERSED.value,
        choices=[v.value for v in VoxelFormat],
    )
    parser.add_argument(
        "--workers",
        type=int,
        help=f"multi-processes. default {get_optimal_jobs()}",
    )
    parser.add_argument(
        "--collinearity",
        type=float,
        default=0.99,
        help="drop features with |r| >= cutoff, 0 disables",
    )
    parser.add_argument("--limit", type=int, help="first N records only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    workers = args.workers if args.workers is not None else get_optimal_jobs()
    if workers < 1:
        raise SystemExit(f"--workers must be 1 or more, got {workers}")
    out = args.out.with_suffix(".parquet")

    dataset = PretrainDataset(
        SplitType.TRAIN,
        args.meta_path,
        tuple(args.crop_size),
        views=1,
        inference=True,
        voxel_format=args.voxel_format,
    )
    records = dataset.records[: args.limit]
    tasks = [(record, dataset.crop_size) for record in records]

    started = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(
                tqdm(pool.map(_extract, tasks), total=len(tasks), unit="lesion")
            )
    else:
        results = [_extract(task) for task in tqdm(tasks, unit="lesion")]
    elapsed = time.perf_counter() - started

    names = results[0][1]
    for uuid, row_names, _ in results:
        if row_names != names:
            raise RuntimeError(f"{uuid}: feature names differ from the first row")
    features = np.stack([values for _, _, values in results]).astype(np.float32)
    uuids = np.asarray([uuid for uuid, _, _ in results])
    n_extracted = int(features.shape[1])

    features, names = _drop_all_nan(features, names)
    finite = np.isfinite(features)
    n_raw = int(features.shape[1])
    features, names, selection = select_features(features, names, args.collinearity)
    if n_raw != n_extracted:
        selection.insert(
            0,
            {
                "stage": "all_nan",
                "removed": n_extracted - n_raw,
                "remaining": n_raw,
            },
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {"uuid": uuids, **{name: features[:, i] for i, name in enumerate(names)}}
    )
    from radiomics import __version__ as radiomics_version

    meta = {
        "params_path": str(DEFAULT_PARAMS),
        "params_sha256": params_hash(DEFAULT_PARAMS),
        "split": SplitType.TRAIN.value,
        "crop_size": list(dataset.crop_size),
        "voxel_format": str(dataset.voxel_format),
        "label": 1,
        "n_records": len(records),
        "n_features_raw": n_raw,
        "n_features": int(features.shape[1]),
        "n_nan": int((~finite).sum()),
        "n_rows_with_nan": int((~finite).any(axis=1).sum()),
        "library": {"pyradiomics": radiomics_version},
        "selection": {
            "collinearity": args.collinearity,
            "n_in": n_extracted,
            "n_out": int(features.shape[1]),
            "stages": selection,
        },
    }
    table = table.replace_schema_metadata({PARQUET_META_KEY: json.dumps(meta).encode()})
    pq.write_table(table, out, compression="zstd")

    print(
        f"{len(records)} lesions x {features.shape[1]} features in {elapsed:.1f}s "
        f"({elapsed / max(1, len(records)):.2f}s/lesion, {workers} workers)"
    )
    print(
        f"features {n_extracted} -> {features.shape[1]}: "
        + ", ".join(f"{s['stage']} -{s['removed']}" for s in selection)
    )
    if meta["n_nan"]:
        print(
            f"WARNING: {meta['n_nan']} non-finite values in {meta['n_rows_with_nan']} "
            f"raw rows, their columns were dropped"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
