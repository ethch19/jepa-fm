"""DeepLesion lesion masks from LCUNet+ predictions, cross-checked against boxes.

``deeplesion_lung.csv`` is the ground truth: one row per in-lung lesion with a
per-slice bbox ``[x1, y1, x2, y2]`` plus the slice it was drawn on. LCUNet+ is a
lung *tumour* segmenter, so it fires on some of these lesions and on unrelated
structures in others, and misses a lot of them (measured: raw probability inside
the box stays below 0.02 for ~70% of lesions). A predicted connected component
therefore survives only if its centroid falls inside a lesion box grown by
``--margin-mm``; everything else in that volume is dropped, and a volume with no
surviving component gets no mask at all. Components are kept whole, so a
segmentation running past the annotation is not cut off mid-lesion. Those
volumes are counted in the manifest, so the coverage loss stays visible.

Box coordinates index the scan's own voxel grid: bbox x spans array axis 1 and
bbox y spans axis 2 (checked against bone contrast and the lesion densities), so
masks stay on the native grid - no resampling, no DICOM windowing.

    uv run --no-sync jepa-preprocess-deeplesion --limit 5   # smoke test
    uv run --no-sync jepa-preprocess-deeplesion               # full run, resumable
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from jepa_fm.preprocessing.utils import (
    IMAGES_DIR,
    MASK_SUFFIX,
    SCAN_SUFFIX,
    SEGMENTATIONS_DIR,
    Job,
    geometry_mismatch,
    resample_mask_to_scan,
    run,
)

DEFAULT_ROOT = Path("/mnt/ethch/data/DeepLesion")
CSV_NAME = "dl_lung_verified.csv"
MANIFEST_NAME = "deeplesion_masks.csv"
COMPONENTS_NAME = "deeplesion_components.parquet"
LCUNET_REPO = Path.home() / "LCUNetPlus"
PRED_PREFIX = "seg_"
MARGIN_MM = 8.0  # glioblastoma-sized slack around the DeepLesion box; 4 mm dropped 90% of blobs
MIN_VOXELS = 20  # 1-voxel specks survive the box filter and give radiomics duplicate rows

MASKED = "masked"
REJECTED = "rejected"
MISSED = "no_prediction"
UNREADABLE = "unreadable"


def usable(scan_path: Path) -> bool:
    """Scans smaller than a NIfTI header are zero-byte truncations; they kill inference."""
    return scan_path.exists() and scan_path.stat().st_size > 1024


def load_lesions(csv_path: Path, images: Path, limit: int) -> pd.DataFrame:
    """In-lung lesion rows whose scan exists on disk, one row per lesion."""
    table = pd.read_csv(csv_path)
    table[["x1", "y1", "x2", "y2"]] = (
        table.bbox.str.strip("[]").str.split(",", expand=True).astype(int)
    )
    table["z"] = table.centroid_z.astype(int)
    table["volume"] = table.Volume_fn.str.removesuffix(SCAN_SUFFIX)
    present = table.volume.map(lambda volume: usable(images / f"{volume}{SCAN_SUFFIX}"))
    table = table[present]
    volumes = sorted(table.volume.unique())[: limit or None]
    return table[table.volume.isin(volumes)].sort_values(["volume", "z"], ignore_index=True)


def lesion_window(
    lesion: pd.Series | object, spacing: tuple[float, ...], margin_mm: float, shape: tuple[int, ...]
) -> tuple[slice, slice, slice]:
    """Grown bbox in array order (z, x, y), clamped to the array."""
    pad_y = max(1, round(margin_mm / spacing[0]))
    pad_x = max(1, round(margin_mm / spacing[1]))
    pad_z = max(1, round(margin_mm / spacing[2]))
    limits = (
        (lesion.z - pad_z, lesion.z + pad_z + 1),
        (lesion.x1 - pad_x, lesion.x2 + pad_x),
        (lesion.y1 - pad_y, lesion.y2 + pad_y),
    )
    return tuple(slice(max(0, lo), min(size, hi)) for (lo, hi), size in zip(limits, shape))


def keep_near_boxes(prediction: sitk.Image, lesions: pd.DataFrame, margin_mm: float) -> np.ndarray:
    """Voxels of every predicted component whose centroid falls in a lesion window.

    Centroid rather than any-overlap: a component that merely reaches into the
    grown box from far away is not that lesion (measured: whole-volume false
    positives are 20-60x the annotation box and only graze its margin).
    Components are kept whole, so a segmentation running past the annotation is
    not cut off mid-lesion.
    """
    labels = sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk.Cast(prediction, sitk.sitkUInt8)))
    zs, xs, ys = np.nonzero(labels)
    if not zs.size:
        return np.zeros(labels.shape, dtype=bool)
    ids = labels[zs, xs, ys]
    sizes = np.bincount(ids)
    counts = sizes.astype(float)
    counts[counts == 0] = 1.0  # label 0 never appears in ids; keep the divide finite
    centroid_z = np.bincount(ids, weights=zs) / counts
    centroid_x = np.bincount(ids, weights=xs) / counts
    centroid_y = np.bincount(ids, weights=ys) / counts
    touching: set[int] = set()
    for lesion in lesions.itertuples():
        z, x, y = lesion_window(lesion, prediction.GetSpacing(), margin_mm, labels.shape)
        hit = (
            (centroid_z >= z.start) & (centroid_z < z.stop)
            & (centroid_x >= x.start) & (centroid_x < x.stop)
            & (centroid_y >= y.start) & (centroid_y < y.stop)
        )
        touching.update(
            int(value)
            for value in np.flatnonzero(hit)
            if value and sizes[value] >= MIN_VOXELS
        )
    if not touching:
        return np.zeros(labels.shape, dtype=bool)
    return np.isin(labels, sorted(touching))


def build_mask(
    scan_path: Path, prediction_path: Path, lesions: pd.DataFrame, margin_mm: float
) -> tuple[sitk.Image | None, int, int, str]:
    """Binary mask of the box-validated prediction, on the scan's grid."""
    if not prediction_path.exists():
        return None, 0, 0, MISSED
    try:
        scan = sitk.ReadImage(str(scan_path))
        prediction = sitk.ReadImage(str(prediction_path))
    except RuntimeError:  # truncated file that still passed the size check
        return None, 0, 0, UNREADABLE
    if geometry_mismatch(scan, prediction):
        prediction = resample_mask_to_scan(prediction, scan)
    predicted = int((sitk.GetArrayViewFromImage(prediction) > 0).sum())
    if not predicted:
        return None, 0, 0, MISSED
    keep = keep_near_boxes(prediction, lesions, margin_mm)
    kept = int(keep.sum())
    if not kept:
        return None, predicted, 0, REJECTED
    mask = sitk.GetImageFromArray(keep.astype(np.uint8))
    mask.CopyInformation(prediction)
    return mask, predicted, kept, MASKED


def stage(volumes: list[str], images: Path, input_dir: Path) -> None:
    """Fresh directory of scan symlinks for one inference invocation."""
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True)
    for volume in volumes:
        (input_dir / f"{volume}{SCAN_SUFFIX}").symlink_to(images / f"{volume}{SCAN_SUFFIX}")


def predict(input_dir: Path, output_dir: Path, repo: Path, model: str, device: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Upstream crashes on an empty queue (zip(*[]) in lcunet_batch_predict.main), so
    # never call it when every staged volume already has a prediction.
    if all((output_dir / f"seg_{path.name}").exists() for path in input_dir.glob("*.nii.gz")):
        return
    uv = Path.home() / ".local/bin/uv"  # not on PATH for non-interactive shells
    command = [
        str(uv) if uv.exists() else (shutil.which("uv") or "uv"),
        "run",
        "lcunet_batch_predict",
        "--model",
        model,
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--device",
        device,
        "--single_fold",
    ]
    logging.info(f"running {' '.join(command)}")
    # TTA off and parallel preprocessing: ~3 s/volume instead of ~120 s.
    env = {**os.environ, "LCUNET_TTA": "0", "LCUNET_PREPROC": "8", "LCUNET_EXPORT": "2"}
    subprocess.run(command, cwd=repo, env=env, check=True)


def self_check() -> None:
    """Box filter keeps the component inside the box and drops the distant one."""
    array = np.zeros((9, 40, 40), dtype=np.uint8)
    array[4, 18:23, 18:23] = 1  # 25 voxels on the box
    array[1, 2:6, 2:6] = 1  # far away
    array[4, 30, 30] = 1  # 1-voxel speck beside the box
    prediction = sitk.GetImageFromArray(array)
    prediction.SetSpacing((0.5, 0.5, 2.5))
    lesions = pd.DataFrame([{"x1": 18, "y1": 18, "x2": 26, "y2": 26, "z": 4}])
    keep = keep_near_boxes(prediction, lesions, margin_mm=1.0)
    assert keep[4, 20, 20], "component on the box must survive"
    assert not keep[1, 3, 3], "component far from the box must be dropped"
    assert not keep[4, 30, 30], "sub-threshold speck must be dropped"
    assert keep[4, 30, 30] is np.False_, "keep is a boolean array"
    assert keep.sum() == 25, f"kept {keep.sum()} voxels, expected the whole 25-voxel blob"
    logging.info("self-check passed")


def preprocess_npy(masks: Path, images: Path, volumes: list[str], args: argparse.Namespace) -> None:
    """Publish 1 mm NPY/IMAGES + NPY/SEGMENTATIONS and the radiomics component table."""
    jobs = [
        Job(
            batch="deeplesion",
            patient_id=volume.split("_")[0],
            stem=volume,
            scan_path=images / f"{volume}{SCAN_SUFFIX}",
            mask_path=masks / f"{volume}{MASK_SUFFIX}",
            img_path=args.npy_root / IMAGES_DIR / f"{volume}.npy",
            seg_path=args.npy_root / SEGMENTATIONS_DIR / args.mask_group / f"{volume}_mask.npy",
        )
        for volume in volumes
    ]
    table_path = args.npy_root / COMPONENTS_NAME
    old = pd.read_parquet(table_path) if table_path.exists() else None
    table = run(jobs, args.npy_root, table_path, workers=args.workers)
    if table is None:
        return
    if old is not None:  # resume: keep rows from earlier runs
        table = pd.concat([old, table]).drop_duplicates("node_id", keep="last")
    table["cohort"] = args.cohort  # PretrainDataset/dataset splits filter on this column
    table.to_parquet(table_path, index=False)
    logging.info(f"{len(table)} component rows -> {table_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_ROOT / CSV_NAME)
    parser.add_argument("--images", type=Path, default=DEFAULT_ROOT / IMAGES_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT / SEGMENTATIONS_DIR)
    parser.add_argument("--work", type=Path, default=DEFAULT_ROOT / "LCUNet")
    parser.add_argument("--repo", type=Path, default=LCUNET_REPO)
    parser.add_argument("--model", default="LCUNet+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--margin-mm", type=float, default=MARGIN_MM)
    parser.add_argument("--limit", type=int, default=0, help="first N volumes only")
    parser.add_argument("--npy-root", type=Path, default=DEFAULT_ROOT / "NPY")
    parser.add_argument("--mask-group", default="LESION")
    parser.add_argument("--cohort", default="train", help="train/val/test for radiomics")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-npy", action="store_true", help="skip the NPY stage")
    parser.add_argument("--skip-predict", action="store_true", help="reuse existing predictions")
    parser.add_argument("--overwrite", action="store_true", help="redo volumes that have a mask")
    parser.add_argument("--batch", type=int, default=500, help="volumes per inference call")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if args.self_check:
        self_check()
        return

    # Two runs sharing --work wipe each other's staged inputs mid-inference.
    args.work.mkdir(parents=True, exist_ok=True)
    lock = (args.work / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another run is using {args.work}") from None

    lesions = load_lesions(args.csv, args.images, args.limit)
    logging.info(
        f"{len(lesions)} in-lung lesions on {lesions.volume.nunique()} volumes "
        f"(margin {args.margin_mm} mm)"
    )
    todo = [
        volume
        for volume in sorted(lesions.volume.unique())
        if args.overwrite or not (args.out / f"{volume}{MASK_SUFFIX}").exists()
    ]
    logging.info(f"{len(todo)} volumes to process, {lesions.volume.nunique() - len(todo)} already done")

    input_dir = args.work / "input"
    prediction_dir = args.work / "predictions"
    path = args.out / MANIFEST_NAME
    for start in range(0, len(todo), args.batch):
        chunk = todo[start : start + args.batch]
        if not args.skip_predict:
            stage(chunk, args.images, input_dir)
            predict(input_dir, prediction_dir, args.repo, args.model, args.device)
        rows = []
        for volume in tqdm(chunk, desc=f"masks {start + 1}-{start + len(chunk)}", unit="vol"):
            selected = lesions[lesions.volume == volume]
            mask, predicted, kept, status = build_mask(
                args.images / f"{volume}{SCAN_SUFFIX}",
                prediction_dir / f"{PRED_PREFIX}{volume}{SCAN_SUFFIX}",
                selected,
                args.margin_mm,
            )
            if mask is not None:
                args.out.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(mask, str(args.out / f"{volume}{MASK_SUFFIX}"), True)
            rows.append(
                {"volume": volume, "lesions": len(selected), "predicted": predicted,
                 "kept": kept, "status": status}
            )
        manifest = pd.DataFrame(rows)
        args.out.mkdir(parents=True, exist_ok=True)
        if path.exists():  # keep rows from earlier runs (--limit, resume, --overwrite)
            old = pd.read_csv(path)
            manifest = pd.concat([old[~old.volume.isin(manifest.volume)], manifest], ignore_index=True)
        manifest.sort_values("volume", ignore_index=True).to_csv(path, index=False)
        logging.info(f"{Counter(manifest.status)} -> {path}")

    masked = sorted(item.name.removesuffix(MASK_SUFFIX) for item in args.out.glob(f"*{MASK_SUFFIX}"))
    logging.info(f"{len(masked)} masks in {args.out}")
    if not args.skip_npy and masked:
        preprocess_npy(args.out, args.images, masked, args)


if __name__ == "__main__":
    main()
