from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

SPACING = (1.0, 1.0, 1.0)
HU_WINDOW = (-1024.0, 2048.0)
SCAN_PAD = -1024.0
MASK_PAD = 0
GEOMETRY_TOLERANCE_MM = 1e-3
SCAN_SUFFIX = ".nii.gz"
MASK_SUFFIX = "_mask.nii.gz"
IMAGES_DIR = "IMAGES"
SEGMENTATIONS_DIR = "SEGMENTATIONS"
TABLE_NAME = "components.parquet"

Row = dict[str, object]


@dataclass(frozen=True)
class Job:
    batch: str
    patient_id: str
    stem: str
    scan_path: Path
    mask_path: Path
    img_path: Path
    seg_path: Path


def _scan_dir(directory: Path) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    subdirs: list[Path] = []
    with os.scandir(directory) as entries:
        for entry in entries:
            (subdirs if entry.is_dir() else files).append(Path(entry.path))
    return files, subdirs


def index_files(root: Path, suffix: str, threads: int = 32) -> list[Path]:
    """Threaded walk: NFS readdir latency dominates, so visit dirs in parallel."""
    paths: list[Path] = []
    level = [root]
    while level:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            results = list(pool.map(_scan_dir, level))
        level = []
        for files, subdirs in results:
            paths.extend(path for path in files if path.name.endswith(suffix))
            level.extend(subdirs)
    return sorted(paths)


def make_job(
    stem: str, scan_path: Path, mask_path: Path, out: Path, mask_group: str
) -> Job:
    batch = scan_path.parent.parent.name
    patient_id = scan_path.parent.name
    return Job(
        batch=batch,
        patient_id=patient_id,
        stem=stem,
        scan_path=scan_path,
        mask_path=mask_path,
        img_path=out / IMAGES_DIR / batch / patient_id / f"{stem}.npy",
        seg_path=(
            out / SEGMENTATIONS_DIR / mask_group / batch / patient_id / f"{stem}_mask.npy"
        ),
    )


def geometry_mismatch(reference: sitk.Image, other: sitk.Image) -> str | None:
    if reference.GetSize() != other.GetSize():
        return f"size {reference.GetSize()} != {other.GetSize()}"
    scale = max(reference.GetSpacing())
    for name, first, second in (
        ("origin", reference.GetOrigin(), other.GetOrigin()),
        ("spacing", reference.GetSpacing(), other.GetSpacing()),
        ("direction", reference.GetDirection(), other.GetDirection()),
    ):
        delta = max(abs(a - b) for a, b in zip(first, second))
        if delta > GEOMETRY_TOLERANCE_MM * scale:
            return f"{name} differs by {delta:.4f}"
    return None


def resample_to_isotropic(
    scan: sitk.Image, spacing: tuple[float, ...] = SPACING
) -> sitk.Image:
    """1 mm^3 grid anchored on the scan's own origin and direction."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputDirection(scan.GetDirection())
    resampler.SetOutputOrigin(scan.GetOrigin())
    resampler.SetSize(
        [
            int(round(size * step))
            for size, step in zip(scan.GetSize(), scan.GetSpacing())
        ]
    )
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(SCAN_PAD)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    return resampler.Execute(scan)


def resample_mask_to_scan(mask: sitk.Image, scan: sitk.Image) -> sitk.Image:
    """Nearest-neighbour mask on the scan's grid, so both arrays are voxel-identical."""
    return sitk.Resample(
        mask,
        scan,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        MASK_PAD,
        sitk.sitkUInt8,
    )


def mm_to_array_index(image: sitk.Image, point: tuple[float, ...]) -> np.ndarray:
    """Physical (x, y, z) mm -> array index (z, y, x)."""
    direction = np.array(image.GetDirection()).reshape(3, 3)
    offset = np.array(point) - np.array(image.GetOrigin())
    return (np.linalg.inv(direction) @ offset / np.array(image.GetSpacing()))[::-1]


def process(job: Job) -> tuple[str, list[Row], list[str]]:
    if not job.scan_path.exists():
        return "missing_scan", [], [f"{job.stem}: no scan at {job.scan_path}"]

    scan = sitk.DICOMOrient(sitk.ReadImage(str(job.scan_path)), "LPS")
    mask = sitk.DICOMOrient(sitk.ReadImage(str(job.mask_path)), "LPS")

    messages = []
    mismatch = geometry_mismatch(scan, mask)
    if mismatch:
        messages.append(
            f"{job.stem}: mask geometry differs from scan ({mismatch}); "
            "mask is resampled onto the scan grid by physical position"
        )

    scan_1mm = resample_to_isotropic(scan)
    mask_1mm = resample_mask_to_scan(mask, scan_1mm)

    scan_array = np.clip(sitk.GetArrayFromImage(scan_1mm), *HU_WINDOW)
    scan_array = np.ascontiguousarray(scan_array).round().astype(np.int16)
    binary_mask = np.ascontiguousarray(sitk.GetArrayFromImage(mask_1mm) > 0)
    binary_mask = binary_mask.astype(np.int16)
    if binary_mask.shape != scan_array.shape:
        return (
            "error",
            [],
            [f"{job.stem}: shape {binary_mask.shape} != {scan_array.shape}"],
        )
    if binary_mask.max(initial=0) == 0:
        return "empty_mask", [], [f"{job.stem}: mask empty after resampling"]

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(
        sitk.RelabelComponent(
            sitk.ConnectedComponent(sitk.Cast(mask_1mm > 0, sitk.sitkUInt8)),
            sortByObjectSize=True,
        )
    )
    n_components = stats.GetNumberOfLabels()

    job.img_path.parent.mkdir(parents=True, exist_ok=True)
    job.seg_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(job.img_path, scan_array)
    np.save(job.seg_path, binary_mask)

    rows = []
    for label in stats.GetLabels():
        centre = stats.GetCentroid(label)
        index = mm_to_array_index(scan_1mm, centre)
        n_voxels = int(stats.GetNumberOfPixels(label))
        rows.append(
            {
                "node_id": f"{job.stem}_c{label - 1}",
                "scan_id": job.stem,
                "batch": job.batch,
                "patient_id": job.patient_id,
                "npy_img_path": str(job.img_path),
                "npy_seg_path": str(job.seg_path),
                "label": label,
                "n_voxels": n_voxels,
                "volume_mm3": float(n_voxels),
                "n_components": n_components,
                "centroid_z": float(index[0]),
                "centroid_y": float(index[1]),
                "centroid_x": float(index[2]),
                "centroid_z_mm": float(centre[2]),
                "centroid_y_mm": float(centre[1]),
                "centroid_x_mm": float(centre[0]),
            }
        )
    return "ok", rows, messages


def _setup_worker(sitk_threads: int) -> None:
    # one thread per process: the pool already saturates the cores
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(sitk_threads)


def run(
    jobs: list[Job],
    out: Path,
    table_path: Path | None = None,
    workers: int = 8,
    overwrite: bool = False,
    sitk_threads: int = 1,
) -> pd.DataFrame | None:
    table_path = table_path or out / TABLE_NAME
    if not overwrite:
        jobs = [
            job for job in jobs if not (job.img_path.exists() and job.seg_path.exists())
        ]
    logging.info(f"{len(jobs)} pairs to process -> {out}")

    rows: list[Row] = []
    counts: dict[str, int] = {}

    def record(result: tuple[str, list[Row], list[str]]) -> None:
        status, job_rows, messages = result
        counts[status] = counts.get(status, 0) + 1
        rows.extend(job_rows)
        for message in messages:
            logging.warning(message)

    if workers <= 1:
        _setup_worker(sitk_threads)
        for job in tqdm(jobs):
            record(process(job))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_setup_worker,
            initargs=(sitk_threads,),
        ) as pool:
            for result in tqdm(pool.map(process, jobs), total=len(jobs)):
                record(result)

    frame = None
    if rows:
        frame = pd.DataFrame(rows).sort_values(["scan_id", "label"], ignore_index=True)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(table_path, index=False)
        logging.info(
            f"{len(frame)} component rows from {frame['scan_id'].nunique()} scans "
            f"-> {table_path}"
        )
    else:
        logging.info("no components found; table not written")

    logging.info("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return frame
