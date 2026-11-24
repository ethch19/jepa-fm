from __future__ import annotations

import argparse
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from jepa_fm.preprocessing.utils import (
    MASK_SUFFIX,
    SCAN_SUFFIX,
    index_files,
    make_job,
    run,
)

DEFAULT_SCAN_ROOT = Path("/mnt/rds/NLST/nifti")
DEFAULT_MASK_ROOT = Path("/mnt/rds/NLST/Masks/tumour_masks")
DEFAULT_OUT = Path("/mnt/ethch/data/NLST/npy")
DEFAULT_PARTICIPANTS = Path("/mnt/rds/NLST/Clinical/participant_d040722.csv")
REPORT_NAME = "nlst_scan_selection.csv"
WORKLIST_NAME = "nlst_segmentation_worklist.csv"

MATCHED = "Matched"
MATCHED_LATER = "Matched later scan"
NO_SCAN = "No matching or later scan"

# topogram: 2 slices at ~265 mm z spacing, no anatomy worth segmenting
SCOUT_MARKER = "SCOUT"
# a re-reconstruction of the same series with wider z coverage; it wins over the base
EQ_SUFFIX = "_Eq_1"

WORKLIST_COLUMNS = [
    "patient_id",
    "cancyr",
    "scan_stem",
    "scan_year",
    "batch",
    "in_picks",
    "scan_path",
    "mask_path",
]


@dataclass(frozen=True)
class Scan:
    patient_id: str
    year: int
    stem: str
    path: Path


def stem_year(stem: str) -> int | None:
    """`<patient_id>_T<year>_<series>` -> year, or None if the name does not match."""
    parts = stem.split("_")
    if len(parts) < 3 or not parts[1].startswith("T"):
        return None
    try:
        return int(parts[1][1:])
    except ValueError:
        return None


def load_cancyr(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path, low_memory=False, usecols=["pid", "cancyr"])
    return {
        str(int(patient_id)): int(year)
        for patient_id, year in zip(frame.pid, frame.cancyr)
        if pd.notna(year)
    }


def is_scout(stem: str) -> bool:
    return SCOUT_MARKER in stem.upper()


def index_scans(scan_root: Path) -> dict[str, list[Scan]]:
    """Usable scans per patient: no topograms, one volume per series."""
    kept: dict[str, dict[str, Scan]] = defaultdict(dict)
    for path in index_files(scan_root, SCAN_SUFFIX):
        stem = path.name[: -len(SCAN_SUFFIX)]
        year = stem_year(stem)
        if year is None or is_scout(stem):
            continue
        patient_id = path.parent.name
        series = kept[patient_id]
        key = stem.removesuffix(EQ_SUFFIX)
        if key not in series or stem.endswith(EQ_SUFFIX):
            series[key] = Scan(patient_id, year, stem, path)
    return {patient_id: list(series.values()) for patient_id, series in kept.items()}


def index_masks(mask_root: Path) -> dict[str, Path]:
    return {
        path.name[: -len(MASK_SUFFIX)]: path
        for path in index_files(mask_root, MASK_SUFFIX)
    }


def pick_scans(
    scans: dict[str, list[Scan]], cancyr: dict[str, int]
) -> tuple[list[tuple[Scan, str]], list[tuple[str, int]]]:
    picks: list[tuple[Scan, str]] = []
    unmatched: list[tuple[str, int]] = []
    for patient_id in sorted(cancyr):
        target = cancyr[patient_id]
        found = scans.get(patient_id, [])
        exact = [scan for scan in found if scan.year == target]
        if exact:
            picks.extend((scan, MATCHED) for scan in exact)
            continue
        later = [scan for scan in found if scan.year > target]
        if later:
            year = min(scan.year for scan in later)
            picks.extend((scan, MATCHED_LATER) for scan in later if scan.year == year)
            continue
        unmatched.append((patient_id, target))
    return picks, unmatched


def selection_table(
    picks: list[tuple[Scan, str]],
    unmatched: list[tuple[str, int]],
    masks: dict[str, Path],
    cancyr: dict[str, int],
) -> pd.DataFrame:
    rows = [
        {
            "patient_id": scan.patient_id,
            "cancyr": cancyr[scan.patient_id],
            "status": status,
            "scan_stem": scan.stem,
            "scan_year": scan.year,
            "has_mask": scan.stem in masks,
            "scan_path": str(scan.path),
        }
        for scan, status in picks
    ]
    rows.extend(
        {
            "patient_id": patient_id,
            "cancyr": target,
            "status": NO_SCAN,
            "scan_stem": pd.NA,
            "scan_year": pd.NA,
            "has_mask": False,
            "scan_path": pd.NA,
        }
        for patient_id, target in unmatched
    )
    return pd.DataFrame(rows).sort_values(
        ["patient_id", "scan_year", "scan_stem"], na_position="last"
    )


def worklist_table(
    scans: dict[str, list[Scan]],
    masks: dict[str, Path],
    cancyr: dict[str, int],
    mask_root: Path,
    picks: list[tuple[Scan, str]],
) -> pd.DataFrame:
    """Every usable scan still missing a mask, plus where that mask must land."""
    picked = {scan.stem for scan, _ in picks}
    rows = [
        {
            "patient_id": scan.patient_id,
            "cancyr": cancyr.get(scan.patient_id, pd.NA),
            "scan_stem": scan.stem,
            "scan_year": scan.year,
            "batch": scan.path.parent.parent.name,
            "in_picks": scan.stem in picked,
            "scan_path": str(scan.path),
            "mask_path": str(
                mask_root / scan.path.parent.parent.name / f"{scan.stem}{MASK_SUFFIX}"
            ),
        }
        for patient_scans in scans.values()
        for scan in patient_scans
        if scan.stem not in masks
    ]
    return pd.DataFrame(rows, columns=WORKLIST_COLUMNS).sort_values(
        ["patient_id", "scan_year", "scan_stem"], ignore_index=True
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, default=DEFAULT_SCAN_ROOT)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--participants", type=Path, default=DEFAULT_PARTICIPANTS)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--worklist", type=Path, default=None)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sitk-threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    cancyr = load_cancyr(args.participants)
    scans = index_scans(args.scan_root)
    masks = index_masks(args.mask_root)
    picks, unmatched = pick_scans(scans, cancyr)

    jobs = [
        make_job(scan.stem, scan.path, masks[scan.stem], args.out, args.mask_root.name)
        for scan, _ in picks
        if scan.stem in masks
    ]

    report = selection_table(picks, unmatched, masks, cancyr)
    report_path = args.report or args.out / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    worklist = worklist_table(scans, masks, cancyr, args.mask_root, picks)
    worklist_path = args.worklist or args.out / WORKLIST_NAME
    worklist_path.parent.mkdir(parents=True, exist_ok=True)
    worklist.to_csv(worklist_path, index=False)

    statuses = Counter(status for _, status in picks)
    collisions = Counter((scan.patient_id, scan.year) for scan, _ in picks)
    logging.info(
        f"{sum(len(v) for v in scans.values())} scans for {len(scans)} patients, "
        f"{len(masks)} masks, {len(cancyr)} cancyr values"
    )
    logging.info(
        f"{MATCHED}={statuses[MATCHED]}, {MATCHED_LATER}={statuses[MATCHED_LATER]}, "
        f"{NO_SCAN}={len(unmatched)}, "
        f"scan_without_mask={sum(1 for scan, _ in picks if scan.stem not in masks)}, "
        f"year_with_multiple_series={sum(1 for n in collisions.values() if n > 1)}"
    )
    logging.info(f"selection report: {report_path}")
    logging.info(
        f"segmentation worklist: {len(worklist)} scans "
        f"({int(worklist['in_picks'].sum())} in current picks) -> {worklist_path}"
    )
    logging.info(f"{len(jobs)} scan+mask pairs to preprocess")

    if args.limit:
        jobs = jobs[: args.limit]
    workers = args.workers or os.cpu_count() or 1
    logging.info(f"workers={workers}, sitk_threads={args.sitk_threads}")
    run(jobs, args.out, args.table, workers, args.overwrite, args.sitk_threads)


if __name__ == "__main__":
    main()
