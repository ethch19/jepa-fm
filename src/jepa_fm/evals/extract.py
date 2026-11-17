import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from jepa_fm.data.downstream import (
    Endpoint,
    EvaluationDataset,
    parse_endpoint,
    validate_endpoint,
)
from jepa_fm.models.utils import forward_latents
from jepa_fm.pretraining.lejepa import AMP_DTYPE, git_sha, mirror_mlflow_env

REPO = Path(__file__).parents[3]


@dataclass(frozen=True)
class Cohort:
    name: str
    role: str
    meta_file: Path
    endpoint: Endpoint


def resolve_meta_file(name: str, meta_path: Path, meta_file: str | None) -> Path:
    if meta_file:
        path = Path(meta_file).expanduser()
        if not path.is_absolute():
            path = meta_path / path
    else:
        candidates = [meta_path / f"{name}{ext}" for ext in (".parquet", ".csv")]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise ValueError(
                f"cohort {name!r}: no metadata file; tried "
                f"{[str(candidate) for candidate in candidates]}"
            )
    if not path.is_file():
        raise ValueError(f"cohort {name!r}: metadata file not found at {path}")
    return path


def parse_datasets(cfg: DictConfig, meta_path: Path) -> list[Cohort]:
    cohorts = []
    for raw in cfg.datasets:
        name = raw["name"]
        if (role := raw.get("role")) not in {"train", "test"}:
            raise ValueError(
                f"cohort {name!r}: role must be 'train' or 'test', got {role!r}"
            )
        cohorts.append(
            Cohort(
                name,
                role,
                resolve_meta_file(name, meta_path, raw.get("meta_file")),
                parse_endpoint(raw["endpoint"]),
            )
        )
    if not cohorts:
        raise ValueError(
            "no datasets declared in config/eval.yaml; add at least one cohort"
        )
    return cohorts


def load_encoder(
    ckpt_path: Path, device: torch.device
) -> tuple[nn.Module, dict, str]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    config = ckpt.get("config", {})
    data = config.get("data", {})
    for key in ("crop_size", "voxel_format"):
        if key not in data:
            raise ValueError(f"checkpoint config has no data: {key}")
    model_cfg = config["model"]
    net = instantiate(OmegaConf.create(model_cfg))
    net.load_state_dict(ckpt["model"])
    net.to(device).eval()
    return (
        net,
        {"crop_size": list(data["crop_size"]), "voxel_format": data["voxel_format"]},
        model_cfg["_target_"],
    )


@torch.no_grad()
def extract_cohort(
    net: nn.Module,
    ds: EvaluationDataset,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, list[str]]:
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    net.eval()
    features: list[np.ndarray] = []
    uuids: list[str] = []
    for imgs, uuid in tqdm(loader, desc=f"extract {ds.name}", unit="batch", leave=False):
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            latents = forward_latents(net, imgs.to(device, non_blocking=True))[:, 0]
        features.append(latents.float().cpu().numpy())
        uuids.extend(uuid)
    return np.concatenate(features), uuids


def cache_paths(cache_dir: Path, ckpt: Path, name: str) -> tuple[Path, Path, Path]:
    # ckpt.stem alone collides between runs (every run has an epoch_10.pt); the parent
    # dir is run_name_<stamp>_<run_id>, so keep it as the run level of the cache.
    root = cache_dir / ckpt.parent.name / ckpt.stem
    return root / f"{name}.npz", root / f"{name}.labels.parquet", root / f"{name}.json"


def cache_mismatch(npz: Path, sidecar: Path, expected: dict) -> str | None:
    labels = npz.with_name(npz.name.replace(".npz", ".labels.parquet"))
    for path in (npz, labels, sidecar):
        if not path.is_file():
            return f"missing {path.name}"
    try:
        found = json.loads(sidecar.read_text())
    except json.JSONDecodeError as err:
        return f"unreadable {sidecar.name}: {err}"
    for key, value in expected.items():
        if found.get(key) != value:
            return f"{key} {found.get(key)!r} != {value!r}"
    return None


def write_cache(
    npz: Path,
    labels_path: Path,
    sidecar_path: Path,
    features: np.ndarray,
    uuids: list[str],
    labels: pd.DataFrame,
    sidecar: dict,
) -> dict:
    if features.shape[0] != len(labels):
        raise RuntimeError(
            f"{labels_path.name}: {features.shape[0]} feature rows for "
            f"{len(labels)} label rows"
        )
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, features=features, uuid=np.asarray(uuids))
    labels.to_parquet(labels_path)
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    return sidecar


def log_extraction(
    name: str,
    sidecar: dict,
    summary: dict[str, float],
    experiment: str,
    decision: str,
) -> None:
    mirror_mlflow_env()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment(experiment)
    values = {**sidecar, **summary}
    with mlflow.start_run(run_name=f"extract-{name}"):
        mlflow.set_tags(
            {
                "dataset": name,
                "role": sidecar["role"],
                "ckpt": sidecar["ckpt"],
                "cache_decision": decision,
            }
        )
        mlflow.log_params(
            {
                key: value if isinstance(value, (str, int, float, bool)) else str(value)
                for key, value in values.items()
            }
        )


@hydra.main(version_base=None, config_path=str(REPO / "config"), config_name="eval")
def main(cfg: DictConfig) -> None:
    load_dotenv(REPO / ".env")
    ckpt = Path(cfg.ckpt).expanduser().resolve()
    if not ckpt.is_file():
        raise ValueError(f"ckpt must point at a checkpoint file, got {cfg.ckpt!r}")
    if cfg.amp not in (None, "bf16"):
        raise ValueError(f"amp must be null or 'bf16', got {cfg.amp!r}")
    if cfg.cache_dir:
        cache_dir = Path(cfg.cache_dir).expanduser()
    elif root := os.environ.get("CKPT_DIR"):
        cache_dir = Path(root).expanduser() / "embeddings"
    else:
        raise RuntimeError(
            "CKPT_DIR is not set and cache_dir is null; set one of them (.env is read from the repo root)"
        )

    device = torch.device(cfg.device)
    amp_dtype = AMP_DTYPE if cfg.amp else None
    cohorts = parse_datasets(cfg, Path(cfg.meta_path).expanduser())
    net, data_cfg, model_target = load_encoder(ckpt, device)
    stat = ckpt.stat()
    expected = {
        "ckpt_size": stat.st_size,
        "ckpt_mtime_ns": stat.st_mtime_ns,
        "crop_size": data_cfg["crop_size"],
        "voxel_format": data_cfg["voxel_format"],
        "model_target": model_target,
    }

    for cohort in cohorts:
        ds = EvaluationDataset(
            cohort.meta_file, data_cfg["crop_size"], data_cfg["voxel_format"]
        )
        summary = validate_endpoint(ds.labels, cohort.endpoint, cohort.name)
        npz, labels_path, sidecar_path = cache_paths(cache_dir, ckpt, cohort.name)

        if (reason := "force: true" if cfg.force else cache_mismatch(npz, sidecar_path, expected)) is None:
            sidecar, decision = json.loads(sidecar_path.read_text()), "hit"
        else:
            print(f"cohort={cohort.name}: recomputing ({reason})")
            features, uuids = extract_cohort(
                net, ds, device, amp_dtype, cfg.batch_size, cfg.num_workers
            )
            sidecar = write_cache(
                npz,
                labels_path,
                sidecar_path,
                features,
                uuids,
                ds.labels,
                {
                    "ckpt": str(ckpt),
                    "ckpt_size": stat.st_size,
                    "ckpt_mtime_ns": stat.st_mtime_ns,
                    "crop_size": data_cfg["crop_size"],
                    "voxel_format": data_cfg["voxel_format"],
                    "model_target": model_target,
                    "feature_dim": int(features.shape[1]),
                    "n_rows": int(features.shape[0]),
                    "amp": cfg.amp,
                    "git_sha": git_sha(),
                    "written_at": datetime.now().isoformat(timespec="seconds"),
                    "role": cohort.role,
                    "endpoint": asdict(cohort.endpoint),
                },
            )
            decision = "written"

        print(
            f"cohort={cohort.name} rows={len(ds)} dim={sidecar['feature_dim']} "
            f"cache={decision}"
        )
        log_extraction(cohort.name, sidecar, summary, cfg.mlflow.experiment, decision)
