import itertools
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import hydra
import mlflow
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from jepa_fm.data.datasets import PretrainDataset
from jepa_fm.data.utils import SplitType
from jepa_fm.models.utils import forward_latents
from jepa_fm.probe import EmbeddingMetric
from jepa_fm.probe.extract import extract_embeddings, group_rows
from jepa_fm.utils import MetricLog, log_metric_figures

AMP_DTYPE = torch.bfloat16


def _loader_kwargs(cfg: DictConfig) -> dict:
    kwargs = dict(
        batch_size=cfg.trainer.lesions_per_batch,
        num_workers=cfg.trainer.num_workers,
        pin_memory=cfg.loader.pin_memory,
    )
    if cfg.trainer.num_workers > 0:
        kwargs |= dict(
            persistent_workers=cfg.loader.persistent_workers,
            prefetch_factor=cfg.loader.prefetch_factor,
        )
    return kwargs


def build_train_loader(cfg: DictConfig) -> DataLoader:
    """A batch is (B, V, D, H, W): B lesions, V random views of each."""
    dataset = PretrainDataset(
        SplitType.TRAIN,
        cfg.data.meta_path,
        cfg.data.crop_size,
        views=cfg.data.views,
        voxel_format=cfg.data.voxel_format,
    )
    return DataLoader(dataset, shuffle=True, drop_last=True, **_loader_kwargs(cfg))


def build_val_loader(cfg: DictConfig) -> DataLoader:
    dataset = PretrainDataset(
        SplitType.VAL,
        cfg.data.meta_path,
        cfg.data.crop_size,
        views=cfg.data.views,
        inference=True,
        voxel_format=cfg.data.voxel_format,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"no {SplitType.VAL} rows under {cfg.data.meta_path}")
    return DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(cfg) | {"persistent_workers": False},
    )


def build_probe_loader(cfg: DictConfig) -> DataLoader:
    dataset = PretrainDataset(
        SplitType.TRAIN,
        cfg.data.meta_path,
        cfg.data.crop_size,
        views=1,
        inference=True,
        voxel_format=cfg.data.voxel_format,
    )
    size = len(dataset)
    if cfg.probe.samples is not None:
        size = min(cfg.probe.samples, size)
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(cfg.seed)
    )[:size].tolist()
    return DataLoader(
        Subset(dataset, indices),
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(cfg) | {"persistent_workers": False},
    )


def build_lr_scheduler(
    optimiser: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 1e-3,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup, then cosine decay from lr to min_lr_ratio * lr.

    Stepped once per optimiser step, in main.
    """

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimiser, factor)


def train_step(
    net: torch.nn.Module,
    projector: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    imgs: torch.Tensor,
    amp_dtype: torch.dtype | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    with torch.autocast(imgs.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        latents = forward_latents(net, imgs)

    b, v = latents.shape[:2]
    proj = projector(latents.float().flatten(0, 1)).view(b, v, -1)
    total, parts = loss_fn(proj)

    optimiser.zero_grad(set_to_none=True)
    total.backward()

    params = [
        p for p in itertools.chain(net.parameters(), projector.parameters()) if p.grad is not None
    ]
    if max_grad_norm:
        # returns the pre-clipping norm, which is the number worth watching
        grad_norm = torch.nn.utils.clip_grad_norm_(
            params, max_grad_norm, error_if_nonfinite=False
        )
    else:
        grad_norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(p.grad) for p in params])
        )
    optimiser.step()

    metrics: dict[str, float] = {
        "loss": total.item(),
        "grad_norm": float(grad_norm),
    }
    metrics |= {name: value.item() for name, value in parts.items()}
    return metrics


@torch.no_grad()
def eval_step(
    net: torch.nn.Module,
    projector: torch.nn.Module,
    loss_fn: torch.nn.Module,
    imgs: torch.Tensor,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    with torch.autocast(imgs.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        latents = forward_latents(net, imgs)

    b, v = latents.shape[:2]
    proj = projector(latents.float().flatten(0, 1)).view(b, v, -1)
    total, parts = loss_fn(proj)

    return {"loss": total.item(), **{name: value.item() for name, value in parts.items()}}


def flatten_config(tree: dict, prefix: str = "") -> dict[str, str | int | float | bool]:
    """Convert config to flat structure scalars"""
    flat: dict[str, str | int | float | bool] = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat |= flatten_config(value, f"{name}.")
        elif isinstance(value, (str, int, float, bool)):
            flat[name] = value
        else:
            flat[name] = str(value)
    return flat


def checkpoint_dir(run_name: str, run_id: str) -> Path:
    root = os.environ.get("CKPT_DIR")
    if not root:
        raise RuntimeError(
            "CKPT_DIR is not set. Point it at the checkpoint disk in .env"
        )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(root).expanduser() / f"{run_name}_{stamp}_{run_id}"


def save_checkpoint(
    path: Path,
    epoch: int,
    net: torch.nn.Module,
    projector: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: DictConfig,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": net.state_dict(),
            "projector": projector.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path,
    net: torch.nn.Module,
    projector: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["model"])
    projector.load_state_dict(ckpt["projector"])
    optimiser.load_state_dict(ckpt["optimiser"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"]


def prune_checkpoints(saved: list[Path], keep: int) -> None:
    while len(saved) > keep:
        saved.pop(0).unlink(missing_ok=True)


def mirror_mlflow_env() -> None:
    for ours, theirs in (
        ("MLFLOW_USERNAME", "MLFLOW_TRACKING_USERNAME"),
        ("MLFLOW_PASSWORD", "MLFLOW_TRACKING_PASSWORD"),
    ):
        if (value := os.environ.get(ours)) is not None:
            os.environ[theirs] = value


def git_sha() -> str:
    """Short sha of the running code"""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def build_model(
    cfg: DictConfig, device: torch.device
) -> tuple[torch.nn.Module, torch.nn.Module]:
    net = instantiate(cfg.model).to(device)
    projector = instantiate(cfg.projector).to(device)
    return net, projector


def build_optimiser(
    cfg: DictConfig, params: list[torch.nn.Parameter]
) -> torch.optim.Optimizer:
    return instantiate(
        cfg,
        _convert_="all",
        params=[
            {"params": [p for p in params if p.dim() >= 2]},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ],
    )


def run_probe(
    net: torch.nn.Module,
    projector: torch.nn.Module,
    metrics: dict[str, EmbeddingMetric],
    loader: DataLoader,
    cfg: DictConfig,
    device: torch.device,
    epoch: int,
    epoch_log: MetricLog,
) -> None:
    started = time.perf_counter()
    embeddings, labels = extract_embeddings(
        net, projector, loader, cfg.probe.spaces, device, AMP_DTYPE
    )
    values: dict[str, float] = {
        "probe/d": float(next(iter(embeddings.values())).shape[1])
    }

    for space, z in embeddings.items():
        for group, rows in group_rows(labels, cfg.probe.group_by).items():
            idx = torch.as_tensor(rows)
            if len(idx) < 4 * z.shape[1]:
                # a rank cannot exceed the row count
                print(
                    f"probe: {space}/{group} has {len(idx)} rows, "
                    f"rankme saturates near {len(idx)}"
                )
            for name, metric in metrics.items():
                values |= {
                    f"probe/{space}/{group}/{name}/{key}": value
                    for key, value in metric(z[idx]).items()
                }

    values |= {
        "probe/n": float(len(labels)),
        "probe/seconds": time.perf_counter() - started,
    }
    epoch_log.log(values, step=epoch)
    epoch_log.flush()


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parents[3] / "config"),
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    load_dotenv(Path(__file__).parents[3] / ".env")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available")

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    amp_dtype = AMP_DTYPE

    net, projector = build_model(cfg, device)
    loss_fn = instantiate(cfg.loss).to(device)
    optimiser = build_optimiser(
        cfg.optim, [*net.parameters(), *projector.parameters()]
    )
    train_loader = build_train_loader(cfg)
    val_loader = build_val_loader(cfg)
    probe_loader = build_probe_loader(cfg)
    probe_metrics = instantiate(cfg.probe.metrics)
    scheduler = build_lr_scheduler(
        optimiser, total_steps=len(train_loader) * cfg.trainer.epochs, **cfg.scheduler
    )
    mirror_mlflow_env()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "jepa-fm-pretrain"))

    with mlflow.start_run(run_name=cfg.run_name) as run:
        run_ckpt_dir = checkpoint_dir(cfg.run_name, run.info.run_id)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_params(flatten_config(resolved))
        mlflow.log_param("ckpt_dir", str(run_ckpt_dir))
        mlflow.log_param("git_sha", git_sha())
        mlflow.log_dict(resolved, "config.yaml")

        start_epoch = 0
        if cfg.trainer.resume_from:
            start_epoch = (
                load_checkpoint(
                    Path(cfg.trainer.resume_from).expanduser(),
                    net,
                    projector,
                    optimiser,
                    scheduler,
                )
                + 1
            )
            mlflow.log_param("resumed_from_epoch", start_epoch - 1)

        step = 0
        n_ckpt = 0
        saved: list[Path] = []
        metrics_log = MetricLog()
        epoch_log = MetricLog()
        for epoch in range(start_epoch, cfg.trainer.epochs):
            net.train()
            projector.train()
            progress = tqdm(train_loader, desc=f"epoch {epoch}", unit="batch", leave=False)

            for imgs, _, _, _ in progress:
                metrics = train_step(
                    net,
                    projector,
                    optimiser,
                    loss_fn,
                    imgs.to(device, non_blocking=True),
                    amp_dtype,
                    cfg.trainer.max_grad_norm,
                )
                step += 1
                scheduler.step()
                progress.set_postfix(**{k: f"{v:.4f}" for k, v in metrics.items()})

                if step % cfg.trainer.log_every == 0:
                    metrics_log.log(
                        {f"train/{name}": value for name, value in metrics.items()}
                        | {"train/lr": scheduler.get_last_lr()[0]},
                        step=step,
                    )

            is_last = epoch == cfg.trainer.epochs - 1
            if cfg.trainer.val_every > 0 and (
                is_last or (epoch + 1) % cfg.trainer.val_every == 0
            ):
                net.eval()
                projector.eval()
                sums: dict[str, float] = {}
                for imgs, _, _, _ in val_loader:
                    metrics = eval_step(
                        net,
                        projector,
                        loss_fn,
                        imgs.to(device, non_blocking=True),
                        amp_dtype,
                    )
                    sums = {k: sums.get(k, 0.0) + v for k, v in metrics.items()}
                metrics_log.log(
                    {f"val/{name}": total / len(val_loader) for name, total in sums.items()},
                    step=step,
                )

            if is_last or (epoch + 1) % cfg.trainer.ckpt_interval == 0:
                saved.append(
                    save_checkpoint(
                        run_ckpt_dir / f"epoch_{epoch:04d}.pt",
                        epoch,
                        net,
                        projector,
                        optimiser,
                        scheduler,
                        cfg,
                    )
                )
                prune_checkpoints(saved, cfg.trainer.keep_last)
                if n_ckpt % cfg.probe.every_n_ckpt == 0 or is_last:
                    run_probe(
                        net,
                        projector,
                        probe_metrics,
                        probe_loader,
                        cfg,
                        device,
                        epoch,
                        epoch_log,
                    )
                n_ckpt += 1

        metrics_log.flush()
        epoch_log.flush()
        try:
            log_metric_figures({**metrics_log.history, **epoch_log.history})
        except Exception as exc:
            print(f"metric figures failed: {exc}")


if __name__ == "__main__":
    main()
