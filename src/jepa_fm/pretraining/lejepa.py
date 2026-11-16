import math
import os
import subprocess
from datetime import datetime
from pathlib import Path

import hydra
import mlflow
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from jepa_fm.data.datasets import PretrainDataset
from jepa_fm.data.utils import SplitType

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
    """Fixed, augmentation-free pass over the val split: (B, V, D, H, W)."""
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


def forward_latents(net: torch.nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """(B, V, D, H, W) volumes -> (B, V, C) latents.

    One encoder pass over all B * V volumes, then global average pooling for each view
    """
    b, v = imgs.shape[:2]
    features = net(imgs.flatten(0, 1).unsqueeze(1))  # (B*V, C, d, h, w)
    return features.mean(dim=(2, 3, 4)).view(b, v, -1)


def train_step(
    net: torch.nn.Module,
    projector: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    imgs: torch.Tensor,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    with torch.autocast(imgs.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        latents = forward_latents(net, imgs)

    b, v = latents.shape[:2]
    proj = projector(latents.float().flatten(0, 1)).view(b, v, -1)
    total, parts = loss_fn(proj)

    optimiser.zero_grad(set_to_none=True)
    total.backward()
    optimiser.step()

    return {"loss": total.item(), **{name: value.item() for name, value in parts.items()}}


@torch.no_grad()
def eval_step(
    net: torch.nn.Module,
    projector: torch.nn.Module,
    loss_fn: torch.nn.Module,
    imgs: torch.Tensor,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Same metrics as train_step, without the optimiser step. Needs .eval() upstream."""
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


def checkpoint_dir(run_name: str) -> Path:
    root = os.environ.get("CKPT_DIR")
    if not root:
        raise RuntimeError(
            "CKPT_DIR is not set. Point it at the checkpoint disk in .env"
        )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(root).expanduser() / f"{run_name}_{stamp}"


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
    """Restore state in place and return the epoch the checkpoint was written at."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["model"])
    projector.load_state_dict(ckpt["projector"])
    optimiser.load_state_dict(ckpt["optimiser"])
    if "scheduler" in ckpt:  # checkpoints from before the scheduler was saved
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
    """Short sha of the running code, suffixed -dirty outside a clean tree."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parents[3],  # hydra may have moved the cwd
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


class MetricLog:
    """Buffer step-indexed metrics and push them to MLflow in batches.

    A failed flush keeps the points for the next attempt instead of killing a
    multi-day run because the tracking box hiccupped.
    """

    def __init__(self, flush_every: int = 50, max_pending: int = 200):
        self.flush_every = flush_every
        self.max_pending = max_pending
        self.pending: dict[int, dict[str, float]] = {}

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.pending[step] = metrics
        if len(self.pending) > self.max_pending:  # tracking server is down
            for stale in sorted(self.pending)[: len(self.pending) - self.max_pending]:
                del self.pending[stale]
        if len(self.pending) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        try:
            for step in sorted(self.pending):
                mlflow.log_metrics(self.pending[step], step=step)
        except Exception as exc:  # logging must never be what ends a run
            print(f"mlflow flush failed, {len(self.pending)} points kept: {exc}")
            return
        self.pending.clear()


def build_model(
    cfg: DictConfig, device: torch.device
) -> tuple[torch.nn.Module, torch.nn.Module]:
    net = instantiate(cfg.model).to(device)
    projector = instantiate(cfg.projector).to(device)
    return net, projector


def build_optimiser(
    cfg: DictConfig, params: list[torch.nn.Parameter]
) -> torch.optim.Optimizer:
    """Split params so weight decay only reaches matrices and kernels.

    Biases and norm/batch-norm scales are 1-D; decaying them is never wanted.
    """
    return instantiate(
        cfg,
        _convert_="all",  # plain dicts/lists for torch, not OmegaConf containers
        params=[
            {"params": [p for p in params if p.dim() >= 2]},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ],
    )


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parents[3] / "config"),
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device available; pretraining runs on GPU only "
            "(check the driver and CUDA_VISIBLE_DEVICES)."
        )

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
    scheduler = build_lr_scheduler(
        optimiser, total_steps=len(train_loader) * cfg.trainer.epochs, **cfg.scheduler
    )
    run_ckpt_dir = checkpoint_dir(cfg.run_name)

    mirror_mlflow_env()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "jepa-fm-pretrain"))

    with mlflow.start_run(run_name=cfg.run_name):
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
        saved: list[Path] = []
        metrics_log = MetricLog()
        for epoch in range(start_epoch, cfg.trainer.epochs):
            net.train()
            projector.train()
            progress = tqdm(train_loader, desc=f"epoch {epoch}", unit="batch", leave=False)

            for imgs, _lesions in progress:
                metrics = train_step(
                    net,
                    projector,
                    optimiser,
                    loss_fn,
                    imgs.to(device, non_blocking=True),
                    amp_dtype,
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

            if cfg.trainer.val_every > 0 and (
                (epoch + 1) % cfg.trainer.val_every == 0
                or epoch == cfg.trainer.epochs - 1
            ):
                net.eval()
                projector.eval()
                sums: dict[str, float] = {}
                for imgs, _lesions in val_loader:
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

            if epoch == cfg.trainer.epochs - 1 or (epoch + 1) % cfg.trainer.ckpt_interval == 0:
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

        metrics_log.flush()


if __name__ == "__main__":
    main()
