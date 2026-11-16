import math
import os
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


def build_train_loader(cfg: DictConfig) -> DataLoader:
    """A batch is (B, V, D, H, W): B lesions, V random views of each."""
    dataset = PretrainDataset(
        SplitType.TRAIN,
        cfg.data.meta_path,
        cfg.data.crop_size,
        views=cfg.data.views,
        voxel_format=cfg.data.voxel_format,
    )

    loader_kwargs = dict(
        batch_size=cfg.trainer.lesions_per_batch,
        num_workers=cfg.trainer.num_workers,
        pin_memory=cfg.loader.pin_memory,
    )
    if cfg.trainer.num_workers > 0:
        loader_kwargs |= dict(
            persistent_workers=cfg.loader.persistent_workers,
            prefetch_factor=cfg.loader.prefetch_factor,
        )

    return DataLoader(dataset, shuffle=True, drop_last=True, **loader_kwargs)


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
    cfg: DictConfig,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": net.state_dict(),
            "projector": projector.state_dict(),
            "optimiser": optimiser.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )
    return path


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

    net = instantiate(cfg.model).to(device)
    projector = instantiate(cfg.projector).to(device)
    loss_fn = instantiate(cfg.loss).to(device)
    optimiser = instantiate(
        cfg.optim, params=[*net.parameters(), *projector.parameters()]
    )
    train_loader = build_train_loader(cfg)
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
        mlflow.log_dict(resolved, "config.yaml")

        step = 0
        saved: list[Path] = []
        for epoch in range(cfg.trainer.epochs):
            net.train()
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
                    mlflow.log_metrics(
                        {f"train/{name}": value for name, value in metrics.items()}
                        | {"train/lr": scheduler.get_last_lr()[0]},
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
                        cfg,
                    )
                )
                prune_checkpoints(saved, cfg.trainer.keep_last)


if __name__ == "__main__":
    main()
