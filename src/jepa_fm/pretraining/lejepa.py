import torch
from torch.utils.data import DataLoader
import hydra
import mlflow
from omegaconf import DictConfig


from jepa_fm.data.datasets import PretrainDataset
from jepa_fm.data.utils import SplitType
from jepa_fm.models.convnextv2 import convnextv2_huge


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    mlflow.set_experiment("LeJEPA LC-FM")
    torch.manual_seed(123)

    train_ds = PretrainDataset(
        SplitType.TRAIN, cfg.data.train.meta_path, cfg.data.train.views
    )
    val_ds = PretrainDataset(
        SplitType.VAL, cfg.data.train.meta_path, cfg.data.train.views
    )

    train = DataLoader(
        train_ds,
        batch_size=cfg.param.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    test = DataLoader(
        val_ds,
        batch_size=cfg.param.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    net = convnextv2_huge()


if __name__ == "__main__":
    main()
