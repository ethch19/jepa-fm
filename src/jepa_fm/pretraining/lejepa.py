import torch
import hydra
import mlflow
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    mlflow.set_experiment("LeJEPA LC-FM")
    torch.manual_seed(123)


if __name__ == "__main__":
    main()
