import hydra
from omegaconf import DictConfig

from config.hydra_compat import patch_hydra_argparse_for_python314
from runner import run_lightning_training

CONFIG_PATH = "configs"
CONFIG_NAME = "train"

patch_hydra_argparse_for_python314()


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    run_lightning_training(cfg)


if __name__ == "__main__":
    main()
