import hydra
from omegaconf import DictConfig
from pathlib import Path

from experiments.cifar.train_conditional import train as cifar_train_conditional
from experiments.celeba.train_conditional import train as celeba_train_conditional

from datetime import datetime

import logging

@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    # Root directory
    diffem_files_dir = Path(cfg.diffem_files_dir).expanduser().resolve()
    diffem_files_dir.mkdir(parents=True, exist_ok=True)

    # Common kwargs
    common_kwargs = dict(
        model=cfg.experiment.model,
        sampler=cfg.experiment.sampler,
        optimizer=cfg.experiment.optimizer,
        training=cfg.training,
        diffem_files_dir=diffem_files_dir,
        corruption_name=cfg.experiment.corruption,
        corruption_level=cfg.experiment.corruption_level,
        run_name=cfg.get("run_name", datetime.now().strftime("%m/%d/%Y_%H:%M:%S")),
        test=cfg.test
    )
    

    # Dispatch is now EXPLICIT in code structure
    # (you can choose this based on import or config if you want)
    if cfg.experiment.dataset_name == 'cifar':
        cifar_train_conditional(**common_kwargs)
    elif cfg.experiment.dataset_name == 'celeba':
        pass
        # celeba_train_conditional(**common_kwargs)
    else:
        logging.error(f"Unsupported experiment: {cfg.experiment}. Supported experiments are 'cifar', 'celeba'.")
        return


if __name__ == "__main__":
    main()
