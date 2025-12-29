from dataclasses import dataclass
from typing import List

from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig

from experiments.cifar.evaluate import evaluate as cifar_evaluate
from experiments.celeba.evaluate import evaluate as celeba_evaluate

def catch_errors(cfg: DictConfig):
    if cfg.diffem_files_dir is None:
        raise ValueError("diffem_files_dir must be specified.")

    if cfg.run_name is None:
        raise ValueError("run_name must be specified.")

    if cfg.checkpoint_index is None:
        raise ValueError("run_name must be specified.")

    for metric in cfg.metrics:
        if metric not in ['fid', 'is', 'fdinf', 'fddinov2', 'prdc']:
            raise ValueError(f"Unsupported metric: {metric}.")

@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg: DictConfig):
    catch_errors(cfg)
    
    diffem_files_dir = Path(cfg.diffem_files_dir)

    if cfg.experiment.dataset_name == 'cifar':
        cifar_evaluate(
            diffem_files_dir=diffem_files_dir,
            experiment=cfg.experiment,
            metrics=cfg.metrics,
            run_name=cfg.run_name,
            checkpoint_index=cfg.checkpoint_index,
            corruption_level=cfg.experiment.corruption_level,
            corruption_name=cfg.experiment.corruption,
            test=cfg.test,
            conditional=cfg.conditional,
        )
    elif cfg.experiment.dataset_name == 'celeba':
        pass
    else:
        raise ValueError(f"Unsupported experiment: {cfg.experiment}.")
    

if __name__ == "__main__":
    main()
