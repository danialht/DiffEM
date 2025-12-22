from pathlib import Path

from .utils import *

def evaluate(
    diffem_files_dir: Path,
    experiment: str,
    metrics: List[str],
    run_name: str,
    checkpoint_index: int,
    corruption_level: int,
    corruption_name: str,
):
    for each_metric in metrics:
        if each_metric not in ['fid', 'is', 'fdinf', 'fddinov2']:
            raise ValueError(f"Unsupported metric: {each_metric}.")

    checkpoint_path = diffem_files_dir / experiment / run_name / ('checkpoint_' + str(checkpoint_index))

    model = load_module(checkpoint_path)

    dataset_path = diffem_files_dir / experiment / 'datasets' / f"{corruption_name}{corruption_level}"
    dataset = load_from_disk(dataset_path)

    breakpoint()