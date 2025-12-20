#!/usr/bin/env python

from datasets import Array3D, Features, load_dataset, DatasetDict
from pathlib import Path

# isort: split
from .utils import *

def corrupt(
    cifar_dir_path: Path,
    maskprob: int,
    test: bool = False,
    seed: int = 123,
    ) -> None:

    rng = np.random.default_rng(seed)

    def transform(row):
        x = from_pil(row['img'])
        A = rng.uniform(size=(32, 32, 1)) > maskprob / 100
        y = rng.normal(loc=A * x, scale=1e-3)

        return {'A': A, 'y': y}

    types = {
        'A': Array3D(shape=(32, 32, 1), dtype='bool'),
        'y': Array3D(shape=(32, 32, 3), dtype='float32'),
    }

    dataset: DatasetDict = load_dataset('cifar10', cache_dir=cifar_dir_path / 'datasets/clean')
    
    if test:
        # only use the first 1024 samples
        for col in dataset.column_names:
            dataset[col] = dataset[col].select(range(1024))
        
    dataset = dataset.map(
        transform,
        features=Features(types),
        remove_columns=['img', 'label'],
        keep_in_memory=True,
        num_proc=4,
    )

    save_path = cifar_dir_path / (f'datasets/cifar-mask-{maskprob}' + ('_test' if test else ''))
    
    dataset.save_to_disk(save_path)

if __name__ == "__main__":
    pass