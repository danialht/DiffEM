import jax
import jax.numpy as jnp
import numpy as npdat
from datasets import load_dataset, Dataset, Array3D, Features, load_from_disk, enable_progress_bars
import inox
from tqdm import trange, tqdm

from .utils import *

IMG_SHAPE = (64, 64, 3)

def corrupt(rng, corruption, dataset: Dataset):
    def transform(row):
        x = from_pil(row['x'])
        A = rng.binomial(n=1, p=1 - corruption / 100, size=IMG_SHAPE[:2] + (1, )).astype(bool)
        y = np.array(A * x)
        return {'y': y, 'A': A}
    
    types = {
        'y': Array3D(shape=IMG_SHAPE, dtype='float32'),
        'A': Array3D(shape=IMG_SHAPE[:2] + (1, ), dtype='bool')
    }

    dataset = dataset.map(
        transform,
        remove_columns=dataset.column_names,
        features=Features(types),
        keep_in_memory=True,
        num_proc=4
    )

    return dataset

def corrupt_dataset(
    dataset_path: str,
    save_path: str,
    maskprob: int,
    seed: int = 123,
    test: bool = False,
):
    """
    Corrupt the CelebA dataset by applying random masks and saving it.
    arg(s):
    dataset_path: the path to the celeba dataset
                center cropped in 64x64 image files
    """
    
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset(dataset_path, keep_in_memory=True)

    if test:
        for col in dataset.column_names:
            dataset[col] = dataset[col].select(range(1024))

    trainset = dataset['train']
    trainset = trainset.rename_column('image', 'x')
    # rng = inox.random.PRNG(seed)
    rng = np.random.default_rng(seed)

    trainset_corrupted = corrupt(rng, maskprob, trainset)

    trainset_corrupted.save_to_disk(save_path)
