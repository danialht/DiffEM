from pathlib import Path

from datasets import load_from_disk, Array3D, Features, load_dataset

from .utils import *

from tqdm import tqdm
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from scipy.linalg import sqrtm

from priors.diffusion import ConditionalDenoiser

import logging

import torch
import torch.nn.functional as F
import torchvision

def to_uint8(images: np.ndarray) -> np.ndarray:
    """
    Convert images from [-2, 2] float32 → [0, 255] uint8
    """
    images = (images + 2.0) / 4.0          # → [0, 1]
    images = np.clip(images, 0.0, 1.0)
    images = (images * 255.0).astype(np.uint8)
    return images

def generate_dataset(model, dataset, rng, batch_size, conditional=True, **kwargs):
    """
    generates dataset by sampling using conditional or unconditional denoiser
    """
    def transform_conditional(batch):
        y_cond = batch['y']
        x = sample_conditional(model, y_cond, rng.split(), **kwargs)
        x = np.asarray(x)

        return {'x': x}

    def transform_unconditional(batch):
        y = batch['y']
        # x = sample(model, y_cond, rng.split(), **kwargs)

        x = sample_any(
            model=model,
            shape=flatten(y).shape,
            shard=True,
            A=None,
            y=None,
            cov_y=1e-3**2,
            key=rng.split(),
        )

        x = unflatten(x, 32, 32)

        x = np.asarray(x)

        return {'x': x}

    transform = transform_conditional if conditional else transform_unconditional

    types = {'x': Array3D(shape=(32, 32, 3), dtype='float32')}

    return dataset.map(
        transform,
        features=Features(types),
        remove_columns=[name for name in dataset.features.keys()],
        keep_in_memory=True,
        batched=True,
        batch_size=batch_size,
        drop_last_batch=True,
        desc="Generating Training Set",
    )

def evaluate(
    diffem_files_dir: Path,
    experiment: str,
    run_name: str,
    checkpoint_index: int,
    corruption_level: int,
    corruption_name: str,
    test: bool = False,
    conditional: bool = True,
) -> None:
    # load model
    checkpoint_path = diffem_files_dir / f'{experiment.dataset_name}/checkpoints/{run_name}'
    if conditional:
        checkpoint_path = checkpoint_path / ('checkpoint_'+str(checkpoint_index)+'.pkl')
    else:
        checkpoint_path = checkpoint_path / ('checkpoint_unconditional_'+str(checkpoint_index)+'.pkl')
    model = load_module(checkpoint_path)
    logging.info(f"Loaded model from {checkpoint_path}")

    # check if clean dataset for eval exists
    clean_dataset_path = diffem_files_dir / experiment.dataset_name / 'datasets' / 'clean'
    clean_dataset_eval_path = diffem_files_dir / experiment.dataset_name / 'datasets_eval' / 'clean'
    if not clean_dataset_eval_path.exists():
        clean_dataset_eval_path.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset('cifar10', cache_dir=clean_dataset_path)
        for i, img in enumerate(dataset['train']['img']):
            img.save(clean_dataset_eval_path / f'{i}.png')
    
    # load corrupted dataset
    corrupted_dataset_name = f"cifar-{corruption_name}-{corruption_level}" + ('_test' if test else '')
    corrupted_dataset_path = diffem_files_dir / experiment.dataset_name / 'datasets' / corrupted_dataset_name
    corrupted_dataset = load_from_disk(corrupted_dataset_path)
    corrupted_dataset.set_format('numpy')

    # Sharding
    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    # RNG
    seed = hash((run_name, checkpoint_index, 'unconditional')) % (1<<16)
    np_rng = np.random.default_rng(seed=seed)
    rng = inox.random.PRNG(seed)

    # SDE
    sde = VESDE(**experiment.sampler.sde)

    # Load datasets
    corrupted_dataset = corrupted_dataset['train']
    original_dataset_path = diffem_files_dir / experiment.dataset_name / 'datasets/clean'
    original_dataset = load_dataset('cifar10', cache_dir=original_dataset_path)
    original_dataset = original_dataset['train'].remove_columns(['label']).rename_column('img', 'x')

    # generate dataset
    generated_dataset = generate_dataset(
            model=model,
            dataset=corrupted_dataset,
            rng=rng,
            batch_size=256,
            shard=True,
            sampler=experiment.sampler.name,
            sde=sde,
            steps=experiment.sampler.discrete,
            maxiter=experiment.sampler.maxiter,
            conditional=conditional
        )

    gen_imgs = to_uint8(np.stack(generated_dataset['x']))
    real_imgs = np.stack([np.array(img) for img in original_dataset['x']]).astype(np.uint8)

    dataset_eval_dir = diffem_files_dir / experiment.dataset_name / 'datasets_eval'
    dataset_eval_dir = dataset_eval_dir / run_name / (("conditional_" if conditional else "unconditional_") + f"checkpoint_{checkpoint_index}" + ("_test" if test else ""))

    dataset_eval_dir.mkdir(parents=True, exist_ok=True)

    # save all the gen images as png in the eval dir
    for i, img in tqdm(enumerate(gen_imgs)):
        Image.fromarray(img).save(dataset_eval_dir / f'{i}.png')

    logging.info(f"Saved generated images to {dataset_eval_dir}")
