# Generating Dataset for DinoV2 evaluation

import inox
import io
import zipfile

from dawgz import job, schedule
from datasets import Array3D, Features, load_from_disk
from functools import partial
from torch import Tensor
from torch.utils import data
from torch_fidelity.fidelity import calculate_metrics
from tqdm import tqdm
from typing import *
import torchvision

# isort: split
from utils import *

CONFIG = {
    'corruption': 75,
    'maxiter': 1
}

def generate_unconditional(checkpoint: Path, num_images: int = 50000):
    # Sharding
    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # RNG
    seed = hash(checkpoint) % 2**16
    rng = inox.random.PRNG(seed)

    # Model
    model = load_module(checkpoint)

    static, arrays = model.partition()
    arrays = jax.device_put(arrays, replicated)
    model = static(arrays)

    # Generate
    images = []
    
    for _ in tqdm(range(0, num_images, 256), ncols=88):
        x = sample_any(
            model=model,
            shape=(256, 32 * 32 * 3),
            shard=True,
            sampler='ddim',
            steps=256,
            key=rng.split(),
        )
        x = unflatten(x, 32, 32)
        x = np.asarray(x)

        for img in map(to_pil, x):
            images.append(img)

    # Archive
    return images


def generate_conditional(lap: int, checkpoint_dir: Path, corrupted_dataset_path, num_images: int = 50000):

    runpath = checkpoint_dir

    with open(runpath / f'checkpoint_{lap}.pkl', 'rb') as f:
        previous = pickle.load(f)

    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    dataset = load_from_disk(corrupted_dataset_path)
    dataset.set_format('numpy')

    trainset_yA = dataset['train']
    testset_yA = dataset['test']

    y_eval = testset_yA[:16]['y']
    y_eval = jax.device_put((y_eval, ), distributed)

    # RNG
    checkpoint = runpath / f'checkpoint_{lap}.pkl'
    seed = hash(checkpoint) % 2**16
    rng = inox.random.PRNG(seed)
    if not checkpoint.exists():
        raise "Checkpoint doesn't exist"

    def generate(checkpoint: Path, seed: int = None):
        # Sharding
        jax.config.update('jax_threefry_partitionable', True)

        mesh = jax.sharding.Mesh(jax.devices(), 'i')
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # Model
        model = load_module(checkpoint)

        static, arrays = model.partition()
        arrays = jax.device_put(arrays, replicated)
        model = static(arrays)

        # Generate
        images = []

        for i in tqdm(range(0, num_images, 256), ncols=88):
            x = sample_conditional(
                model=model,
                y_cond = trainset_yA['y'][i: i + 256],
                key=rng.split(),
                shard=True,
                sampler='ddpm',
                steps=256,
                maxiter=CONFIG['maxiter'],
            )
            # x = unflatten(x, 32, 32)
            x = np.asarray(x)

            for img in map(to_pil, x):
                images.append(img)


        return images


    return generate(checkpoint, seed)
    


def moment_matching():
    def generate(model, dataset, rng, batch_size, **kwargs):
        def transform(batch):
            y, A = batch['y'], batch['A']
            x = sample(model, y, A, rng.split(), **kwargs)
            x = np.asarray(x)

            return {'x': x}

        types = {'x': Array3D(shape=(32, 32, 3), dtype='float32')}

        return dataset.map(
            transform,
            features=Features(types),
            remove_columns=['y', 'A'],
            keep_in_memory=True,
            batched=True,
            batch_size=batch_size,
            drop_last_batch=True,
        )

    CONFIG = {
        # Data
        'corruption': 90,
        # Architecture
        'hid_channels': (128, 256, 384),
        'hid_blocks': (5, 5, 5),
        'kernel_size': (3, 3),
        'emb_features': 256,
        'heads': {1: 4},
        'dropout': 0.1,
        # Sampling
        'sampler': 'ddpm',
        'sde': {'a': 1e-3, 'b': 1e2},
        'heuristic': None,
        'discrete': 256,
        'maxiter': 1,
        # Training
        'epochs': 256,
        'batch_size': 256,
        'scheduler': 'constant',
        'lr_init': 2e-4,
        'lr_end': 1e-6,
        'lr_warmup': 0.0,
        'optimizer': 'adam',
        'weight_decay': None,
        'clip': 1.0,
        'ema_decay': 0.9999,
    }

    PATH = Path('[some path]/cifar_backup_original_paper_75')

    # Sharding
    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    # RNG
    seed = hash('evaluation of EM + Moment Matching') % 2**16
    rng = inox.random.PRNG(seed)


    # SDE
    sde = VESDE(**CONFIG.get('sde'))


    # Data
    corruption = CONFIG['corruption']
    dataset = load_from_disk(f'[some path]/cifar_dir/hf/cifar-mask-{corruption}')
    dataset.set_format('numpy')

    trainset_yA = dataset['train']

    previous = load_module('[some path]/cifar_dir/emmps_90mask/checkpoints/checkpoint_31.pkl')

    static, arrays = previous.partition()
    arrays = jax.device_put(arrays, replicated)
    previous = static(arrays)

    trainset = generate(
        model=previous,
        dataset=trainset_yA,
        rng=rng,
        batch_size=CONFIG['batch_size'],
        shard=True,
        sampler=CONFIG['sampler'],
        sde=sde,
        steps=CONFIG['discrete'],
        maxiter=CONFIG['maxiter'],
    )

    counter = 0
    for img in map(to_pil, trainset['x']):
        img.save(f'[some path]/cifar_dir/datasets_for_eval/conditional/momentmatching_mask90/{counter}.png')
        counter += 1


def generate_blurry_dataset():
    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    corruption = CONFIG['corruption']
    dataset = load_from_disk(f'[some path]/cifar_dir/hf/cifar-mask-gaussian-blur-2')
    dataset.set_format('numpy')

    dataset = dataset['train']

    class Counter:
        def __init__(self):
            self.cnt = 0
        def inc(self):
            self.cnt += 1
        def get(self):
            return self.cnt


    counter = Counter()

    def save_transform(row):
        to_pil(row['y']).save(f'[some path]/cifar_dir/datasets_for_eval/blurry_gaussian_2/{counter.get()}.png')
        counter.inc()

    dataset.map(
        save_transform
    )
    



if __name__ == '__main__':
    
    conditional = False
    dir_category = 'conditional' if conditional else 'unconditional'
    
    indices = [31]
    
    for i in indices:
        # Has to set the arguments
        if conditional:
            images = generate_conditional(
                lap = i,
                corrupted_dataset_path = '[some path]/cifar_dir/hf/cifar-mask-90',
                checkpoint_dir = Path('[some path]/cifar_dir/conditional_90mask')
                )
        else:
            images = generate_unconditional(
                checkpoint = f'[some path]/cifar_dir/unconditional_90mask/checkpoint_{i}.pkl'
                )

        # Has to set the dir name
        dir_name = f'diffem_mask90/checkpoint_{i}'
        Path(f'[some path]/cifar_dir/datasets_for_eval/{dir_category}/{dir_name}').mkdir(parents=True, exist_ok=True)

        for idx, image in enumerate(images):
            image.save(f'[some path]/cifar_dir/datasets_for_eval/{dir_category}/{dir_name}/{idx}.png')
