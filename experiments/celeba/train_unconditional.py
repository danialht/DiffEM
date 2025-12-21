r"""Loads some conditional Model, Generates data with that and trains an unconditional model on that data"""

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset, Dataset, Array3D, Features, load_from_disk, concatenate_datasets
import inox
from tqdm import trange, tqdm

import random

from functools import partial
import wandb

import re
from .utils import *
import jax.numpy as jnp
from omegaconf import DictConfig
import numpy as np
from datasets import load_dataset, Dataset, Array3D, Features, load_from_disk
import inox
from tqdm import trange, tqdm

import wandb

import re

# CONFIG = {
#     # Data
#     # 'duplicate': 2,
#     'corruption': 75,
#     'img_shape': (64, 64, 3),
#     # Architecture
#     'hid_channels': (128, 256, 384, 512),
#     'hid_blocks': (3, 3, 3, 3),
#     'kernel_size': (3, 3),
#     'emb_features': 256,
#     'heads': {3: 4},
#     'dropout': 0.1,
#     # Sampling
#     'sampler': 'ddpm',
#     'heuristic': None,
#     'sde': {'a': 1e-3, 'b': 1e2},
#     'discrete': 64,
#     'maxiter': 3,
#     # Training
#     'epochs': 512,
#     'batch_size': 256,
#     'scheduler': 'constant',
#     'lr_init': 1e-4,
#     'lr_end': 1e-6,
#     'lr_warmup': 0.0,
#     'optimizer': 'adam',
#     'weight_decay': None,
#     'clip': 1.0,
#     'ema_decay': 0.999,
# }
# config = MyDict(CONFIG)

# TEST_MODE = False
# CHECKPOINT_TO_LOAD = Path('[some path]/celeba_dir/checkpoints_mask75/checkpoint_23.pkl')
# DATASET_PATH = f'[some path]/celeba_64_mask{config.corruption}' + ('_test' if TEST_MODE else '')
# RUN_NAME = f'mask75_checkpoint_23_unconditional'
# PATH = Path(f'[some path]/celeba_dir/unconditional-mask{config.corruption}')


def generate_conditional(model, config, dataset, rng, batch_size, sde, **kwargs):
    
    def transform(batch):
        y_cond = np.asarray(batch['y'])

        x = sample_conditional(
                model,
                y_cond,
                rng.split(),
                shard=True,
                sampler=config.sampler,
                steps=config.discrete,
                maxiter=config.maxiter
                )
        x = np.asarray(x)

        return {'x': x}

    types = {'x': Array3D(shape=(64, 64, 3), dtype='float32')}  

    return dataset.map(
        transform,
        features=Features(types),
        remove_columns=['y', 'A'],
        keep_in_memory=True,
        batched=True,
        batch_size=batch_size,
        drop_last_batch=True,
        desc="Sampling"
    )

def train_helper(
    run_name: str,
    diffem_files_dir: Path,
    train_config: dict,
    checkpoint_index: int,
    test: bool = False,
):
    runid = wandb.util.generate_id()
    checkpoint_dir = diffem_files_dir / 'celeba/checkpoints' / run_name

    run = wandb.init(
        project='celeba-unconditional',
        name=run_name+f'_checkpoint_{checkpoint_index}',
        id=runid,
        resume='allow',
        dir=checkpoint_dir,
        config=train_config,
    )
    config = run.config

    # Sharding
    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    # RNG
    seed = hash(run.name) % 2**16
    rng = inox.random.PRNG(seed)

    # SDE
    sde = VESDE(**train_config.get('sde'))

    # Load the corrupted dataset
    dataset_name = f'{config.corruption_name}{config.corruption_level}' + ('_test' if test else '')
    dataset_path = diffem_files_dir / 'celeba/datasets' / dataset_name
    dataset = load_from_disk(dataset_path)
    dataset.set_format('numpy')

    # Generate data from conditional model
    model = load_module(checkpoint_dir / f'checkpoint_{checkpoint_index}.pkl')
    trainset = generate_conditional(
        model=model,
        config=config,
        dataset=dataset,
        rng=rng,
        batch_size=config.batch_size,
        sde=sde,
    )

    # Train unconditional model on the generated data
    model = make_model(key=rng.split(), **train_config)
    model.train(True)
    static, params, others = model.partition(nn.Parameter)

    # Objective
    objective = DenoiserLoss(sde=sde)

    # Optimizer
    steps = config.epochs * len(dataset) // config.batch_size
    optimizer = Adam(
        steps=steps,
        scheduler = 'constant',
        lr_init = config.lr_init,
        lr_end = config.lr_end,
        lr_warmup = config.lr_warmup,
        weight_decay = config.weight_decay,
        clip = config.clip
        )
    opt_state = optimizer.init(params)

    # EMA
    ema = EMA(decay=config.ema_decay)
    avrg = params

    # Training
    avrg, params, others, opt_state = jax.device_put((avrg, params, others, opt_state), replicated)

    @jax.jit
    @jax.vmap
    def augment(x, key):
        keys = jax.random.split(key, 2)

        x = random_flip(x, keys[0], axis=-2)
        x = random_shake(x, keys[1], delta=4)

        return x

    @jax.jit
    def ell(params, others, x, key):
        keys = jax.random.split(key, 3)

        z = jax.random.normal(keys[0], shape=x.shape)
        t = jax.random.beta(keys[1], a=3, b=3, shape=x.shape[:1])

        return objective(static(params, others), x, z, t, key=keys[2])

    @jax.jit
    def sgd_step(avrg, params, others, opt_state, x, key):
        loss, grads = jax.value_and_grad(ell)(params, others, x, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        avrg = ema(avrg, params)

        return loss, avrg, params, opt_state

    for epoch in (bar := trange(config.epochs, ncols=88)):
        loader = trainset.shuffle(seed=seed + epoch).iter(
            batch_size=config.batch_size, drop_last_batch=True
        )

        losses = []

        for batch in prefetch(loader):
            x = batch['x']
            x = jax.device_put(x, distributed)
            x = augment(x, rng.split(len(x)))
            x = flatten(x)

            loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, x, key=rng.split())
            losses.append(loss)

        loss_train = np.stack(losses).mean()

        ## Eval
        if (epoch + 1) % 16 == 0:
            model = static(avrg, others)
            model.train(False)

            x = sample(
                model=model,
                y=None,
                A=None,
                key=rng.split(),
                shard=True,
                sampler=config.sampler,
                steps=config.discrete,
                maxiter=config.maxiter,
            )
            x = x.reshape(4, 4, 64, 64, 3)

            run.log({
                'loss': loss_train,
                'samples': wandb.Image(to_pil(x, zoom = 4)),
            })
        else:
            run.log({
                'loss': loss_train,
            })

    ## Checkpoint
    model = static(avrg, others)
    model.train(False)

    dump_module(model, checkpoint_dir / f'checkpoint_unconditional_{checkpoint_index}.pkl')


def train(
    model: DictConfig,
    sampler: DictConfig,
    optimizer: DictConfig,
    training: DictConfig,
    diffem_files_dir: Path,
    checkpoint_index: int,
    corruption_name: str,
    corruption_level: int,
    run_name: str,
    test: bool = True,
):

    config = {
        # Data
        'corruption_name': corruption_name,
        'corruption_level': corruption_level,
        # Architecture
        'hid_channels': model.hid_channels,
        'hid_blocks': model.hid_blocks,
        'kernel_size': model.kernel_size,
        'emb_features': model.emb_features,
        'heads': model.heads,
        'dropout': model.dropout,
        # Sampling
        'sampler': sampler.name,
        'sde': sampler.sde,
        'discrete': sampler.discrete,
        'maxiter': sampler.maxiter,
        # Training
        'epochs': training.epochs,
        'batch_size': training.batch_size,
        'scheduler': optimizer.scheduler,
        'lr_init': optimizer.lr_init,
        'lr_end': optimizer.lr_end,
        'lr_warmup': optimizer.lr_warmup,
        'optimizer': optimizer.name,
        'weight_decay': optimizer.weight_decay,
        'clip': training.clip,
        'ema_decay': training.ema_decay,
    }

    train_helper(
        run_name=run_name,
        diffem_files_dir=diffem_files_dir,
        train_config=config,
        checkpoint_index=checkpoint_index,
        test=test,
    )

    pass