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

from omegaconf import DictConfig
from datetime import datetime

from .utils import *
from jax.scipy.signal import convolve2d

IMG_SHAPE = (64, 64, 3)

def corrupt(rng, corruption, dataset: Dataset):

    def transform(row):
        x = np.asarray(row['x'])
        A = rng.binomial(n=1, p=1 - corruption / 100, size=IMG_SHAPE[:2] + (1,))
        y = np.array(A * x)
        return {'y': y}
    
    types = {
        'y': Array3D(shape=IMG_SHAPE, dtype='float32'),
    }

    return dataset.map(
        transform,
        remove_columns=dataset.column_names,
        features=Features(types),
        keep_in_memory=True,
        num_proc=1,
        desc=f"corruption the dataset with {corruption=}"
    )

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
        remove_columns=['y'],
        keep_in_memory=True,
        batched=True,
        batch_size=batch_size,
        drop_last_batch=True,
    )
    
def generate(model, dataset, rng, batch_size, **kwargs):
    def transform(batch):
        y, A = batch['y'], batch['A']
        x = sample(model, y, A, rng.split(), **kwargs)
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
    )

def train_helper(
    runid: int,
    lap: int,
    diffem_files_dir: Path,
    train_config: dict,
    run_name: str,
    test: bool = False,
    ):
    
    checkpoint_dir = diffem_files_dir / 'celeba/checkpoints' / run_name
    run = wandb.init(
            project='priors-celeba-mask-conditional',
            id=runid,
            resume='allow',
            dir=checkpoint_dir,
            config=train_config,
            name=f'{run_name}_lap{lap}',
        )

    config = run.config

    # Sharding
    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    # RNG
    seed = hash((run_name, lap)) % (1<<16)
    np_rng = np.random.default_rng(seed=seed)
    rng = inox.random.PRNG(seed)

    # SDE
    sde = VESDE(**train_config.get('sde'))

    dataset_path=diffem_files_dir / 'celeba/datasets'
    dataset_path = dataset_path / (f'{config.corruption_name}{config.corruption}' + ('_test' if test else ''))
    dataset = load_from_disk(dataset_path, keep_in_memory=True)
    dataset.set_format('numpy')

    if lap == 0:
        # MMPS initialization
        y_fit, A_fit = dataset[:1<<17]['y'], dataset[:1<<17]['A']
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)

        mu_x, cov_x = fit_moments(
            features=64*64*3,
            rank=1024,
            shard=True,
            A=inox.Partial(measure, A_fit),
            y=flatten(y_fit),
            cov_y=1e-2**2,
            sampler='ddim',
            sde=sde,
            steps=4,
            maxiter=5,
            key=rng.split(),
        )

        del y_fit, A_fit

        previous = GaussianDenoiser(mu_x, cov_x)

        trainset = generate(
            model=previous,
            dataset=dataset,
            rng=rng,
            batch_size=config.batch_size,
            shard=True,
            sampler=config.sampler,
            sde=sde,
            steps=config.discrete,
            maxiter=config.maxiter,
        )

        model = make_model_conditional(key = rng.split(), **train_config)
        wandb.log({'mmps_result': wandb.Image(to_pil(trainset['x'][0]))})
    else:
        model = load_module(checkpoint_dir / f'checkpoint_{lap - 1}.pkl')
        print(f'Loaded checkpoint_{lap - 1}.pkl')
        trainset = generate_conditional(model, config, dataset.remove_columns('A'), rng, config.batch_size, sde)


    trainset_corrupted = corrupt(np_rng, config.corruption, trainset)
    trainset_corrupted.set_format(type='numpy', columns=['y'])

    evalset = np.array(dataset[:16]['y'])

    # Sharding
    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    model.train(True)

    static, params, others = model.partition(nn.Parameter)

    # Objective
    objective = ConditionalDenoiserLoss(sde=sde)

    # Optimizer
    steps = config.epochs * len(dataset) // config.batch_size
    optimizer = Adam(
        steps=steps,
        scheduler = 'constant',
        lr_init = config.lr_init,
        lr_end = config.lr_end,
        lr_warmup = config.lr_warmup,
        weight_decay = config.weight_decay,
        clip = config.clip,
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
        keys = jax.random.split(key, 3)
        x = random_flip(x, keys[0], axis=-2)
        x = random_shake(x, keys[1], delta=4)

        return x

    @jax.jit
    def ell(params, others, x, y_cond, key):
        keys = jax.random.split(key, 3)

        z = jax.random.normal(keys[0], shape=x.shape)
        t = jax.random.beta(keys[1], a=3, b=3, shape=x.shape[:1])

        return objective(static(params, others), x, z, t, y_cond, key=keys[2])

    @jax.jit
    def sgd_step(avrg, params, others, opt_state, x, y_cond, key):
        loss, grads = jax.value_and_grad(ell)(params, others, x, y_cond, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        avrg = ema(avrg, params)

        return loss, avrg, params, opt_state


    # num_epochs = config.epochs if lap > 0 else config.epochs // 4

    for epoch in (bar := trange(config.epochs, ncols=88)):
        loader = trainset.shuffle(seed=seed + lap * config.epochs + epoch).iter(
            batch_size=config.batch_size, drop_last_batch=True
        )
        loader_corrupted = trainset_corrupted.shuffle(seed=seed + lap * config.epochs + epoch).iter(
            batch_size=config.batch_size, drop_last_batch=True
        )
        losses = []

        for batch_x, batch_y in zip(prefetch(loader), prefetch(loader_corrupted)):
            x = np.array(batch_x['x'])
            y = batch_y['y']
            aug_key = rng.split(len(x))

            x = jax.device_put(x, distributed)
            x = augment(x, aug_key)
            x = flatten(x)
                
            y = jax.device_put(y, distributed)
            y = augment(y, aug_key)
            y = flatten(y)


            loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, x, y, key=rng.split())
            losses.append(loss)

        loss_train = np.stack(losses).mean()

        bar.set_postfix(loss=loss_train)

        ## Eval
        if (epoch + 1) % 16 == 0:
            model = static(avrg, others)
            model.train(False)

            x = sample_conditional(
                model=model,
                y_cond=evalset,
                key=rng.split(),
                shard=True,
                sampler=config.sampler,
                steps=config.discrete,
                maxiter=config.maxiter,
            )
            
            x = x.reshape(4, 4, 64, 64, 3)

            run.log({
                'loss': loss_train,
                'samples': wandb.Image(to_pil(x)),
            })
        else:
            run.log({
                'loss': loss_train,
            })

    dump_module(model, checkpoint_dir / f'checkpoint_{lap}.pkl')
    print(f'Saved checkpoint_{lap}.pkl')


def train(
    model: DictConfig,
    sampler: DictConfig,
    optimizer: DictConfig,
    training: DictConfig,
    diffem_files_dir: Path,
    run_name: str|None,
    corruption_name: str,
    corruption_level: float,
    test: bool = False,
):

    num_laps = training.num_laps
    last_lap = training.last_lap

    if(run_name is None):
        run_name = datetime.now().strftime("%m_%d_%Y_%H:%M:%S.%f")[:-3]
    
    if(num_laps <= 0 and last_lap <= -1):
        raise ValueError("Either laps (number of laps to train) \
        or last_lap (the last lap-number to stop at) must be specified.")

    if(num_laps > 0 and last_lap > -1):
        raise ValueError("Only one of laps (number of laps to train) \
        or last_lap (the last lap-number to stop at) can be specified.")


    if not diffem_files_dir.exists():
        raise ValueError(f"The specified diffem_files_dir does not exist: {diffem_files_dir.resolve()}")

    
    checkpoint_dir = diffem_files_dir / 'celeba' / 'checkpoints' / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_lap = 0

    # Find the maximum checkpoint index and set teh start_lap
    for file in checkpoint_dir.iterdir():
        if file.name.startswith('checkpoint_') and file.suffix == '.pkl':
            lap_num = int(file.name[len('checkpoint_'):-len('.pkl')])
            if lap_num >= start_lap:
                start_lap = lap_num + 1
    
    last_lap = start_lap + num_laps - 1 if num_laps > 0 else last_lap
    
    
    runid = wandb.util.generate_id()
    
    
    config = {
        # Data
        'corruption_name': corruption_name,
        'corruption': corruption_level,
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

    for lap in range(start_lap, last_lap+1):
        train_helper(
            runid = runid,
            lap = lap,
            train_config=config,
            diffem_files_dir=diffem_files_dir,
            run_name=run_name,
            test=test,
        )


if __name__ == '__main__':
    pass
