# train EM MMPS on CelebA

import inox
import inox.nn as nn
import jax
import numpy as np
import optax
import wandb

import regex as re

from datasets import Array3D, Features, concatenate_datasets, load_from_disk
from dawgz import job, schedule
from functools import partial
from tqdm import trange
from typing import *

# isort: split
from utils import *

CONFIG = {
    # Data
    'duplicate': 2,
    'corruption': 75,
    # Architecture
    'hid_channels': (128, 256, 384, 512),
    'hid_blocks': (3, 3, 3, 3),
    'kernel_size': (3, 3),
    'emb_features': 256,
    'heads': {3: 4},
    'dropout': 0.1,
    # Sampling
    'sampler': 'ddpm',
    'heuristic': None,
    'sde': {'a': 1e-3, 'b': 1e2},
    'discrete': 256, # default 256
    'maxiter': 1, # default 3
    # Training
    'epochs': 64,
    'batch_size': 256,
    'scheduler': 'constant',
    'lr_init': 1e-4,
    'lr_end': 1e-6,
    'lr_warmup': 0.0,
    'optimizer': 'adam',
    'weight_decay': None,
    'clip': 1.0,
    'ema_decay': 0.999,
    # MMPS
    'rank': 64, # 256, # default: 64
    'fit_size': 1<<14,# 1<<16, # 65536 default: 1<<14=16384
    'cov_y': 1e-2**2,
    'fit_moments_steps': 4,
    "fit_moments_maxiter": 5,
}
config = MyDict(CONFIG)

RUN_NAME = "75mask"
TEST_MODE = False

PATH = Path('[some path]/celeba_dir/em-mmps-test') if TEST_MODE \
  else Path('[some path]/celeba_dir/em-mmps' + f'_{RUN_NAME}')

DATASET_PATH = f'[some path]/celeba_64_mask{config.corruption}/' if TEST_MODE==False \
                else f'[some path]/celeba_64_mask{config.corruption}_test/'

PATH.mkdir(parents=True, exist_ok=True)
(Path(PATH) / 'checkpoints').mkdir(parents=True, exist_ok=True)

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

def train(runid: int, lap: int):
    run = wandb.init(
        project='celeba-EM-MMPS',
        id=runid,
        resume='allow',
        dir=PATH,
        name=f'em-mmps-celeba_{RUN_NAME}_[{lap}, )' + ('_test' if TEST_MODE else ''),
        config=CONFIG,
    )

    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    config = run.config

    # Sharding
    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    # RNG
    seed = hash((runpath, lap)) % 2**16
    rng = inox.random.PRNG(seed)

    # SDE
    sde = VESDE(**CONFIG.get('sde'))

    # Data
    dataset = load_from_disk(DATASET_PATH)
    dataset.set_format('numpy')

    def normalize_map(row):
        x, A = row['y'], row['A']
        x = (x * 4 / 256) - 2
        x = A * x # the corrupted pixels are set to zero instead of -2 (TODO: Does this really matter?)
        return {'y': x, 'A': A}

    dataset = dataset.map(
        normalize_map
    )

    trainset_yA = dataset
    # testset_yA = dataset['val']

    y_eval, A_eval = trainset_yA[:16]['y'], trainset_yA[:16]['A']
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)

    # Previous
    if lap > 0:
        previous = load_module(PATH / f'checkpoints/checkpoint_{lap - 1}.pkl')
    else:
        y_fit, A_fit = trainset_yA[:config.fit_size]['y'], trainset_yA[:config.fit_size]['A']
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)

        print("Starting fit_moments...")

        mu_x, cov_x = fit_moments(
            features=64 * 64 * 3,
            rank=CONFIG['rank'],
            shard=True,
            A=inox.Partial(measure, A_fit),
            y=flatten(y_fit),
            cov_y=CONFIG['cov_y'],
            sampler='ddim',
            sde=sde,
            steps=CONFIG['fit_moments_steps'], # TODO: needs tuning
            maxiter=CONFIG["fit_moments_maxiter"], # TODO: needs tuning
            key=rng.split(),
        )

        print("Finished fit_moments.")

        del y_fit, A_fit

        previous = GaussianDenoiser(mu_x, cov_x)

    ## Generate
    static, arrays = previous.partition()
    arrays = jax.device_put(arrays, replicated)
    previous = static(arrays)

    trainset = generate(
        model=previous,
        dataset=trainset_yA,
        rng=rng,
        batch_size=config.batch_size,
        shard=True,
        sampler=config.sampler,
        sde=sde,
        steps=config.discrete,
        maxiter=config.maxiter,
    )


    if lap == 0:
        sample_image = to_pil(trainset['x'][:16].reshape(4, 4, 64, 64, 3))
        wandb.log({"lap0fitmoments": wandb.Image(sample_image)})

    ## Moments
    x_fit = trainset[:config.fit_size]['x']
    x_fit = flatten(x_fit)
    x_fit = jax.device_put(x_fit, distributed)

    mu_x, cov_x = ppca(x_fit, rank=CONFIG['rank'], key=rng.split())

    del x_fit

    # Model
    if lap > 0:
        model = previous
    else:
        model = make_model(key=rng.split(), **CONFIG)

    model.mu_x = mu_x

    if config.heuristic == 'zeros':
        model.cov_x = jnp.zeros_like(mu_x)
    elif config.heuristic == 'ones':
        model.cov_x = jnp.ones_like(mu_x)
    elif config.heuristic == 'cov_t':
        model.cov_x = jnp.ones_like(mu_x) * 1e6
    elif config.heuristic == 'cov_x':
        model.cov_x = cov_x

    model.train(True)

    static, params, others = model.partition(nn.Parameter)

    # Objective
    objective = DenoiserLoss(sde=sde)

    # Optimizer
    steps = config.epochs * len(trainset) // config.batch_size
    optimizer = Adam(steps=steps, **config)
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
        # x = random_shake(x, keys[1], delta=4)
        x = random_hue(x, keys[1], delta=1e-2)
        x = random_saturation(x, keys[2], lower=0.95, upper=1.05)

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
        loader = trainset.shuffle(seed=seed + lap * config.epochs + epoch).iter(
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
                y=y_eval,
                A=A_eval,
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

    ## Checkpoint
    model = static(avrg, others)
    model.train(False)

    dump_module(model, PATH / f'checkpoints/checkpoint_{lap}.pkl')


if __name__ == '__main__':
    runid = wandb.util.generate_id()

    jobs = []
    start_lap = 0

    for file in (PATH / 'checkpoints').iterdir():
        match = re.fullmatch(r'checkpoint_(\d+)\.pkl', file.name)
        if match:
            lap = int(match.groups()[0])
            if lap >= start_lap:
                start_lap = lap + 1

    for lap in range(start_lap, 64):
        print(f'Lap {lap} started...')
        train(runid = runid, lap = lap)

