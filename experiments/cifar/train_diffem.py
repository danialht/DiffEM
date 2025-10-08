import time
import inox
import inox.nn as nn
import jax
import numpy as np
import optax
import wandb

from datasets import Array3D, Features, load_from_disk
from dawgz import job, schedule
from functools import partial
from tqdm import trange
from typing import *

# isort: split
from utils import *

# Smaller Config
CONFIG_ORIGINAL = {
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
    'epochs': 512, # 256,
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

# Larger Config
CONFIG_LARGE = {
    # Data
    'corruption': 75,
    # Architecture
    'hid_channels': (256, 384, 512), # (128, 256, 384),
    'hid_blocks': (6, 6, 6), # (5, 5, 5),
    'kernel_size': (3, 3),
    'emb_features': 512, # 256,
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

CONFIG = CONFIG_ORIGINAL
DATASET_PATH = f'[somepath]/cifar_dir/hf/cifar-mask-{CONFIG["corruption"]}'
RUN_NAME = 'conditional_90mask'
PATH = Path(f'[somepath]/cifar_dir/{RUN_NAME}')
PATH.mkdir(parents=True, exist_ok=True)

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

def generate_conditional(model, dataset, rng, batch_size, **kwargs):
    def transform(batch):
        y_cond, A = batch['y'], batch['A']
        x = sample_conditional(model, y_cond, rng.split(), **kwargs)
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

def corrupt(corruption, dataset, rng):

    def transform(row):
        x = row['x']
        A = rng.uniform(shape=(32, 32, 1)) > corruption / 100
        y = 1e-3 * rng.normal(shape = A.shape) + A * x
        y = np.array(y)
        return {'y': y}

    types = {
        'y': Array3D(shape=(32, 32, 3), dtype='float32'),
    }

    return dataset.map(
        transform,
        features=Features(types),
        remove_columns=['x'],
        keep_in_memory=True,
        num_proc=1,
    )


def train(runid: int, lap: int, dir_name: str):
    print('Beginning lap', lap)
    begin_t = time.time()
    run = wandb.init(
        project='priors-cifar-mask-conditional',
        id=runid,
        resume='allow',
        name=f'cifar-diffEM_[{lap}, )' + f'_{RUN_NAME}',
        dir=PATH,
        config=CONFIG,
    )

    runpath = PATH
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

    trainset_yA = dataset['train']
    testset_yA = dataset['test']

    y_eval, A_eval = testset_yA[:16]['y'], testset_yA[:16]['A']
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)

    # Previous
    if lap > 0:
        # raise Exception("Not Implemented")
        previous = load_module(runpath / f'checkpoint_{lap - 1}.pkl')
        print('Loaded previous model from' + str(runpath / f'checkpoint_{lap - 1}.pkl'))
    else:
        y_fit, A_fit = trainset_yA[:16384]['y'], trainset_yA[:16384]['A']
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)

        mu_x, cov_x = fit_moments(
            features=32 * 32 * 3,
            rank=64,
            shard=True,
            A=inox.Partial(measure, A_fit),
            y=flatten(y_fit),
            cov_y=1e-3**2,
            sampler='ddim',
            sde=sde,
            steps=256,
            maxiter=None,
            key=rng.split(),
        )

        del y_fit, A_fit

        previous = GaussianDenoiser(mu_x, cov_x)

    ## Generating
    static, arrays = previous.partition()
    arrays = jax.device_put(arrays, replicated)
    previous = static(arrays)

    ## Generating Pi
    if lap == 0:
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

        testset = generate(
            model=previous,
            dataset=testset_yA,
            rng=rng,
            batch_size=config.batch_size,
            shard=True,
            sampler=config.sampler,
            sde=sde,
            steps=config.discrete,
            maxiter=config.maxiter,
        )
    else:
        trainset = generate_conditional(
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
    
        testset = generate_conditional(
            model=previous,
            dataset=testset_yA,
            rng=rng,
            batch_size=config.batch_size,
            shard=True,
            sampler=config.sampler,
            sde=sde,
            steps=config.discrete,
            maxiter=config.maxiter,
        )
    
    testset_corrupted = corrupt(config.corruption, testset, rng)

    ## Moments
    x_fit = trainset[:16384]['x']
    x_fit = flatten(x_fit)

    mu_x, cov_x = ppca(x_fit, rank=64, key=rng.split())

    del x_fit

    # Model
    if lap > 0:
        model = previous
    else:
        model = make_model_conditional(key=rng.split(), **CONFIG)

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
    objective = ConditionalDenoiserLoss(sde=sde)

    # Optimizer
    steps = config.epochs * len(trainset_yA) // config.batch_size
    optimizer = Adam(
        steps=steps,
        scheduler = config.scheduler,
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
        keys = jax.random.split(key, 3)
    
        x = random_flip(x, keys[0], axis=-2)
        x = random_hue(x, keys[1], delta=1e-2)
        x = random_saturation(x, keys[2], lower=0.95, upper=1.05)
    
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

    
    pi0 = trainset # This is the latent data generated by the previous model
    y0 = corrupt(config.corruption, pi0, rng)

    for epoch in (bar := trange(config.epochs, ncols=88)):
        loader_pi0 = pi0.shuffle(seed=seed + epoch * config.epochs + epoch).iter(
            batch_size=config.batch_size, drop_last_batch=True
        )

        loader_y0 = y0.shuffle(seed=seed + epoch * config.epochs + epoch).iter(
            batch_size=config.batch_size, drop_last_batch=True
        )

        losses = []

        for batch_pi0, batch_y0 in zip(prefetch(loader_pi0), prefetch(loader_y0)):
            x = batch_pi0['x']
            x = jax.device_put(x, distributed)
            aug_key = rng.split(len(x))
            x = augment(x, aug_key)
            x = flatten(x)

            y_cond = batch_y0['y']
            y_cond = jax.device_put(y_cond, distributed)
            y_cond = augment(y_cond, aug_key)
            y_cond = flatten(y_cond)

            loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, x, y_cond, key=rng.split())
            losses.append(loss)

        loss_train = np.stack(losses).mean()

        ## Validation
        loader_testset = testset.iter(batch_size=config.batch_size, drop_last_batch=True)
        loader_testset_corrupted = testset_corrupted.iter(batch_size=config.batch_size, drop_last_batch=True)
        losses = []
        

        for batch_testset_corrupted, batch_testset in zip(prefetch(loader_testset_corrupted), prefetch(loader_testset)):
            x = batch_testset['x']
            x = jax.device_put(x, distributed)
            x = flatten(x)

            y_cond = batch_testset_corrupted['y']
            y_cond = jax.device_put(y_cond, distributed)
            y_cond = flatten(y_cond)

            # loss = ell(avrg, others, x, y_cond, key=rng.split())
            loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, x, y_cond, key=rng.split())
            losses.append(loss)

        loss_val = np.stack(losses).mean()

        bar.set_postfix(loss=loss_train, loss_val=loss_val)

        ## Eval
        if (epoch + 1) % 16 == 0:
            model = static(avrg, others)
            model.train(False)

            x = sample_conditional(
                model=model,
                y_cond = y_eval,
                key=rng.split(),
                shard=True,
                sampler=config.sampler,
                steps=config.discrete,
                maxiter=config.maxiter,
            )
            x = x.reshape(4, 4, 32, 32, 3)

            run.log({
                'loss': loss_train,
                'loss_val': loss_val,
                'samples': wandb.Image(to_pil(x, zoom=4)),
            })
        else:
            run.log({
                'loss': loss_train,
                'loss_val': loss_val,
            })

    ## Checkpoint
    model = static(avrg, others)
    model.train(False)

    dump_module(model, runpath / f'checkpoint_{lap}.pkl')
    print('Finished Lap', lap, 'in', (time.time() - begin_t) / 60, 'minutes')


if __name__ == '__main__':
    runid = wandb.util.generate_id()
    print('itnog')

    start_lap = 20

    for lap in range(start_lap, 32):
        train(runid, lap, 'actual_run_1')
        print(f'LAP {lap} FINISHED!')
