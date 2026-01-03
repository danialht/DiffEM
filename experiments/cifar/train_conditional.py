import time
import inox
import inox.nn as nn
import jax
import numpy as np
import optax
import wandb

from omegaconf import DictConfig

from datasets import Array3D, Features, load_from_disk
# from dawgz import job, schedule
from functools import partial
from tqdm import trange

from datetime import datetime

from typing import *

# isort: split
from .utils import *

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
        A = rng.uniform(size=(32, 32, 1)) > corruption / 100
        y = 1e-3 * rng.normal(size = A.shape) + A * x
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


def train_helper(
    runid: int,
    lap: int,
    diffem_files_dir: Path,
    train_config: dict,
    run_name: str,
    test: bool = False
) -> None:
    print('Beginning lap', lap)
    begin_t = time.time()

    checkpoint_dir = diffem_files_dir/'cifar'/'checkpoints'/run_name

    run = wandb.init(
        project='priors-cifar-mask-conditional',
        id=runid,
        resume='allow',
        name=f'cifar-diffEM_[{lap}, )' + f'_{run_name}',
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
    seed = hash((run_name, lap)) % 2**16
    rng = inox.random.PRNG(seed)
    np_rng = np.random.default_rng(seed)

    # SDE
    sde = VESDE(**train_config.get('sde'))

    # Data
    dataset = load_from_disk(diffem_files_dir / 'cifar' / 'datasets' / f'cifar-mask-{config.corruption}')
    
    # on test mode only use the first 1024 data points
    if test:
        for col in dataset.column_names:
            dataset[col] = dataset[col].select(range(1024))

    dataset.set_format('numpy')

    trainset_yA = dataset['train']
    testset_yA = dataset['test']

    y_eval, A_eval = testset_yA[:16]['y'], testset_yA[:16]['A']
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)

    # Previous
    if lap > 0:
        # raise Exception("Not Implemented")
        previous = load_module(checkpoint_dir / f'checkpoint_{lap - 1}.pkl')
        print('Loaded previous model from' + str(checkpoint_dir / f'checkpoint_{lap - 1}.pkl'))
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
    
    testset_corrupted = corrupt(config.corruption, testset, np_rng)

    ## Moments
    x_fit = trainset[:16384]['x']
    x_fit = flatten(x_fit)

    mu_x, cov_x = ppca(x_fit, rank=64, key=rng.split())

    del x_fit

    # Model
    if lap > 0:
        model = previous
    else:
        model = make_model_conditional(key=rng.split(), **train_config)

    model.mu_x = mu_x

    # This is by default not set in this code.
    if config.get('heuristic', None) is not None:
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
    y0 = corrupt(config.corruption, pi0, np_rng)

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

    dump_module(model, checkpoint_dir / f'checkpoint_{lap}.pkl')
    print('Finished Lap', lap, 'in', (time.time() - begin_t) / 60, 'minutes')




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
    """
    Trains DiffEM on Cifar-10 dataset with random masking corruption
    Args:
        model: model configs, see /conf/

    """
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

    
    checkpoint_dir = diffem_files_dir / 'cifar' / 'checkpoints' / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_lap = 0

    # Find the maximum checkpoint index and set teh start_lap
    for file in checkpoint_dir.iterdir():
        if (file.name.startswith('checkpoint_')
            and file.suffix == '.pkl'
            and (not file.name.startswith('checkpoint_unconditional'))
            ):
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
