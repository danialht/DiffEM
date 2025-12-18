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
from jax.scipy.signal import convolve2d

CONFIG = {
    # Data
    # 'duplicate': 2,
    'corruption': 50,
    'img_shape': (64, 64, 3),
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
    'discrete': 64,
    'maxiter': 3,
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
}
config = MyDict(CONFIG)

TEST_MODE = False
RUN_NAME = "mask50"

CHECKPOINT_PATH = Path('[some path]/celeba_dir/checkpoints_test') if TEST_MODE \
else Path('[some path]/celeba_dir/checkpoints_' + RUN_NAME)

DATASET_PATH = Path(f'[some path]/celeba_64_mask{config.corruption}_test/') if TEST_MODE \
else Path(f'[some path]/celeba_64_mask{config.corruption}/')

DATASET = None

def corrupt(rng, corruption, dataset: Dataset):

    def transform(row):
        x = np.asarray(row['x'])
        A = rng.bernoulli(p = 1 - corruption / 100, shape = config.img_shape[:2] + (1, ))
        y = np.array(A * x)
        return {'y': y}
    
    types = {
        'y': Array3D(shape=config.img_shape, dtype='float32'),
    }

    return dataset.map(
        transform,
        remove_columns=dataset.column_names,
        features=Features(types),
        keep_in_memory=True,
        num_proc=1
    )

def generate_conditional(model, dataset, rng, batch_size, sde, **kwargs):
    
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

@jax.vmap
def blur_initialization_vfun(x, A, kernel):
    """
    input is an image in range [-2, 2] and a mask A
    """
    x += 2
    W = (convolve2d(A.squeeze(), kernel, mode='same', boundary='fill')**-1)[..., None]
    W = jnp.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    x_blurry = jnp.stack(
        [convolve2d(x[:,:,i], kernel, mode='same', boundary='fill') for i in range(3)],
        axis=-1
    ) * W
    x = A * x + (1 - A) * x_blurry
    x -= 2
    return x

def blur_initialization(row: dict, kernel):
    """
    input is an image in range [-2, 2] and a mask A
    """
    x, A = row['x'], row['A']
    x += 2
    W = (convolve2d(A.squeeze(), kernel, mode='same', boundary='fill')**-1)[..., None]
    W = jnp.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    x_blurry = np.stack(
        [convolve2d(x[:,:,i], kernel, mode='same', boundary='fill') for i in range(3)],
        axis=-1
    ) * W
    x = A * x + (1 - A) * x_blurry
    x -= 2
    return {'x': x}

def train(lap, runid):
    run = wandb.init(
            project='priors-celeba-mask-conditional',
            id=runid,
            resume='allow',
            dir=CHECKPOINT_PATH,
            config=CONFIG,
            name=f'celeba_itnog_lap[{start_lap},)' + ('_test' if TEST_MODE else '_' + RUN_NAME)
        )

    # Sharding
    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    # RNG
    seed = hash(('celebA', lap)) % (1<<16)
    rng = inox.random.PRNG(seed)

    # SDE
    sde = VESDE(**CONFIG.get('sde'))

    dataset = DATASET
    breakpoint()

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

        model = make_model_conditional(key = rng.split(), **CONFIG)
        wandb.log({'mmps_result': wandb.Image(to_pil(trainset['x'][0]))})
    else:
        model = load_module(CHECKPOINT_PATH / f'checkpoint_{lap - 1}.pkl')
        print(f'Loaded checkpoint_{lap - 1}.pkl')
        trainset = generate_conditional(model, dataset.remove_columns('A'), rng, config.batch_size, sde)


    trainset_corrupted = corrupt(rng, config.corruption, trainset)
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
        print(f'epoch {epoch}')
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

    dump_module(model, CHECKPOINT_PATH / f'checkpoint_{lap}.pkl')
    print(f'Saved checkpoint_{lap}.pkl')

def init_dataset():
    """
    Maps a dataset to [-2, 2] and sets the 
    dead pixels to 0 (not -2).
    """
    global DATASET
    if TEST_MODE:
        dataset = load_from_disk(f'[some path]/celeba_64_mask{config.corruption}_test/', keep_in_memory=True)
    else:
        dataset = load_from_disk(f'[some path]/celeba_64_mask{config.corruption}/', keep_in_memory=True)
    dataset.set_format('numpy')
    def normalize(row):
        return {'y': row['y'] * 4 / 256 - 2, 'A': row['A']}
    DATASET = dataset.map(normalize, desc="Normalizing dataset") # TODO: remove the 1<<10

if __name__ == "__main__":
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    init_dataset()
    # Automatically finding the start_lap by looping over the files in directory
    checkpoint_name_regex = r'checkpoint_(\d+).pkl'
    max_checkpoint = -1
    for child in Path(CHECKPOINT_PATH).iterdir():
        if not child.is_file():
            continue
        if not re.fullmatch(checkpoint_name_regex, child.name):
            continue
        max_checkpoint = max(max_checkpoint, int(re.fullmatch(checkpoint_name_regex, child.name).group(1)))
        
    start_lap = max_checkpoint + 1

    runid = wandb.util.generate_id()

    for lap in range(start_lap, 64):
        train(lap, runid)
