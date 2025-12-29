from .utils import *
from tqdm import trange
import wandb
from datasets import Array3D, Features, load_from_disk
import pickle
from pathlib import Path
import logging
from omegaconf import DictConfig

def train_helper(
    run_name: str,
    diffem_files_dir: Path,
    train_config: DictConfig,
    checkpoint_index: int,
    test: bool = True
    ):
    r"""
    Arguments:
        dataset_name            (str):  name of the dataset saved using the generate.
        dataset_path            (Path): path to the parent directory where the dataset is
        save_path               (str):  full path to save the model e.g. '/root/model.pkl'
        pretrained_model_path   (str):  path to an already trained model, if given its
                                        training will be continued.
    Return:
        None
    """

    runid = wandb.util.generate_id()
    checkpoint_dir = diffem_files_dir / 'cifar/checkpoints' / run_name

    run = wandb.init(
            project='unconditional-cifar-mask',
            id=runid,
            resume='allow',
            dir=checkpoint_dir,
            name=run_name+f"_checkpoint_{checkpoint_index}",
            config=train_config,
        )

    config = run.config

    # Sharding
    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    # SDE
    sde = VESDE(**train_config.get('sde'))

    # RNG
    seed = hash('Random Hash') % 2**16
    rng = inox.random.PRNG(seed = seed)

    # Loading the corrupted Dataset
    logging.info('Loading dataset')
    dataset_path = diffem_files_dir / 'cifar/datasets'
    dataset_path = dataset_path / (f'cifar-{config.corruption_name}-{config.corruption_level}' + ('_test' if test else ''))
    dataset = load_from_disk(dataset_path)
    dataset.set_format('numpy')

    trainset_yA = dataset['train']

    conditional_model = load_module(checkpoint_dir / f'checkpoint_{checkpoint_index}.pkl')

    def generate_conditional(model, dataset, rng, batch_size, **kwargs):
        def transform(batch):
            y_cond = batch['y']
            x = sample_conditional(model, y_cond, rng.split(), **kwargs)
            x = np.asarray(x)

            return {'x': x}

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

    trainset = generate_conditional(
            model=conditional_model,
            dataset=trainset_yA,
            rng=rng,
            batch_size=config.batch_size,
            shard=True,
            sampler=config.sampler,
            sde=sde,
            steps=config.discrete,
            maxiter=config.maxiter,
        )

    # Model
    model = make_model(key = rng.split(), **train_config)

    model.train(True)

    static, params, others = model.partition(nn.Parameter)

    objective = DenoiserLoss(sde=sde)

    print(trainset.shape)

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
        keys = jax.random.split(key, 3)

        x = random_flip(x, keys[0], axis=-2)
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

            x = sample_any(
                model=model,
                shape=(16, 32 * 32 * 3),
                shard=True,
                sampler='ddim',
                steps=256,
                key=rng.split(),
            )
            x = unflatten(x, 32, 32)
            x = np.asarray(x)

            run.log({
                'loss': loss_train,
                'samples': wandb.Image(to_pil(x, zoom=4)),
            })
        else:
            run.log({
                'loss': loss_train,
            })

    ## Checkpoint
    model = static(avrg, others)
    model.train(False)

    save_path = checkpoint_dir / f'checkpoint_unconditional_{checkpoint_index}.pkl'
    dump_module(model, save_path)
    run.finish()

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
    test: bool = False,
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
