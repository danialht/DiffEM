from utils import *
from tqdm import trange
import wandb
from datasets import Array3D, Features, load_from_disk
import pickle
from pathlib import Path

CONFIG = {
    # Data
    'corruption': 75,
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

CONFIG_LONG_TRAINED = {
    # Data
    'corruption': 75,
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
    'epochs': 512,
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

CONFIG = CONFIG_LONG_TRAINED

def train(dataset_name: str, dataset_path: Path, save_path: str, pretrained_model_path: str = None):
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

    PATH = dataset_path

    runid = wandb.util.generate_id()

    run = wandb.init(
            project='unconditional-cifar-mask',
            id=runid,
            resume='allow',
            dir=dataset_path,
            name=str(dataset_name),
            config=CONFIG,
        )

    config = run.config

    # Sharding
    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    # SDE
    sde = VESDE(**CONFIG.get('sde'))

    # RNG
    seed = hash('Random Hash') % 2**16
    rng = inox.random.PRNG(seed = seed)

    # Loading Data
    print('Loading dataset')
    dataset = pickle.load(open(str(dataset_path / dataset_name), 'rb'))
    print('Dataset Loaded')
    
    dataset.set_format('numpy')

    trainset = dataset

    # Model
    model = None
    if pretrained_model_path == None:
        model = make_model(key = rng.split(), **CONFIG)
    else:
        model = load_module(pretrained_model_path)

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

    # dump_module(model, runpath / f'checkpoint_{lap}.pkl')
    dump_module(model, save_path)
    run.finish()


def generate_data_conditional(checkpoint: str, path: Path, dataset_path: str, save_path: str):
    r"""
    Arguments:
        checkpoint      (str):  the name of the checkpoint file, e.g. 'checkpoint_5.pkl' 
        path            (Path): the path to the run directory (where all checkpoints are stored at) e.g. Path('/data/checkpoints/')
        dataset_path    (str):  path to dataset e.g. '/root/hf/cifar-mask-75'
        save_path       (str):  path to save the generated dataset e.g. '/root/generated_data/dataset.pkl'
    Return:
        None
    """


    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    class MyDict: # just so that we could use config.something instead of CONFIG['something']
        def __init__(self, dict):
            self.dict = dict
        def __getattr__(self, key):
            return self.dict[key]

    config = MyDict(CONFIG)

    # RNG
    seed = hash(checkpoint) % 2**16
    rng = inox.random.PRNG(seed)

    # SDE
    sde = VESDE(**CONFIG.get('sde'))

    # Data
    dataset = load_from_disk(dataset_path)
    dataset.set_format('numpy')

    trainset_yA = dataset['train']

    previous = load_module(path / checkpoint)

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
        )

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

    dump_module(trainset, save_path)


if __name__ == '__main__':
    for i in range(32):
        generate_data_conditional(
            checkpoint = f'checkpoint_{i}.pkl',
            dataset_path = f'[some path]/cifar_dir/hf/cifar-mask-90',
            path =  Path('[some path]/cifar_dir/conditional_90mask'),
            save_path = f'[some path]/cifar_dir/conditional_90mask/generated_data_checkpoint{i}.pkl'
            )
        
        train(
            dataset_name = f'generated_data_checkpoint{i}.pkl',
            dataset_path = Path('[some path]/cifar_dir/conditional_90mask/'),
            save_path = f'[some path]/cifar_dir/unconditional_90mask/checkpoint_{i}.pkl',
            )

