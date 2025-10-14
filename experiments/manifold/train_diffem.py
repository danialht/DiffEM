import inox
import inox.nn as nn
import jax
import numpy as np
import optax
import wandb
import click

import datetime

from dawgz import job, schedule
from tqdm import tqdm, trange
from typing import *

# isort: split
from utils import *

CONFIG = {
    # Data
    'seed': 0,
    'samples': 65536,
    'features': 5,
    'observe': 2,
    'noise': 1e-2,
    # Architecture
    'features_latent': 5,
    'features_cond': 2 + 5 * 2,
    'hid_features': (256, 256, 256),
    'emb_features': 64,
    'normalize': True,
    # Sampling
    'sampler': 'pc',
    'heuristic': 'cov_x',
    'sde': {'a': 1e-3, 'b': 1e1},
    'discrete': 4096,
    'maxiter': None,
    # Training
    'laps': 64,
    'epochs': 65536,
    'batch_size': 1024,
    'scheduler': 'linear',
    'lr_init': 1e-3,
    'lr_end': 1e-6,
    'lr_warmup': 0.0,
    'optimizer': 'adam',
    'weight_decay': None,
    'clip': 1.0,
}

CONFIG_TINY = {
    # Data
    'seed': 0,
    'samples': 512,
    'features': 5,
    'observe': 2,
    'noise': 1e-2,
    # Architecture
    'features_latent': 5,
    'features_cond': 2 + 5 * 2,
    'hid_features': (32, 32, 32),
    'emb_features': 16,
    'normalize': True,
    # Sampling
    'sampler': 'pc',
    'heuristic': 'cov_x',
    'sde': {'a': 1e-3, 'b': 1e1},
    'discrete': 4096,
    'maxiter': None,
    # Training
    'laps': 64,
    'epochs': 100,
    'batch_size': 1024,
    'scheduler': 'linear',
    'lr_init': 1e-3,
    'lr_end': 1e-6,
    'lr_warmup': 0.0,
    'optimizer': 'adam',
    'weight_decay': None,
    'clip': 1.0,
}

CONFIG_LARGE = {
    # Data
    'seed': 0,
    'samples': 65536,
    'features': 5,
    'observe': 2,
    'noise': 1e-2,
    # Architecture
    'features_latent': 5,
    'features_cond': 2 + 5 * 2,
    'hid_features': (512, 512, 512),
    'emb_features': 128,
    'normalize': True,
    # Sampling
    'sampler': 'ddpm',
    'heuristic': 'cov_x',
    'sde': {'a': 1e-3, 'b': 1e1},
    'discrete': 4096,
    'maxiter': None,
    # Training
    'laps': 64,
    'epochs': 65536,
    'batch_size': 1024,
    'scheduler': 'linear',
    'lr_init': 1e-3,
    'lr_end': 1e-6,
    'lr_warmup': 0.0,
    'optimizer': 'adam',
    'weight_decay': None,
    'clip': 1.0,
}

CONFIGS = {
    'original': CONFIG,
    'tiny': CONFIG_TINY,
    'large': CONFIG_LARGE,
}

@click.group()
def cli():
    pass

@cli.command()
@click.argument('config_name')
def train(config_name):

    run = wandb.init(
            project='conditional-priors-manifold-linear',
            dir=PATH,
            config=CONFIGS[config_name],
        )

    @jax.vmap
    def concat_y_and_A(y: Array, A: Array):
        return jnp.concatenate((y, A.reshape((-1,))))

    def generate(model: nn.Module, **kwargs) -> Array:
        def fun(A: Array, y: Array, key: Array) -> Array:
            return sample_any(
                model=model,
                shape=(len(y), config.features),
                A=inox.Partial(measure, A),
                y=y,
                cov_y=cov_y,
                sampler=config.sampler,
                sde=sde,
                steps=config.discrete,
                maxiter=config.maxiter,
                key=key,
                **kwargs,
            )

        x = jax.vmap(fun)(
            rearrange(A, '(M N) ... -> M N ...', M=256),
            rearrange(y, '(M N) ... -> M N ...', M=256),
            rng.split(256),
        )

        return rearrange(x, 'M N ... -> (M N) ...')

    def generate_conditional(model: nn.Module, y_cond: Array, **kwargs) -> Array:
        """
        Generates samples from P(X | y_cond) where y_cond is (A, y) and y is (Ax + ~N)
        """
        # for a set of Ys generate a set of
        # Xs from the distribution P(X | Y)
        # raise Exception("Sampler not Implemented.")
        def fun(y_cond: Array, key: Array) -> Array:
            return sample_any_conditional(
                model=model,
                shard = True,
                shape=(y_cond.shape[0], config.features), #TODO: Is this correct?
                y_cond=y_cond,
                A=inox.Partial(measure, A),
                y=y,
                cov_y=cov_y,
                sampler=config.sampler,
                sde=sde,
                steps=config.discrete,
                maxiter=config.maxiter,
                key=key,
                **kwargs,
            )

        x = jax.vmap(fun)(
            rearrange(y_cond, '(M N) ... -> M N ...', M=256), # TODO: why the fuck??? better parallelization?
            rng.split(256),
        )

        return rearrange(x, 'M N ... -> (M N) ...')

    def corrupt(x: Array, key: Array) -> Array:
        """ TODO: This is a little bit cheating...
        Given a batch of X samples from P(Y | X) and returns
        a batch of Y.
        Arguments:
            x: batch of latents of shape (B, d)
        Return:
            (y, A): a tuple of the batch of ys and As
                    y has shape (B, d) and A has shape (B, t, d)
        """
        A = jax.random.normal(key[1], (x.shape[0], config.observe, config.features))
        corrupted = measure(A, x)
        corrupted += jnp.sqrt(cov_y) * rng.normal(shape = corrupted.shape)
        return corrupted, A

    def sample_ycond(rng: inox.random.PRNG, size: int, y: Array, A: Array):
        """
        Takes a batch of samples from A and y. Each A has shape (t, d)
        and each y has shape (d)
        
        Parameters:
            rng: Pseudo Random Number Generator
            size: sample size
        
        Returns:
            a tensor of size (size, t * d + d) which is `size` samples
            of y and A vectorized and then concatenated to each other.
        """
        i = rng.randint(shape=(size,), minval=0, maxval=len(pi))
        y_sample = y[i]
        A_sample = A[i]
        return concat_y_and_A(y_sample, A_sample)

    class MyDict():
        def __init__(self, x):
            self.x = x
        def __getattr__(self, attr):
            return self.x[attr]



    config = MyDict(CONFIGS[config_name])

    # RNG
    seed = hash("DiffEM") % 2**16
    rng = inox.random.PRNG(seed)

    # SDE
    sde = VESDE(**config.sde)

    # Data
    keys = jax.random.split(jax.random.key(config.seed))

    ## Latent
    x = smooth_manifold(keys[0], shape=(config.samples,), m=1, n=config.features)
    x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    x = 4.0 * x - 2.0

    ## Observations
    A = jax.random.normal(keys[1], (config.samples, config.observe, config.features))
    A = A / jnp.linalg.norm(A, axis=-1, keepdims=True)

    cov_y = config.noise**2 * jnp.ones(config.observe)

    y = measure(A, x) + jnp.sqrt(cov_y) * rng.normal((config.samples, config.observe))

    yA = concat_y_and_A(y, A)

    mu_x, cov_x = fit_moments(
        features=config.features,
        rank=config.features,
        A=inox.Partial(measure, A),
        y=y,
        cov_y=cov_y,
        sampler='ddim',
        sde=sde,
        steps=256,
        maxiter=None,
        key=rng.split(),
    )


    # Model

    # Training from scratch:
    model = make_model_conditional(key=rng.split(), **CONFIGS[config_name])
    pi = generate(ConditionalGaussianDenoiser(mu_x, cov_x))
    ######


    model.train(True)

    static, params, others = model.partition(nn.Parameter)

    # Objective
    objective = ConditionalDenoiserLoss(sde=sde)

    # Optimizer
    optimizer = Adam(
        steps=config.epochs,
        scheduler = config.scheduler,
        lr_init = config.lr_init,
        lr_end = config.lr_end,
        lr_warmup = config.lr_warmup,
        weight_decay = config.weight_decay,
        clip = config.clip
    )
    opt_state = optimizer.init(params)


    # Training
    @jax.jit
    def ell(params, others, x, y_cond, key):
        keys = jax.random.split(key, 5)

        z = jax.random.normal(keys[0], shape=x.shape)
        # t = jax.random.beta(keys[1], a=3.5, b=2, shape=x.shape[:1])kk
        t1 = jax.random.beta(keys[1], a=2, b=5, shape=x.shape[:1])
        t2 = jax.random.beta(keys[2], a=5, b=2, shape=x.shape[:1])
        b = jax.random.bernoulli(keys[3], p = 0.5, shape = x.shape[:1])
        t = t1 * b + (1 - b) * t2

        return objective(static(params, others), x, z, t, y_cond, key=keys[4])

    @jax.jit
    def sgd_step(params, others, opt_state, x, y_cond, key):
        loss, grads = jax.value_and_grad(ell)(params, others, x, y_cond, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return loss, params, opt_state

    generated_y, generated_A = corrupt(pi, rng.split(1))
    generated_ycond = concat_y_and_A(generated_y, generated_A)


    
    yA1 = yA
    pi1 = pi

    all_losses = []

    for lap in trange(32):
        losses = []

        for epoch in range(config.epochs):
            i = rng.randint(shape=(config.batch_size,), minval=0, maxval=len(x))
            loss, params, opt_state = sgd_step(params, others, opt_state, pi1[i], yA1[i], rng.split())
            losses.append(loss)
            if(epoch == config.epochs - 1):
                print(f'EPOCH {epoch}: LOSS: {loss}')

        losses = np.stack(losses)

        all_losses.append(losses)

        dump_module(model, f'[some path]/manifold_dir/checkpoints_schedule_mixa2b5/checkpoint_{lap}.pkl')

        model = static(params, others)
        model.train(False)

        # Generating pi1
        model=model
        shape=(config.samples, config.features)
        sampler='ddpm'
        sde=sde
        steps=config.discrete
        maxiter=config.maxiter
        key=rng.split()

        mu_x = None # getattr(model, 'mu_x', None)
        cov_x = None # getattr(model, 'cov_x', None)

        if sampler == 'ddpm':
            sampler = ConditionalDDPM(model)
        elif sampler == 'ddim':
            sampler = DDIM(model)
        elif sampler == 'pc':
            sampler = PredictorCorrector(model)

        z = jax.random.normal(key, shape)

        if mu_x is None:
            x1 = sampler.sde(0.0, z, 1.0)
        else:
            x1 = sampler.sde(mu_x, z, 1.0)

        pi1 = sampler(x1, t = 1.0, y = yA, steps=steps, key=key)


        yA1 = corrupt(pi1, rng.split(1))
        yA1 = concat_y_and_A(yA1[0], yA1[1])

        divergence = sinkhorn_divergence(
                x[:16384],
                x[-16384:],
                pi1[:16384],
            )
        
        fig = show_corner(pi1)._figure

        run.log({
            'loss': np.mean(losses),
            'loss_std': np.std(losses),
            'divergence': divergence,
            'corner': wandb.Image(fig),
        })


        opt_state = optimizer.init(params)


if __name__ == "__main__":
    cli()
