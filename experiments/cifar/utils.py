r"""CIFAR experiment helpers"""

import os

from jax import Array
from pathlib import Path
from typing import *

# isort: split
from priors.common import *
from priors.data import *
from priors.diffusion import *
from priors.image import *
from priors.nn import *
from priors.optim import *

if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'priors/cifar'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


def measure(A: Array, x: Array) -> Array:
    return flatten(A * unflatten(x, 32, 32))

# maybe jit?
# @jax.jit
def measure2(A, x):
    return (A @ jnp.expand_dims(x, axis = 2)).squeeze()

def sample(
    model: nn.Module,
    y: Array,
    A: Array,
    key: Array,
    shard: bool = False,
    A_is_func: bool = False,
    **kwargs,
) -> Array:
    if shard and not A_is_func:
        y, A = distribute((y, A))
    elif shard:
        y = distribute(y)

    if A_is_func:
        x = sample_any(
            model=model,
            shape=flatten(y).shape,
            shard=shard,
            A=A,
            y=flatten(y),
            cov_y=1e-3**2,
            key=key,
            **kwargs,
        )
    else:
        x = sample_any(
            model=model,
            shape=flatten(y).shape,
            shard=shard,
            A=inox.Partial(measure, A),
            y=flatten(y),
            cov_y=1e-3**2,
            key=key,
            **kwargs,
        )

    x = unflatten(x, 32, 32)

    return x

def sample_conditional(
    model: nn.Module,
    y_cond: Array,
    # A: Array,
    key: Array,
    shard: bool = False,
    **kwargs,
) -> Array:
    # if shard:
    #     y, A = distribute((y, A))

    x = sample_any_conditional(
        model=model,
        shape=flatten(y_cond).shape,
        shard=shard,
        y_cond=flatten(y_cond),
        cov_y=1e-3**2,
        key=key,
        **kwargs,
    )

    x = unflatten(x, 32, 32)

    return x


def make_model(
    key: Array,
    hid_channels: Sequence[int] = (64, 128, 256),
    hid_blocks: Sequence[int] = (3, 3, 3),
    kernel_size: Sequence[int] = (3, 3),
    emb_features: int = 256,
    heads: Dict[int, int] = {2: 1},
    dropout: float = None,
    **absorb,
) -> Denoiser:
    return Denoiser(
        network=FlatUNet(
            in_channels=3,
            out_channels=3,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=kernel_size,
            emb_features=emb_features,
            heads=heads,
            dropout=dropout,
            key=key,
        ),
        emb_features=emb_features,
    )


class FlatUNet(UNet):
    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        x = unflatten(x, width=32, height=32)
        x = super().__call__(x, t, key)
        x = flatten(x)

        return x


def make_model_conditional(
    key: Array,
    hid_channels: Sequence[int] = (64, 128, 256),
    hid_blocks: Sequence[int] = (3, 3, 3),
    kernel_size: Sequence[int] = (3, 3),
    emb_features: int = 256,
    heads: Dict[int, int] = {2: 1},
    dropout: float = None,
    **absorb,
) -> Denoiser:
    return ConditionalDenoiser(
        network=ConditionalFlatUNet(
            in_channels=6,              # 3 for the input image and 3 for the image we are conditioning on
            out_channels=6,             # Increasing the output channels
            num_returned_channels=3,    # We only need the first 3 channels to be outputed to us
            hid_channels=hid_channels,  # Keeping the intermediate layers the same size
            hid_blocks=hid_blocks,      # Keeping the intermediate layers the same size
            kernel_size=kernel_size,
            emb_features=emb_features,
            heads=heads,
            dropout=dropout,
            key=key,
        ),
        emb_features=emb_features,
    )

class ConditionalFlatUNet(UNet):
    def __call__(self, x: Array, t: Array, y_cond: Array, key: Array = None) -> Array:
        x = unflatten(x, width=32, height=32)
        y_cond = unflatten(y_cond, width = 32, height=32)
        res = jnp.concatenate((x, y_cond), axis = -1)

        res = super().__call__(res, t, key)

        # TODO: Dropping some of the channels
        res = res[..., :self.num_returned_channels]
        
        res = flatten(res)

        return res
    

class Config:
    def __init__(self, d):
        self.d = d
    def __getattr__(self, s):
        return self.d[s]
    def __getitem__(self, key):
        return d[key]
    
