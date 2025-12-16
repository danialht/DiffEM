r"""CelebA experiment helpers"""

import os

from datasets import Dataset

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
    return flatten(A * unflatten(x, 64, 64))


def sample(
    model: nn.Module,
    y: Array|None,
    A: Array|None,
    key: Array,
    shard: bool = False,
    **kwargs,
) -> Array:
    if shard:
        y, A = distribute((y, A))

    x = sample_any(
        model=model,
        shape=flatten(y).shape if y is not None else (16, 64 * 64 * 3),
        shard=shard,
        A=inox.Partial(measure, A) if A is not None else None,
        y=flatten(y) if y is not None else None,
        cov_y=1e-3**2,
        key=key,
        **kwargs,
    )

    x = unflatten(x, 64, 64)

    return x


def normalize_dataset(dataset: Dataset, col_name: str, apply_corruption: bool):
    def normalize_transform_cor(batch):
        x = batch[col_name] * 4 / 256 - 2
        A = batch['A']
        return {col_name: A * x}

    def normalize_transform(batch):
        x = batch[col_name] * 4 / 256 - 2
        return {col_name: x}

    dataset = dataset.map(
        normalize_transform if not apply_corruption else normalize_transform_cor,
        batched=True,
        batch_size=256,
        num_proc=4,
        desc="Normalizing dataset",
    )

    return dataset


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

    x = unflatten(x, 64, 64)

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
        x = unflatten(x, width=64, height=64)
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
        x = unflatten(x, width=64, height=64)
        y_cond = unflatten(y_cond, width = 64, height=64)
        res = jnp.concatenate((x, y_cond), axis = -1)


        res = super().__call__(res, t, key)

        # TODO: Dropping some of the channels
        res = res[..., :self.num_returned_channels]
        
        res = flatten(res)

        return res
    

class MyDict:
    def __init__(self, d):
        self.d = d
    def __getattr__(self, s):
        return self.d[s]