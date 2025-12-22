from pathlib import Path

from datasets import load_from_disk, Array3D, Features, load_dataset

from .utils import *

from scipy.special import softmax
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from scipy.linalg import sqrtm

from priors.diffusion import ConditionalDenoiser

import logging

import torch
import torch.nn.functional as F
import torchvision


class InceptionV3ForEval(torch.nn.Module):
    """
    Inception v3 wrapper that exposes:
      - pooled features (2048-d) for FID / FD-infinity / PRDC
      - logits (1000-d) for Inception Score

    Matches the behavior used in dgm-eval.
    """
    def __init__(self):
        super().__init__()

        weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
        self.inception = torchvision.models.inception_v3(
            weights=weights,
            aux_logits=True,
            transform_input=False,
        )
        self.inception.eval()

        # Disable gradients
        for p in self.inception.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, 299, 299), ImageNet-normalized

        Returns:
            feats:  (B, 2048)
            logits: (B, 1000)
        """
        m = self.inception

        # forward copied from torchvision, stopping at pooling
        x = m.Conv2d_1a_3x3(x)
        x = m.Conv2d_2a_3x3(x)
        x = m.Conv2d_2b_3x3(x)
        x = m.maxpool1(x)

        x = m.Conv2d_3b_1x1(x)
        x = m.Conv2d_4a_3x3(x)
        x = m.maxpool2(x)

        x = m.Mixed_5b(x)
        x = m.Mixed_5c(x)
        x = m.Mixed_5d(x)

        x = m.Mixed_6a(x)
        x = m.Mixed_6b(x)
        x = m.Mixed_6c(x)
        x = m.Mixed_6d(x)
        x = m.Mixed_6e(x)

        x = m.Mixed_7a(x)
        x = m.Mixed_7b(x)
        x = m.Mixed_7c(x)

        # Global average pooling → (B, 2048)
        feats = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        feats = feats.flatten(1)

        # Logits for IS
        logits = m.fc(feats)

        return feats, logits

def load_inception_model(device: str | None = None) -> InceptionV3ForEval:
    """
    Load frozen Inception v3 model for evaluation.

    Returns:
        InceptionV3ForEval (eval mode, no gradients)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = InceptionV3ForEval().to(device)
    model.eval()
    return model

def to_uint8(images: np.ndarray) -> np.ndarray:
    """
    Convert images from [-2, 2] float32 → [0, 255] uint8
    """
    images = (images + 2.0) / 4.0          # → [0, 1]
    images = np.clip(images, 0.0, 1.0)
    images = (images * 255.0).astype(np.uint8)
    return images

def preprocess_inception(images_uint8: np.ndarray, device=None) -> torch.Tensor:
    """
    images_uint8: (B,H,W,C) uint8 in [0,255]
    returns: torch.Tensor (B,3,299,299) normalized
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
    tfm = weights.transforms()

    out = []
    for img in images_uint8:
        pil = torchvision.transforms.functional.to_pil_image(img)
        out.append(tfm(pil))
    return torch.stack(out, dim=0).to(device)

def load_dinov2_model(model_name="dinov2_vitb14", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    model.to(device)
    return model


def preprocess_dinov2(images_uint8: np.ndarray, device=None) -> torch.Tensor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    out = []
    for img in images_uint8:
        pil = torchvision.transforms.functional.to_pil_image(img)
        out.append(tfm(pil))
    return torch.stack(out, dim=0).to(device)


def extract_inception_features(images: np.ndarray, batch_size=64):
    """
    Returns:
        features: (N, 2048)
        logits:   (N, 1000)  (for IS)
    """
    model = load_inception_model()

    feats, logits = [], []

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch = preprocess_inception(batch, device=next(model.parameters()).device)
        f, l = model(batch) # model(batch, return_logits=True)
        feats.append(f.detach().cpu())
        logits.append(l.detach().cpu())

    return np.concatenate(feats), np.concatenate(logits)

def extract_dinov2_features(images: np.ndarray, batch_size=64):
    """
    Returns:
        features: (N, D)
    """
    model = load_dinov2_model()
    device = next(model.parameters()).device

    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch = preprocess_dinov2(batch, device=device)  # resize 224, imagenet norm
        f = model(batch)
        if f.ndim == 4:
            f = f.mean(axis=(-2, -1))
        feats.append(f.detach().cpu())

    return np.concatenate(feats)

def compute_fid_from_features(real_feats, gen_feats):
    mu_r, mu_g = real_feats.mean(0), gen_feats.mean(0)
    cov_r = np.cov(real_feats, rowvar=False)
    cov_g = np.cov(gen_feats, rowvar=False)

    covmean = sqrtm(cov_r @ cov_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu_r - mu_g) ** 2) + np.trace(cov_r + cov_g - 2 * covmean)
    return float(fid)

def compute_fd_infinity(real_feats, gen_feats, num_points=15):
    n = min(len(real_feats), len(gen_feats))
    sizes = np.unique(
        np.maximum(2, (np.linspace(0.1, 1.0, num_points) * n).astype(int))
    )

    fds = []
    inv_ns = []

    for s in sizes:
        idx_r = np.random.choice(len(real_feats), s, replace=False)
        idx_g = np.random.choice(len(gen_feats), s, replace=False)

        fd = compute_fid_from_features(real_feats[idx_r], gen_feats[idx_g])
        fds.append(fd)
        inv_ns.append(1.0 / s)

    reg = LinearRegression().fit(np.array(inv_ns).reshape(-1, 1), fds)
    return float(reg.intercept_)

def compute_prdc(real_feats, gen_feats, k=5):
    nn_real = NearestNeighbors(n_neighbors=k).fit(real_feats)
    nn_gen = NearestNeighbors(n_neighbors=k).fit(gen_feats)

    real_to_real = nn_real.kneighbors(real_feats)[0][:, -1]
    gen_to_gen   = nn_gen.kneighbors(gen_feats)[0][:, -1]

    gen_to_real = nn_real.kneighbors(gen_feats)[0][:, 0]
    real_to_gen = nn_gen.kneighbors(real_feats)[0][:, 0]

    precision = np.mean(gen_to_real < real_to_real)
    recall    = np.mean(real_to_gen < gen_to_gen)
    density   = np.mean(real_to_gen < real_to_real[:, None])
    coverage  = np.mean(np.min(real_to_gen, axis=1) < real_to_real)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'density': float(density),
        'coverage': float(coverage),
    }

def compute_inception_score(logits, splits=10):
    probs = softmax(logits, axis=1)
    scores = []

    for part in np.array_split(probs, splits):
        p_y = part.mean(axis=0)
        kl = entropy(part.T, p_y[:, None])
        scores.append(np.exp(kl.mean()))

    return float(np.mean(scores)), float(np.std(scores))

def generate_dataset(model, dataset, rng, batch_size, conditional=True, **kwargs):
    """
    generates dataset by sampling using conditional or unconditional denoiser
    """
    def transform_conditional(batch):
        y_cond = batch['y']
        x = sample_conditional(model, y_cond, rng.split(), **kwargs)
        x = np.asarray(x)

        return {'x': x}

    def transform_unconditional(batch):
        y_cond = batch['y']
        x = sample(model, y_cond, rng.split(), **kwargs)
        x = np.asarray(x)

        return {'x': x}

    transform = transform_conditional if conditional else transform_unconditional

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

def evaluate(
    diffem_files_dir: Path,
    experiment: str,
    metrics: List[str],
    run_name: str,
    checkpoint_index: int,
    corruption_level: int,
    corruption_name: str,
    # output: str,
    test: bool = False,
) -> None:
    # load model
    checkpoint_path = diffem_files_dir / f'{experiment.dataset_name}/checkpoints/{run_name}' / ('checkpoint_'+str(checkpoint_index)+'.pkl')
    model = load_module(checkpoint_path)

    # load corrupted dataset
    corrupted_dataset_name = f"cifar-{corruption_name}-{corruption_level}" + ('_test' if test else '')
    corrupted_dataset_path = diffem_files_dir / experiment.dataset_name / 'datasets' / corrupted_dataset_name
    corrupted_dataset = load_from_disk(corrupted_dataset_path)
    corrupted_dataset.set_format('numpy')

    # Sharding
    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    # RNG
    seed = hash((run_name, checkpoint_index, 'unconditional')) % (1<<16)
    np_rng = np.random.default_rng(seed=seed)
    rng = inox.random.PRNG(seed)

    # SDE
    sde = VESDE(**experiment.sampler.sde)

    # Load datasets
    corrupted_dataset = corrupted_dataset['train']
    original_dataset_path = diffem_files_dir / experiment.dataset_name / 'datasets/clean'
    original_dataset = load_dataset('cifar10', cache_dir=original_dataset_path)
    original_dataset = original_dataset['train'].remove_columns(['label']).rename_column('img', 'x')

    # generate dataset
    generated_dataset = generate_dataset(
            model=model,
            dataset=corrupted_dataset,
            rng=rng,
            batch_size=256,
            shard=True,
            sampler=experiment.sampler.name,
            sde=sde,
            steps=experiment.sampler.discrete,
            maxiter=experiment.sampler.maxiter,
            conditional=isinstance(model, ConditionalDenoiser)
        )

    gen_imgs = to_uint8(np.stack(generated_dataset['x']))
    real_imgs = to_uint8(np.stack(original_dataset['x']))

    results = {}

    if 'fid' in metrics or 'fdinf' in metrics or 'is' in metrics:
        logging.info("extracting inception features")
        real_feats, _ = extract_inception_features(real_imgs)
        gen_feats, gen_logits = extract_inception_features(gen_imgs)

    if 'fid' in metrics:
        logging.info("computing FID score")
        results['fid'] = compute_fid_from_features(real_feats, gen_feats)

    if 'is' in metrics:
        logging.info("computing inception score")
        is_mean, is_std = compute_inception_score(gen_logits)
        results['is_mean'] = is_mean
        results['is_std'] = is_std

    if 'fdinf' in metrics:
        logging.info("computing fd infinity score")
        results['fdinf'] = compute_fd_infinity(real_feats, gen_feats)

    del gen_logits
    del real_feats
    del gen_feats

    real_dino = None
    gen_dino = None
    if 'prdc' in metrics:
        logging.info("extracting dinov2 features")
        real_dino = extract_dinov2_features(real_imgs)
        gen_dino  = extract_dinov2_features(gen_imgs)

        logging.info("computing prdc")
        results.update(compute_prdc(real_dino, gen_dino))
        # results.update(compute_prdc(real_feats, gen_feats))

    if 'fddinov2' in metrics:
        logging.info("computing fd dinov2")
        if real_dino is None: real_dino = extract_dinov2_features(real_imgs)
        if gen_dino is None: gen_dino  = extract_dinov2_features(gen_imgs)
        results['fddinov2'] = compute_fid_from_features(real_dino, gen_dino)

    with open(output, 'a') as f:
        f.write(str(results) + "\n")
    