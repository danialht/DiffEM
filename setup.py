import argparse
import logging
from pathlib import Path
from datasets import load_dataset
import zipfile
import numpy as np
from PIL import Image
from tqdm import tqdm

from experiments.cifar.data import corrupt as cifar_corrupt
from experiments.celeba.data import corrupt_dataset as celeba_corrupt

log = logging.getLogger(__name__)

def transform_img(img: Image, width, height):
    """Center crop an image to a square, convert to RGB if needed, and resize."""
    img = np.array(img)
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
    if img.ndim == 2:
        img = img[:, :, np.newaxis].repeat(3, axis=2)
    img = Image.fromarray(img, 'RGB')
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    return img

def setup_directory(
    path: str,
    dataset: str,
    maskprob: int,
    celeba_path: str
    ) -> None:
    """
    Setup the directory for checkpoints, logs,
    and train/test datasets, generated datasets and samples
    for a specific experiment.

    Args:
        path: The DiffEM path
    """
    path = Path(path)
    log.info(f"Setting up DiffEM directory at: {path.resolve()}")
    
    if dataset not in ["manifold", "cifar", "celeba"]:
        log.error(f"Unsupported dataset: {dataset}.\
        Supported datasets are 'manifold', 'cifar', 'celeba'.")
        return
    
    if not path.exists():
        log.error(f"The specified path does not exist: {path.resolve()}")
        return

    # Create a directory for the dataset
    dataset_dir = path / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Create Subdirectories
    if dataset == "manifold":
        subdirs = ["checkpoints"]
    else:
        subdirs = ["checkpoints", "datasets_eval", "datasets"]
    for subdir in subdirs:
        dir_path = dataset_dir / subdir
        dir_path.mkdir(parents=True, exist_ok=True)

    # Fill the datasets directory with the clean dataset
    # for the experiment
    if dataset=="cifar":
        log.info("Loading CIFAR-10 dataset...")
        load_dataset('cifar10', cache_dir=dataset_dir / 'datasets/clean/')
        log.info(f"Creating image version of CIFAR-10 in {path}/datasets_eval/ for easier evaluation...")
        cifar_corrupt(
            cifar_dir_path=dataset_dir,
            maskprob=maskprob,
        )

    elif dataset=="celeba":
        # expects the file img_align_celeba.zip to be unzipped and the folder be
        # to be in the directory
        if celeba_path == "":
            log.info("You need to provide the path to img_align_celeba directory using --celeba_path")
            return
        celeba_path = Path(celeba_path)
        if not celeba_path.exists() or not celeba_path.is_dir():
            log.error(f"The specified CelebA directory directory does not exist: {celeba_path}")
            return
        log.info(f"Loading CelebA dataset from {celeba_path}")
        log.info(f"Creating image version of CelebA in {path}/datasets_eval/ for easier evaluation...")
        # center crop, resize to 64x64 and save
        save_dir = dataset_dir / 'datasets_eval/clean'
        save_dir.mkdir(parents=True, exist_ok=True)
        for counter, img_file in tqdm(enumerate(sorted(celeba_path.iterdir())), desc="Transforming CelebA images"):
            img = Image.open(img_file)
            img = transform_img(img, 64, 64)
            save_path = save_dir / f"{counter}.png"
            img.save(save_path)
        
        # save the dataset using datasets library
        celeba_corrupt(
            str(save_dir),
            dataset_dir/f'datasets/clean',
            maskprob=maskprob,
            seed=123)
        
        # corrupt and save the dataset with corruptoin probability
        celeba_corrupt(
            str(save_dir),
            dataset_dir/f'datasets/mask{maskprob}',
            maskprob=maskprob,
            seed=123)
    
    elif dataset=="manifold":
        log.info("No setup avialable yet for manifold dataset.")
        return


def main():
    """Main entry point for the setup script."""
    parser = argparse.ArgumentParser(
        description="Setup the DiffEM directory at a specified path",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n\npython setup.py ~/diffem_files cifar\npython setup.py /root/diffem_files celeba"
    )
    
    parser.add_argument(
        "path",
        type=str,
        help="Path where the setup will be performed"
    )
    
    parser.add_argument(
        "dataset",
        type=str,
        help="Name of the dataset to be used,\
        could be one of ['manifold', 'cifar', 'celeba']"
    )

    parser.add_argument(
        "--maskprob",
        type=int,
        default=75,
        help="Masking probability as an integer (default: 75)"
    )

    parser.add_argument(
        "--celeba_path",
        type=str,
        default="",
        help="Path to the CelebA dataset zip file if not in the current directory"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="\n*** %(message)s\n"
    )
    
    # Run setup
    setup_directory(args.path, args.dataset, args.maskprob, args.celeba_path)

if __name__ == "__main__":
    main()
