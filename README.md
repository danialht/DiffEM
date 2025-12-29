# [DiffEM: Learning from Corrupted Data with Diffusion Models via Expectation Maximization](https://arxiv.org/abs/2510.12691)

## Abstract

Diffusion models have emerged as powerful generative priors for high-dimensional
inverse problems, yet learning them when only corrupted or noisy observations are
available remains challenging. In this work, we propose a novel method for training
diffusion models with Expectation-Maximization (EM) from corrupted data. Our
proposed method, DiffEM, utilizes conditional diffusion models to reconstruct
clean data from observations in the E-step, and then uses the reconstructed data
to refine the conditional diffusion model in the M-step. Theoretically, we provide
monotonic convergence guarantees for the DiffEM iteration, assuming appropriate
statistical conditions. We demonstrate the effectiveness of our approach through
experiments on various image reconstruction tasks.

## How to use

### Setup working directory and data

Create a directory for storing all the training datasets (clean or corrupted), checkpoints, and all other files. You just need to create the directory, everything inside of directory will be setup by `cli/setup.py`. We will refer to this directory as `diffem_files`, for less confusion we recommend choosing the same name for your directory.

First create the directory:

```bash
mkdir ~/diffem_files
```

For setting up the Cifar-10 experiment data use the following (replace `<diffem_files directory>` with `~/diffem_files` or whatever directory you made with the directory you just created, you can use a different corurption levels, for a corruption level of $\rho_{corrupted}=0.75$ set `<maskprob>=75`):

```bash
python ./cli/setup.py <diffem_files directory> cifar --maskprob <maskprob>
```

For setting up the CelebA experiment you first need to access the dataset from the [official website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), download the `img_align_celeba.zip` and unzip the dataset. Then you can setup your `diffem_files` directory using:

```bash
python ./cli/setup <diffem_files directory> celeba --celeba_path <path to img_align_celeba directory> --maskprob 75
```

After setting up the the `diffem_files` directory the structure would look like this:

```
diffem_files/
├── celeba/
│   ├── checkpoints/          (Where the checkpoints will be saved)
│   ├── datasets/             (Datasets in the HuggingFace format, used by the training pipeline)
│   │   ├── clean/            (Clean dataset)
│   │   └── mask75/           (Corrupted dataset with your chosen noise level, here is shown for 75%)
│   └── datasets_eval/
│       └── clean/            (64x64 PNG images)
│
└── cifar/
    ├── checkpoints/          (Where the checkpoints will be saved)
    ├── datasets/             (Datasets in the HuggingFace format, used by the training pipeline)
    │   ├── clean/            
    │   └── cifar-mask-90/    (Corrupted dataset with your chosen noise level, here is shown for 75%)
    └── datasets_eval/        (Dataset in PNG Image format)
```

## Training

The pipeline first trains a conditional model for $K$ laps, then takes the last model and generates a dataset using that and conditioning on the corrupted dataset, finally trains an unconditional model on this new dataset.

### Training Conditional Model

In order to train the conditional model you simply use the `train.py` script.

```bash
python train.py experiment=cifar run_name=test_run diffem_files_dir=<path/to/diffem_files>\
                training.num_laps=20 experiment.corruption_level=90
```

In order to have a much more fine grained control over training you can use a more sophisticated command, not mentioning of the hyperparameters will automatically set it to default (specified in `conf` directory):

```bash
python train.py run_name=<run name> experiment=<experiment_name e.g. cifar, cleba>\
    diffem_files_dir=<path/to/diffem_files> \
    experiment.corruption_level=<corruption level> experiment.corruption=<corruption name> \
    training.num_laps=<num EM laps> training.epochs=<num epochs> \
    training.batch_size=<batch size> training.clip=<clip> training.ema_decay=<ema_decay> \
    experiment.sampler.name=<sampler name> experiment.sampler.sde.a=<a> experiment.sampler.sde.b=<b> \
    ...
```

You can check the `conf` directory to see what hyperparameters you can change.

### Training the Unconditional Model

To load `checkpoint_x.pkl` from the `checkpoints/run_name` and train a uncondtional model on it use the following and set the index to be `x`. The trained unconditional model will be dumped in the same directory with name `checkpoint_uncondtional_x.pkl`.

```bash
python train_uncond.py experiment=cifar run_name=test_run diffem_files_dir=<path/to/diffem_files> \
    checkpoint_index=<conditional model checkpoint index>
```
Or you can set any of the hyperparameters you desire like the following:

```bash
python train_uncond.py experiment=cifar run_name=test_run diffem_files_dir=<path/to/diffem_files> \
    checkpoint_index=<conditional model checkpoint index>
    experiment.corruption_level=<corruption level> experiment.corruption=<corruption name> \
    training.num_laps=<num EM laps> training.epochs=<num epochs> \
    training.batch_size=<batch size> training.clip=<clip> training.ema_decay=<ema_decay> \
    ...
```

You can check the `conf` directory to see what hyperparameters you can tune.


### Evaluating

For evaluation you use `evaluate.py` to generate a dataset of png, then we recommend using [dgm-eval](https://github.com/layer6ai-labs/dgm-eval) for measuring different metrics. Here is an example usage.

```bash
python ./evaluate.py experiment=cifar checkpoint_index=20 \
    diffem_files_dir=<path/to/diffem_files_dir> run_name=<run_name> conditional=true

python -m dgm_eval diffem_files_dir/cifar/dataset_eval/clean \
    diffem_files_dir/cifar/dataset_eval/run_name/conditional_checkpoint_20 \
    --metrics fd --model dinov2 --nsample 50000
```
