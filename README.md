# <img src="assets/scallop-yellow.png" alt="SCALLOP shell" width="30"> SCALLOP: image experiments

[![Conference](https://img.shields.io/badge/NeurIPS-2026-purple)](#reference)
[![arXiv](https://img.shields.io/badge/arXiv-2606.29110-b31b1b.svg)](https://arxiv.org/abs/2606.29110)

[**Project page: SCALLOP — Few-Step Boltzmann Generators via Scalable Likelihood Flow Maps**](https://tonyauyeung.github.io/scallop/)

This repository contains the JAX implementation of the **CelebA-64 image experiments** in
[SCALLOP: Few-Step Boltzmann Generators via Scalable Likelihood Flow Maps](https://arxiv.org/pdf/2606.29110) (NeurIPS 2026).
The PyTorch implementation for the peptide experiments is at
[tonyauyeung/scallop](https://github.com/tonyauyeung/scallop).

- [Environment](#environment)
- [Data](#data)
- [Experiments](#experiments)
  - [Pretrain the LSD flow map teacher](#pretrain-the-lsd-flow-map-teacher)
  - [Likelihood flow maps: SCALLOP and F2D2](#likelihood-flow-maps)
  - [Evaluation: BPD and FID](#evaluation)
- [Code layout](#code-layout)
- [Acknowledgments](#acknowledgments)
- [Reference](#reference)

A likelihood flow map jumps from noise to data in a few steps and predicts the change in
log density along the way. SCALLOP and F2D2 differ only in the divergence target used to
train that likelihood head. F2D2 uses Hutchinson's trace estimator. SCALLOP uses a
Hutchinson-free target computed from the interpolant, which needs no Jacobian-vector
product and gives lower-variance training.

<a id="environment"></a>

## Environment 🏝️

```bash
conda create -n scallop-image python=3.11
conda activate scallop-image
pip install -r requirements.txt
# NVIDIA GPUs:
pip install "jax[cuda12]==0.6.2"
```

Training uses every local GPU through `jax.pmap` (single node).
TensorFlow is used only for data loading on the CPU.

<a id="data"></a>

## Data 🖼️

CelebA is loaded with [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/celeb_a)
(`celeb_a`) from `--dataset_location` and center-cropped and resized to 64×64.
If the automatic download fails (the files are hosted on Google Drive),
place the raw files in TFDS's manual-download directory.

Compute FID reference statistics on the training set once. The training config looks for them at
`<dataset_location>/celeb_a/celeba_stats.npz`:

```bash
python py/launchers/calc_dataset_fid_stats.py --dataset_location '<data-dir>' \
    --out '<data-dir>/celeb_a/celeba_stats.npz'
```

The Inception weights for FID ship in `py/data/inception_v3_weights_fid.pickle`.

<a id="experiments"></a>

## Experiments 🧪

All hyperparameters are in `configs/celeba64.py` (EDM2 UNet).
Checkpoints are written to `--output_folder` as `<run-name>_<step // save_freq>.pkl`.
Metrics are logged to Weights & Biases (project `scallop`; set `WANDB_ENTITY` or `WANDB_MODE=offline`).

<a id="pretrain-the-lsd-flow-map-teacher"></a>

### Pretrain the LSD flow map teacher

The teacher is a flow map trained from scratch with Lagrangian self-distillation (LSD),
combining flow matching on the diagonal s = t with self-distillation on jumps (s, t).

```bash
python py/launchers/learn.py --mode pretrain_flowmap \
    --dataset_location '<data-dir>' --output_folder '<run-dir>/lsd'
```

<a id="likelihood-flow-maps"></a>

### Likelihood flow maps: SCALLOP and F2D2

Both methods initialize from the LSD teacher and add only a divergence head, which predicts a
per-pixel divergence field. The teacher also supplies the velocity target on the diagonal.

#### <img src="assets/scallop-yellow.png" alt="SCALLOP shell" width="15"> SCALLOP

```bash
python py/launchers/learn.py --mode dlogpdt \
    --dataset_location '<data-dir>' --output_folder '<run-dir>/scallop' \
    --network_load_path '<lsd-checkpoint.pkl>' --teacher_load_path '<lsd-checkpoint.pkl>'
```

#### F2D2

```bash
python py/launchers/learn.py --mode hutchinson \
    --dataset_location '<data-dir>' --output_folder '<run-dir>/f2d2' \
    --network_load_path '<lsd-checkpoint.pkl>' --teacher_load_path '<lsd-checkpoint.pkl>'
```

<a id="evaluation"></a>

### Evaluation: BPD and FID

BPD is computed on the full CelebA test set with uniform dequantization. The dequantization
noise is seeded per image, so every model is evaluated on the same inputs. Each run writes
per-image BPDs to `<out-dir>/bpd_<estimator>_<n>steps.csv` and a JSON summary.

**Teacher reference** (Euler steps of the teacher velocity with Hutchinson's estimator):

```bash
python py/launchers/eval_bpd.py --dataset_location '<data-dir>' \
    --ckpt_path '<lsd-checkpoint.pkl>' --div_head none \
    --estimator hutchinson --n_steps 1024 --out_dir '<run-dir>/eval/teacher'
```

**SCALLOP / F2D2** with the divergence head. Passing the teacher CSV
also reports the deviation of the mean BPD from the teacher (|Δ|) and the mean per-image
absolute BPD error (Err):

```bash
python py/launchers/eval_bpd.py --dataset_location '<data-dir>' \
    --ckpt_path '<scallop-checkpoint.pkl>' --n_steps 1 2 4 8 \
    --reference_csv '<run-dir>/eval/teacher/bpd_hutchinson_1024steps.csv' \
    --out_dir '<run-dir>/eval/scallop'
```

**FID** of n-step samples:

```bash
python py/launchers/sample_fid.py --ckpt_path '<scallop-checkpoint.pkl>' \
    --stats '<data-dir>/celeb_a/celeba_stats.npz' --n_steps 1 2 4 8 \
    --out_dir '<run-dir>/eval/scallop'
```

Use `--div_head none` for checkpoints without a divergence head, such as the LSD teacher.
The eval scripts check that the checkpoint matches the architecture exactly. Both use EMA
0.9999 weights by default (`--ema_fac -1` selects the raw weights).

<a id="code-layout"></a>

## Code layout

```
py/
  configs/celeba64.py           # training recipes (LSD teacher / likelihood flow map) and eval config
  common/
    edm2_net.py                 # EDM2 UNet with optional (vectorized) divergence head
    network_utils.py, flow_map.py   # flow map X_{s,t}, velocity b_t, few-step sampling
    losses.py                   # diagonal + LSD losses; Hutchinson (F2D2) and SCALLOP targets
    loss_args.py                # time sampling and batch assembly
    likelihood.py               # log-likelihood / BPD integrators
    fid_utils.py                # JAX Inception-v3 and FID
    datasets.py, logging.py, state_utils.py, updates.py, dist_utils.py, interpolant.py
  launchers/
    learn.py                    # training
    eval_bpd.py                 # test-set BPD
    sample_fid.py               # FID of n-step samples
    calc_dataset_fid_stats.py   # FID reference statistics
  tests/                        # CPU smoke tests: pytest py/tests
```

The tests run every launcher end to end on a tiny UNet with mocked CelebA, so no data
download is needed:

```bash
pip install pytest
pytest py/tests -q
```

<a id="acknowledgments"></a>

## Acknowledgments ⚽️

This codebase builds on [JAX-F2D2](https://github.com/Keely-Ai/JAX-F2D2) and on the
flow map self-distillation code by Nicholas M. Boffi. The JAX Inception-v3 for FID is
adapted from [jax-fid](https://github.com/matthias-wright/jax-fid) and
[shortcut-models](https://github.com/kvfrans/shortcut-models).

<a id="reference"></a>

## Reference 🐨

```bibtex
@inproceedings{ouyang2026fewstep,
  title = {Few-Step Boltzmann Generators via Scalable Likelihood Flow Maps},
  author = {OuYang, RuiKang and Yu, Hanlin and Ai, Xinyue and He, Yutong and Boffi, Nicholas Matthew and Ravikumar, Pradeep and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel and Simchowitz, Max and Miller, Benjamin Kurt and Chehab, Omar},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2026},
  url = {https://openreview.net/forum?id=mfdTHHRIid},
  note = {Also accepted at SPIGM@ICML.},
}
```
