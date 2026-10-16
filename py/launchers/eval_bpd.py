"""
Bits per dimension on the CelebA-64 test set.

  --estimator div_head    likelihood flow map (F2D2 / SCALLOP) with its divergence head
  --estimator hutchinson  Euler integration of the velocity with Hutchinson probes
                          (teacher reference; 1024 steps in the paper)

Per-image BPDs are written to ``<out_dir>/bpd_<estimator>_<n>steps.csv``. Passing
``--reference_csv`` (e.g. the teacher's 1024-step CSV) also reports the absolute
deviation of the mean BPD (|Delta|) and the mean per-image absolute error (Err).
Dequantization noise is seeded per test image, so runs are directly comparable.
"""

# isort: off
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import jax
import jax.numpy as jnp
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
# isort: on

import argparse
import importlib
import json

import common.datasets as datasets
import common.likelihood as likelihood
import common.state_utils as state_utils
import numpy as np
from tqdm.auto import tqdm


def get_test_dataset(args):
    ds = datasets.load_celeb_a(args.dataset_location, "test", shuffle_files=False)
    if args.max_samples is not None:
        ds = ds.take(args.max_samples)
    dequantize = not args.no_dequantize

    def _map(index, example):
        seed = tf.stack([tf.constant(args.seed, tf.int64), index])
        return datasets.preprocess_celeb_a(example["image"], dequantize=dequantize, seed=seed)

    ds = (
        ds.enumerate()
        .map(_map, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds.as_numpy_iterator()


def parse_args():
    parser = argparse.ArgumentParser("CelebA-64 test BPD")
    parser.add_argument("--dataset_location", type=str, required=True)
    parser.add_argument("--cfg_path", type=str, default="configs.celeba64")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--div_head",
        type=str,
        default="vector",
        choices=["none", "vector", "scalar"],
        help="Divergence head of the checkpoint (none for the LSD teacher).",
    )
    parser.add_argument("--divergence_rescale", type=float, default=None)
    parser.add_argument(
        "--estimator", type=str, default="div_head", choices=["div_head", "hutchinson"]
    )
    parser.add_argument("--n_steps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--n_probes", type=int, default=1, help="Hutchinson probes per step.")
    parser.add_argument("--ema_fac", type=float, default=0.9999, help="Negative: raw params.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_dequantize", action="store_true")
    parser.add_argument("--reference_csv", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.estimator == "div_head" and args.div_head == "none":
        raise ValueError("--estimator div_head needs a checkpoint with a divergence head.")

    cfg_module = importlib.import_module(args.cfg_path)
    cfg = cfg_module.get_eval_config(args.dataset_location, args.div_head, args.divergence_rescale)
    ema_fac = args.ema_fac if args.ema_fac >= 0 else None
    prng_key = jax.random.PRNGKey(args.seed)
    net, params = state_utils.load_params_for_eval(cfg, args.ckpt_path, ema_fac, prng_key)
    d = int(np.prod(cfg.problem.image_dims))
    os.makedirs(args.out_dir, exist_ok=True)

    reference = None
    if args.reference_csv is not None:
        reference = np.loadtxt(args.reference_csv, delimiter=",", skiprows=1)[:, 1]

    summary = {}
    for n_steps in args.n_steps:
        bpds = []
        hutch_key = jax.random.PRNGKey(args.seed + 1)
        for batch in tqdm(get_test_dataset(args), desc=f"{n_steps} steps"):
            x1 = jnp.asarray(batch)
            if args.estimator == "div_head":
                logp = likelihood.inverse_logp_div_head(
                    net.apply,
                    params,
                    x1,
                    n_steps,
                    cfg.network.rescale,
                    cfg.network.divergence_rescale,
                )
            else:
                hutch_key, batch_key = jax.random.split(hutch_key)
                logp = likelihood.inverse_logp_hutchinson(
                    net.apply, params, x1, n_steps, cfg.network.rescale, batch_key, args.n_probes
                )
            bpd, _ = likelihood.logp_to_bpd(logp, d, dequantize=not args.no_dequantize)
            bpds.append(np.asarray(bpd))
        bpds = np.concatenate(bpds)

        csv_path = os.path.join(args.out_dir, f"bpd_{args.estimator}_{n_steps}steps.csv")
        np.savetxt(
            csv_path,
            np.stack([np.arange(len(bpds)), bpds], axis=1),
            delimiter=",",
            header="index,bpd",
            comments="",
            fmt=["%d", "%.8f"],
        )
        result = {"bpd": float(bpds.mean()), "n_images": int(len(bpds))}
        if reference is not None:
            if len(reference) != len(bpds):
                raise ValueError("Reference CSV and evaluation cover different images.")
            result["abs_delta"] = float(abs(bpds.mean() - reference.mean()))
            result["err"] = float(np.abs(bpds - reference).mean())
        summary[f"{n_steps}_steps"] = result
        print(f"{n_steps} steps: {result}")

    with open(os.path.join(args.out_dir, f"bpd_{args.estimator}_summary.json"), "w") as f:
        json.dump({"args": vars(args), "results": summary}, f, indent=2)


if __name__ == "__main__":
    main()
