"""
Reference Inception statistics (mu, sigma) of the CelebA-64 training set.
"""

# isort: off
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

# Prevent TensorFlow from pre-allocating GPU memory
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
# isort: on

import argparse

import common.datasets as datasets
import common.fid_utils as fid_utils
import jax.numpy as jnp
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("CelebA-64 FID reference statistics")
    parser.add_argument("--dataset_location", type=str, required=True)
    parser.add_argument("--out", type=str, default="celeba_stats.npz")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ds = datasets.load_celeb_a(args.dataset_location, "train", shuffle_files=False)
    if args.max_samples is not None:
        ds = ds.take(args.max_samples)
    ds = (
        ds.map(
            # same preprocessing as training, back to HWC for Inception
            lambda x: tf.transpose(datasets.preprocess_celeb_a(x["image"]), [1, 2, 0]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(args.batch_size)
        .as_numpy_iterator()
    )

    inception = fid_utils.get_fid_network()
    stats = fid_utils.OnlineStats()
    for batch in ds:
        stats.update(np.asarray(fid_utils.resize_and_incept(jnp.asarray(batch), inception)))
        print(f"\rProcessed {stats.n}", end="", flush=True)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out, mu=stats.mu, sigma=stats.sigma)
    print(f"\nSaved stats over {stats.n} images to {args.out}")


if __name__ == "__main__":
    main()
