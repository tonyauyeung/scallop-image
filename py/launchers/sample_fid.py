"""
FID of n-step flow map samples on CelebA-64 (50k samples in the paper).
"""

# isort: off
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)
# isort: on

import argparse
import importlib
import json

import common.datasets as datasets
import common.fid_utils as fid_utils
import common.flow_map as flow_map
import common.state_utils as state_utils
import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser("CelebA-64 FID")
    parser.add_argument("--cfg_path", type=str, default="configs.celeba64")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--stats", type=str, required=True, help="Reference FID stats (.npz).")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--div_head", type=str, default="vector", choices=["none", "vector", "scalar"]
    )
    parser.add_argument("--n_steps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--n_images", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--ema_fac", type=float, default=0.9999, help="Negative: raw params.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_samples", type=int, default=0, help="Also save this many samples as .npz."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = importlib.import_module(args.cfg_path).get_eval_config("", args.div_head)
    ema_fac = args.ema_fac if args.ema_fac >= 0 else None
    net, params = state_utils.load_params_for_eval(
        cfg, args.ckpt_path, ema_fac, jax.random.PRNGKey(0)
    )

    ref = np.load(args.stats)
    inception = fid_utils.get_fid_network()
    sample_rho0 = datasets.setup_base(cfg, jnp.zeros(cfg.problem.image_dims))
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for n_steps in args.n_steps:
        prng_key = jax.random.PRNGKey(args.seed)  # same noise for every step count
        stats = fid_utils.OnlineStats()
        saved = []
        pbar = tqdm(total=args.n_images, desc=f"{n_steps} steps")
        while stats.n < args.n_images:
            curr_bs = min(args.batch_size, args.n_images - stats.n)
            prng_key, step_key = jax.random.split(prng_key)
            x0 = sample_rho0(curr_bs, step_key)
            imgs = flow_map.batch_sample(net.apply, params, x0, n_steps, None)
            imgs = jnp.clip(imgs, -1, 1).transpose(0, 2, 3, 1)  # NCHW -> NHWC
            stats.update(np.asarray(fid_utils.resize_and_incept(imgs, inception)))
            if sum(len(s) for s in saved) < args.save_samples:
                saved.append(np.asarray(datasets.unnormalize_image(imgs)))
            pbar.update(curr_bs)
        pbar.close()

        fid = fid_utils.fid_from_stats(stats.mu, stats.sigma, ref["mu"], ref["sigma"])
        results[f"{n_steps}_steps"] = fid
        print(f"{n_steps} steps: FID = {fid:.4f}")
        if args.save_samples > 0:
            samples = np.concatenate(saved)[: args.save_samples]
            np.savez(os.path.join(args.out_dir, f"samples_{n_steps}steps.npz"), samples=samples)

    with open(os.path.join(args.out_dir, "fid_summary.json"), "w") as f:
        json.dump({"args": vars(args), "fid": results}, f, indent=2)


if __name__ == "__main__":
    main()
