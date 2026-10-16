"""
Nicholas M. Boffi
10/5/25

Main training loop for (likelihood) flow maps on CelebA-64.
"""

# isort: off
import os
import pathlib
import sys

# Set up path for imports FIRST
script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

# Suppress TensorFlow logging before any TF imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

# Import JAX before TensorFlow to avoid node-specific init order issues.
import jax
import jax.numpy as jnp

# Force TensorFlow to use CPU only for data loading - no GPU ops
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")  # Hide all GPUs from TensorFlow
# isort: on

import argparse
import importlib
import time
from typing import Tuple

import common.datasets as datasets
import common.dist_utils as dist_utils
import common.fid_utils as fid_utils
import common.interpolant as interpolant
import common.logging as logging
import common.loss_args as loss_args
import common.losses as losses
import common.state_utils as state_utils
import common.updates as updates
import matplotlib as mpl
import numpy as np
import wandb
from ml_collections import config_dict  # type: ignore
from tqdm.auto import tqdm as tqdm

mpl.rc_file(f"{pathlib.Path(__file__).resolve().parent}/matplotlibrc")


def switch_diag_fraction(
    cfg: config_dict.FrozenConfigDict,
    statics: state_utils.StaticArgs,
    diag_fraction: float,
) -> Tuple[config_dict.FrozenConfigDict, state_utils.StaticArgs]:
    """Rebuild the loss with a new diagonal / off-diagonal batch split."""
    cfg = config_dict.ConfigDict(cfg)
    cfg.optimization.diag_fraction = diag_fraction
    cfg = config_dict.FrozenConfigDict(cfg)
    print(f"Switching diag_fraction to {diag_fraction}.")
    statics = statics._replace(loss=losses.setup_loss(cfg, statics.net, statics.interp))
    return cfg, statics


def train_loop(
    cfg: config_dict.FrozenConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: np.ndarray,
) -> None:
    """Carry out the training loop."""

    logging.register_signal_handlers(cfg, train_state)
    start_step = int(dist_utils.safe_index(cfg, train_state.step))
    switch = cfg.optimization.diag_fraction_switch
    if switch is not None and start_step >= switch[0]:
        cfg, statics = switch_diag_fraction(cfg, statics, switch[1])

    pbar = tqdm(range(start_step, cfg.optimization.total_steps))
    for step in pbar:
        if switch is not None and step == switch[0] and step > start_step:
            cfg, statics = switch_diag_fraction(cfg, statics, switch[1])

        # construct loss function arguments
        start_time = time.time()
        loss_fn_args, prng_key = statics.get_loss_fn_args(
            cfg, statics, train_state, prng_key
        )

        # take a step on the loss
        train_state, loss_value, grads, metrics = statics.train_step(
            train_state, statics.loss, loss_fn_args
        )
        end_time = time.time()

        # compute update to EMA params
        train_state = statics.update_ema_params(train_state)

        # log to wandb
        if (step % cfg.logging.log_freq) == 0:
            prng_key = logging.log_metrics(
                cfg,
                statics,
                train_state,
                grads,
                loss_value,
                metrics,
                loss_fn_args,
                prng_key,
                end_time - start_time,
            )
            pbar.set_postfix(loss=float(np.asarray(loss_value).mean()))

    # dump one final time
    logging.save_state(train_state, cfg)


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Train (likelihood) flow maps on CelebA-64.")
    parser.add_argument("--cfg_path", type=str, default="configs.celeba64")
    parser.add_argument("--dataset_location", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="dlogpdt",
        choices=losses.DIVERGENCE_MODES,
        help="pretrain_flowmap: LSD teacher; hutchinson: F2D2; dlogpdt: SCALLOP.",
    )
    parser.add_argument("--network_load_path", type=str, default=None)
    parser.add_argument("--teacher_load_path", type=str, default=None)
    parser.add_argument(
        "--div_loss_weighting_type",
        type=str,
        default=None,
        choices=["uniform", "linear", "square"],
        help="Diagonal divergence loss weighting; default linear for dlogpdt, "
        "uniform for hutchinson (always uniform for hutchinson).",
    )
    parser.add_argument(
        "--use_importance_sample_for_diag_t",
        action="store_true",
        help="Sample diagonal times with density 1/(1-t) (dlogpdt with linear weighting).",
    )
    parser.add_argument(
        "--scalar_divergence",
        action="store_true",
        help="Use a scalar divergence head instead of the vectorized (per-pixel) head.",
    )
    parser.add_argument(
        "--divergence_rescale",
        type=float,
        default=None,
        help="Divergence target scale; default 1 (vectorized) or d (scalar).",
    )
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    return parser.parse_args()


def setup_config_dict() -> config_dict.ConfigDict:
    args = parse_command_line_arguments()
    cfg_module = importlib.import_module(args.cfg_path)
    cfg = cfg_module.get_config(args.dataset_location, args.output_folder, args.mode)

    if args.network_load_path is not None:
        cfg.network.load_path = args.network_load_path
    if args.teacher_load_path is not None:
        cfg.teacher.load_path = args.teacher_load_path
    if args.total_steps is not None:
        cfg.optimization.total_steps = args.total_steps

    if args.mode == "pretrain_flowmap":
        cfg.teacher.load_path = ""
    else:
        if not cfg.teacher.load_path:
            raise ValueError(f"--mode {args.mode} needs --teacher_load_path (LSD flow map).")

        cfg.network.vectorized_divergence = not args.scalar_divergence
        if args.divergence_rescale is not None:
            cfg.network.divergence_rescale = args.divergence_rescale
        elif cfg.network.vectorized_divergence:
            cfg.network.divergence_rescale = 1.0
        else:
            cfg.network.divergence_rescale = float(cfg.problem.d)

        if args.mode == "hutchinson":
            cfg.network.div_loss_weighting_type = "uniform"
        else:
            cfg.network.div_loss_weighting_type = args.div_loss_weighting_type or "linear"

        if args.use_importance_sample_for_diag_t:
            if cfg.network.div_loss_weighting_type != "linear":
                raise ValueError("Importance sampling of t pairs with linear weighting.")
            cfg.training.time_distr = "linear"

    if args.wandb_name is not None:
        cfg.logging.wandb_name = args.wandb_name
        cfg.logging.output_name = args.wandb_name

    print(f"Mode: {args.mode}")
    if args.mode != "pretrain_flowmap":
        head = "vector" if cfg.network.vectorized_divergence else "scalar"
        print(f"Divergence head: {head}, rescale {cfg.network.divergence_rescale}")
        print(f"Divergence loss weighting: {cfg.network.div_loss_weighting_type}")
    print(f"Diagonal time distribution: {cfg.training.time_distr}")
    return cfg


def setup_state(cfg: config_dict.ConfigDict, prng_key: jnp.ndarray) -> Tuple[
    config_dict.FrozenConfigDict,
    state_utils.StaticArgs,
    state_utils.EMATrainState,
    jnp.ndarray,
]:
    """Construct static arguments and training state objects."""
    cfg, ds = datasets.setup_target(cfg)
    ex_input = next(ds)["image"][0]
    interp = interpolant.setup_interpolant(cfg)

    # prefetch batches to device(s) after grabbing an example
    cfg.training.batch_is_sharded = cfg.training.ndevices > 1
    ds = datasets.prefetch_to_device(cfg, ds, buffer_size=2)
    cfg = config_dict.FrozenConfigDict(cfg)

    # define training state
    train_state, net, schedule, prng_key = state_utils.setup_training_state(
        cfg,
        ex_input,
        prng_key,
    )

    # load optional teacher params (replicated once for pmap)
    teacher_params = state_utils.load_teacher_params(cfg, train_state)
    if teacher_params is not None:
        teacher_params = dist_utils.safe_replicate(cfg, teacher_params)

    # initialize FID network if FID computation is enabled
    inception_fn = None
    if cfg.logging.fid_freq > 0:
        print("Initializing Inception network for FID computation...")
        inception_fn = fid_utils.get_fid_network()

    statics = state_utils.StaticArgs(
        net=net,
        schedule=schedule,
        loss=losses.setup_loss(cfg, net, interp),
        get_loss_fn_args=loss_args.get_loss_fn_args,
        train_step=updates.setup_train_step(cfg),
        update_ema_params=updates.setup_ema_update(cfg),
        ds=ds,
        interp=interp,
        sample_rho0=datasets.setup_base(cfg, ex_input),
        inception_fn=inception_fn,
        teacher_params=teacher_params,
    )

    train_state = dist_utils.safe_replicate(cfg, train_state)

    return cfg, statics, train_state, prng_key


if __name__ == "__main__":
    cfg = setup_config_dict()
    os.makedirs(cfg.logging.output_folder, exist_ok=True)

    # single-node data parallelism over all local devices
    cfg.training.ndevices = jax.device_count()
    print(f"Initialized with {cfg.training.ndevices} local devices")

    prng_key = jax.random.PRNGKey(cfg.training.seed)

    wandb.init(
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        name=cfg.logging.wandb_name,
        config=cfg.to_dict(),
        dir=cfg.logging.output_folder,
    )

    cfg, statics, train_state, prng_key = setup_state(cfg, prng_key)
    train_loop(cfg, statics, train_state, prng_key)
