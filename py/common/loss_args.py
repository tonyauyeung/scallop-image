"""
Nicholas M. Boffi
10/5/25

Code for setting up arguments for loss functions.
"""

import functools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import dist_utils, state_utils


def _sample_diagonal(
    key: jnp.ndarray, bs: int, tmin: float, tmax: float, distr: str = "uniform"
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample points on the diagonal (s=t)."""
    if distr == "uniform":
        s = jax.random.uniform(key, shape=(bs,), minval=tmin, maxval=tmax)
    elif distr == "linear":
        # density proportional to 1 / (1 - t) on [0, 1 - epsilon]
        epsilon = max(1.0 - tmax, 1e-3)
        u = jax.random.uniform(key, shape=(bs,))
        logs = (1.0 - u) * jnp.log(epsilon)
        s = 1.0 - jnp.exp(logs)
    else:
        raise ValueError(f"Unknown time distribution: {distr}")
    return s, s


def _sample_square(
    key1: jnp.ndarray,
    key2: jnp.ndarray,
    bs: int,
    tmin: float,
    tmax: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample (s, t) uniformly on [tmin, tmax]^2 (forward and backward jumps)."""
    s = jax.random.uniform(key1, shape=(bs,), minval=tmin, maxval=tmax)
    t = jax.random.uniform(key2, shape=(bs,), minval=tmin, maxval=tmax)
    return s, t


def _get_diag_offdiag_bs(cfg: config_dict.ConfigDict, bs: int) -> Tuple[int, int]:
    """Get diagonal and off-diagonal batch sizes."""
    diag_bs = max(1, int(bs * cfg.optimization.diag_fraction))
    offdiag_bs = bs - diag_bs
    return diag_bs, offdiag_bs


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def get_loss_fn_args_randomness(
    prng_key: jnp.ndarray,
    cfg: config_dict.ConfigDict,
    sample_rho0: Callable,
    diag_bs: int,
    offdiag_bs: int,
) -> Tuple:
    """Draw random values needed for each loss function iteration."""
    tkey, skey, _, x0key, tkey2 = jax.random.split(prng_key, num=5)
    bs = cfg.optimization.bs
    tmin = cfg.training.tmin
    tmax = cfg.training.tmax
    x0batch = sample_rho0(bs, x0key)

    s_diag, t_diag = _sample_diagonal(
        skey, diag_bs, tmin, tmax, distr=cfg.training.time_distr
    )
    if offdiag_bs > 0:
        s_offdiag, t_offdiag = _sample_square(tkey, tkey2, offdiag_bs, tmin, tmax)
        sbatch = jnp.concatenate([s_diag, s_offdiag])
        tbatch = jnp.concatenate([t_diag, t_offdiag])
    else:
        sbatch, tbatch = s_diag, t_diag

    dropout_keys = jax.random.split(tkey, num=bs).reshape((bs, -1))
    prng_key = jax.random.split(dropout_keys[0])[0]

    if getattr(cfg.training, "batch_is_sharded", False):

        def _shard_batch(x):
            return x.reshape((cfg.training.ndevices, -1, *x.shape[1:]))

        x0batch = _shard_batch(x0batch)
        sbatch = _shard_batch(sbatch)
        tbatch = _shard_batch(tbatch)
        dropout_keys = _shard_batch(dropout_keys)
    return tbatch, sbatch, x0batch, dropout_keys, prng_key


def get_loss_fn_args(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
) -> Tuple:
    """Assemble (teacher_params, x0, x1, label, s, t, dropout_keys)."""
    diag_bs, offdiag_bs = _get_diag_offdiag_bs(cfg, cfg.optimization.bs)
    tbatch, sbatch, x0batch, dropout_keys, prng_key = get_loss_fn_args_randomness(
        prng_key,
        cfg,
        statics.sample_rho0,
        diag_bs,
        offdiag_bs,
    )

    # unconditional generation: labels are never used
    x1batch = next(statics.ds)["image"]
    label_batch = None

    # the external teacher (pretrained LSD flow map, replicated once at setup)
    # supplies the diagonal velocity; LSD pretraining passes the current params unused
    teacher_params = statics.teacher_params
    if teacher_params is None:
        teacher_params = train_state.params

    loss_fn_args = (x0batch, x1batch, label_batch, sbatch, tbatch, dropout_keys)
    loss_fn_args = dist_utils.replicate_loss_fn_args(cfg, loss_fn_args)
    return (teacher_params, *loss_fn_args), prng_key
