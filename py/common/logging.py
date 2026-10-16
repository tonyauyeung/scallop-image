"""
Nicholas M. Boffi
10/5/25

Code for basic wandb visualization and logging.
"""

import functools
import signal
import sys
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
import wandb
from flax.serialization import to_bytes
from jax.flatten_util import ravel_pytree
from matplotlib import pyplot as plt
from ml_collections import config_dict

from . import datasets, dist_utils, fid_utils, flow_map, likelihood, state_utils
from .losses import _hutchinson_divergence_with_phi

Parameters = Dict[str, Dict]


def _as_list(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def get_params_for_sampling(
    cfg: config_dict.ConfigDict,
    train_state: state_utils.EMATrainState,
    param_type: str = "visual",
) -> Parameters:
    """Select EMA (or instantaneous) params for "visual", "fid" or "bpd".

    Visual/BPD params are unreplicated; FID params stay replicated for pmap.
    """
    ema_factor = getattr(cfg.logging, f"{param_type}_ema_factor", None)
    if ema_factor is None:
        params = train_state.params
    elif ema_factor in train_state.ema_params:
        params = train_state.ema_params[ema_factor]
    else:
        print(f"Warning: EMA factor {ema_factor} not found. Using instantaneous params.")
        params = train_state.params

    if param_type == "fid":
        return params
    return dist_utils.safe_unreplicate(cfg, params)


def compute_fid_on_the_fly(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
    n_samples: int = 10000,
    batch_size: int = 256,
    n_steps: int = 8,
) -> Tuple[float, jnp.ndarray]:
    """FID of n-step flow map samples, sampled in parallel across devices."""
    fid_stats = np.load(cfg.logging.fid_stats_path)
    mu_real, sigma_real = fid_stats["mu"], fid_stats["sigma"]

    ndev = cfg.training.ndevices
    per_device_bs = -(-batch_size // ndev)
    sampler = flow_map.batch_sample if ndev == 1 else flow_map.pmap_batch_sample
    params = get_params_for_sampling(cfg, train_state, param_type="fid")

    stats = fid_utils.OnlineStats()
    while stats.n < n_samples:
        curr_bs = min(per_device_bs * ndev, n_samples - stats.n)
        padded_bs = -(-curr_bs // ndev) * ndev

        prng_key, sample_key = jax.random.split(prng_key)
        x0 = statics.sample_rho0(padded_bs, sample_key)
        if ndev > 1:
            x0 = x0.reshape(ndev, padded_bs // ndev, *cfg.problem.image_dims)
        imgs = sampler(train_state.apply_fn, params, x0, n_steps, None)
        imgs = imgs.reshape(padded_bs, *cfg.problem.image_dims)[:curr_bs]
        imgs = jnp.clip(imgs, -1, 1).transpose(0, 2, 3, 1)  # NCHW -> NHWC

        stats.update(np.asarray(fid_utils.resize_and_incept(imgs, statics.inception_fn)))

    fid_score = fid_utils.fid_from_stats(stats.mu, stats.sigma, mu_real, sigma_real)
    return fid_score, prng_key


def compute_bpd_on_batch(
    cfg: config_dict.ConfigDict,
    train_state: state_utils.EMATrainState,
    x1batch: jnp.ndarray,
    n_steps_list: Tuple[int, ...],
) -> Dict[str, float]:
    """Per-step BPD of a training batch using the divergence head.

    Training images are not dequantized, so this is a cheap proxy for the test
    BPD reported by ``launchers/eval_bpd.py``.
    """
    x1batch = dist_utils.unreplicate_batch(cfg, x1batch)
    x1batch = x1batch[: cfg.logging.bpd_batch_size]
    params = get_params_for_sampling(cfg, train_state, param_type="bpd")
    d = int(np.prod(cfg.problem.image_dims))

    metrics = {}
    for n_steps in n_steps_list:
        logp = likelihood.inverse_logp_div_head(
            train_state.apply_fn,
            params,
            x1batch,
            int(n_steps),
            cfg.network.rescale,
            cfg.network.divergence_rescale,
        )
        bpd, _ = likelihood.logp_to_bpd(logp, d, dequantize=True)
        metrics[f"bpd_{n_steps}_steps"] = float(jnp.mean(bpd))
    return metrics


def _save_ckpt_on_signal(
    cfg: config_dict.ConfigDict, train_state: state_utils.EMATrainState, *_
) -> None:
    save_state(train_state, cfg)
    sys.exit(0)


def register_signal_handlers(
    cfg: config_dict.ConfigDict,
    train_state: state_utils.EMATrainState,
) -> None:
    """Drop a checkpoint on SIGTERM or SIGINT."""
    handler = functools.partial(_save_ckpt_on_signal, cfg, train_state)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def save_state(
    train_state: state_utils.EMATrainState,
    cfg: config_dict.ConfigDict,
) -> None:
    """Save flax training state as <output_name>_<step // save_freq>.pkl."""
    step = int(dist_utils.safe_index(cfg, train_state.step))
    path = f"{cfg.logging.output_folder}/{cfg.logging.output_name}_{step // cfg.logging.save_freq}.pkl"
    with open(path, "wb") as f:
        state = jax.device_get(dist_utils.safe_unreplicate(cfg, train_state))
        f.write(to_bytes(state))


@jax.jit
def compute_grad_norm(grads: Dict) -> float:
    """Computes the norm of the gradient, treated as a flattened PyTree."""
    flat_params = ravel_pytree(grads)[0]
    return jnp.linalg.norm(flat_params)


def log_metrics(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    grads: jnp.ndarray,
    loss_value: float,
    loss_metrics: Dict[str, jnp.ndarray],
    loss_fn_args: Tuple,
    prng_key: jnp.ndarray,
    step_time: float,
) -> jnp.ndarray:
    """Log some metrics to wandb, make a figure, and checkpoint the parameters."""

    grads = dist_utils.safe_unreplicate(cfg, grads)
    loss_value = dist_utils.safe_index(cfg, jnp.array(loss_value))
    loss_metrics = dist_utils.safe_unreplicate(cfg, loss_metrics)
    step = int(dist_utils.safe_index(cfg, train_state.step))

    metrics = {
        "loss": loss_value,
        "grad": compute_grad_norm(grads),
        "learning_rate": statics.schedule(step),
        "step_time": step_time,
    }
    metrics.update(loss_metrics)

    # FID of few-step samples
    if cfg.logging.fid_freq > 0 and step % cfg.logging.fid_freq == 0 and step > 0:
        try:
            for n_steps in _as_list(cfg.logging.fid_n_steps):
                fid_score, prng_key = compute_fid_on_the_fly(
                    cfg,
                    statics,
                    train_state,
                    prng_key,
                    n_samples=cfg.logging.fid_n_samples,
                    batch_size=cfg.logging.fid_batch_size,
                    n_steps=n_steps,
                )
                metrics[f"fid_{n_steps}_steps"] = fid_score
        except Exception as e:
            print(f"Warning: FID computation failed: {e}")

    # BPD of the current batch with the divergence head
    has_div_head = cfg.network.divergence_mode != "pretrain_flowmap"
    if (
        has_div_head
        and cfg.logging.bpd_freq > 0
        and step % cfg.logging.bpd_freq == 0
        and step > 0
    ):
        try:
            # loss_fn_args = (teacher_params, x0, x1, label, s, t, dropout_keys)
            metrics.update(
                compute_bpd_on_batch(
                    cfg, train_state, loss_fn_args[2], tuple(_as_list(cfg.logging.bpd_n_steps))
                )
            )
        except Exception as e:
            print(f"Warning: BPD computation failed: {e}")

    if step % cfg.logging.visual_freq == 0:
        make_loss_fn_args_plot(cfg, loss_fn_args, wandb_step=step)
        prng_key = make_image_plot(cfg, statics, train_state, prng_key, wandb_step=step)
        if has_div_head:
            prng_key, div_metrics = make_stratified_div_error(cfg, statics, train_state, prng_key)
            metrics.update(div_metrics)

    wandb.log(metrics, step=step)
    if step % cfg.logging.save_freq == 0:
        save_state(train_state, cfg)

    return prng_key


def make_image_plot(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
    wandb_step: int = 0,
) -> jnp.ndarray:
    """Log grids of n-step samples from the same noise."""
    params = get_params_for_sampling(cfg, train_state, param_type="visual")

    plt.close("all")
    sns.set_palette("deep")
    fontsize = 12.5
    n_images = cfg.logging.plot_bs
    nrows = int(np.floor(np.sqrt(n_images)))
    ncols = -(-n_images // nrows)

    x0s = statics.sample_rho0(n_images, prng_key)
    prng_key = jax.random.split(prng_key)[0]

    for n_steps in _as_list(cfg.logging.visual_n_steps):
        xhats = flow_map.batch_sample(train_state.apply_fn, params, x0s, n_steps, None)
        xhats = np.asarray(datasets.unnormalize_image(xhats)).transpose(0, 2, 3, 1)

        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(ncols, nrows),
            constrained_layout=True,
            squeeze=False,
        )
        title = f"{n_steps}-step"
        fig.suptitle(title, fontsize=fontsize)
        for index, ax in enumerate(axs.ravel()):
            ax.set_axis_off()
            if index < n_images:
                ax.imshow(xhats[index])

        wandb.log({title: wandb.Image(fig)}, step=wandb_step)

    return prng_key


def make_loss_fn_args_plot(
    cfg: config_dict.ConfigDict,
    loss_fn_args: Tuple,
    wandb_step: int = 0,
) -> None:
    """Scatter the sampled (s, t) pairs."""
    # loss_fn_args = (teacher_params, x0, x1, label, s, t, dropout_keys)
    sbatch = np.asarray(dist_utils.unreplicate_batch(cfg, loss_fn_args[4]))
    tbatch = np.asarray(dist_utils.unreplicate_batch(cfg, loss_fn_args[5]))

    plt.close("all")
    fontsize = 12.5
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_aspect("equal")
    ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.set_title(r"$(s, t)$", fontsize=fontsize)
    ax.scatter(sbatch, tbatch, s=0.1, alpha=0.5, marker="o")

    wandb.log({"loss_fn_args": wandb.Image(fig)}, step=wandb_step)


def make_stratified_div_error(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
    n_times: int = 10,
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """Relative L2 error of the divergence head against a Hutchinson estimate, per t."""
    prng_key, sample_key = jax.random.split(prng_key)
    x1s = next(statics.ds)["image"]
    x1s = dist_utils.unreplicate_batch(cfg, x1s)
    x0s = statics.sample_rho0(x1s.shape[0], sample_key)
    params = dist_utils.safe_unreplicate(cfg, train_state.params)

    @jax.jit
    def compute_step(t_curr, step_key):
        xt = statics.interp.calc_It(t_curr, x0s, x1s)

        def div_wrapper(x_in, rng_in):
            _, div = _hutchinson_divergence_with_phi(
                params, t_curr, x_in, None, rng_in, X=statics.net, method="calc_b"
            )
            return div

        step_keys = jax.random.split(step_key, xt.shape[0])
        div_hat = jax.vmap(div_wrapper)(xt, step_keys)
        _, div_tt = statics.net.apply(
            params, t_curr, xt, None, train=False, method="calc_b", return_div=True
        )
        div_hat = div_hat.reshape((div_hat.shape[0], -1)).sum(axis=-1)
        div_tt = div_tt.reshape((div_tt.shape[0], -1)).sum(axis=-1)
        div_tt = div_tt * cfg.network.divergence_rescale

        l2_error = jnp.linalg.norm(div_hat - div_tt)
        l2_magnitude = jnp.linalg.norm(div_tt)
        return l2_error, l2_magnitude

    metrics = {}
    for t_curr in np.linspace(0.0, 0.999, n_times):
        prng_key, step_key = jax.random.split(prng_key)
        l2_error, l2_magnitude = compute_step(t_curr, step_key)
        prefix = "eval/stratified_div"
        metrics[f"{prefix}/l2_error/t={t_curr:.2f}"] = float(l2_error)
        metrics[f"{prefix}/l2_magnitude/t={t_curr:.2f}"] = float(l2_magnitude)
        metrics[f"{prefix}/l2_relative_error/t={t_curr:.2f}"] = float(
            l2_error / l2_magnitude
        )
    return prng_key, metrics
