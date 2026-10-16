"""
Nicholas M. Boffi
10/5/25

Loss functions for learning.

Three training modes are supported (``cfg.network.divergence_mode``):
  - ``pretrain_flowmap``: LSD flow map without a divergence head (the teacher).
  - ``hutchinson``: F2D2, divergence target from Hutchinson's trace estimator.
  - ``dlogpdt``: SCALLOP, Hutchinson-free divergence target.
"""

import functools
import math
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import flow_map as flow_map
from . import interpolant as interpolant
from . import loss_args

Parameters = Dict[str, Dict]

DIVERGENCE_MODES = ("pretrain_flowmap", "hutchinson", "dlogpdt")


# Coded for xt = alpha(t) * x0 + beta(t) * x1 with x0 = sigma * z, where
# alpha(t) = 1 - t
# beta(t) = t


def mean_reduce(func):
    """
    A decorator that computes the mean of the output of the decorated function.
    Designed to be used on functions that are already batch-processed (e.g., with jax.vmap).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        batched_outputs = func(*args, **kwargs)
        return jax.tree_util.tree_map(lambda x: jnp.mean(x), batched_outputs)

    return wrapper


def _hutchinson_divergence_with_phi(
    params: Parameters,
    t: float,
    x: jnp.ndarray,
    label: jnp.ndarray,
    rng: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    method: str = "calc_b",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Estimate div(phi) with Hutchinson's estimator and return phi.

    ``x`` is a single (C, H, W) image. With a vectorized divergence head the
    per-pixel contributions eps * (J^T eps) are returned; otherwise their sum.
    """
    eps_key = rng["dropout"] if isinstance(rng, dict) else rng
    eps = jax.random.normal(eps_key, shape=x.shape)

    def phi_fn(x_in):
        return X.apply(params, t, x_in, label, train=False, method=method, rngs=rng)

    phi, vjp_fn = jax.vjp(phi_fn, x)
    div_hat = vjp_fn(eps)[0] * eps
    if not X.config.vectorized_divergence:
        div_hat = jnp.sum(div_hat)
    return phi, div_hat


def _calc_divergence_with_phi(
    params: Parameters,
    t: float,
    xt: jnp.ndarray,
    x0: jnp.ndarray,  # noise = sigma * z
    x1: jnp.ndarray,  # clean data
    label: jnp.ndarray,
    rng: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    method: str = "calc_b",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return the teacher velocity and the divergence regression target."""
    if X.config.divergence_mode == "hutchinson":
        return _hutchinson_divergence_with_phi(
            params=params, t=t, x=xt, label=label, rng=rng, X=X, method=method
        )

    if X.config.divergence_mode != "dlogpdt":
        raise ValueError(f"Invalid divergence_mode: {X.config.divergence_mode}")

    # Hutchinson-free target: E[div_hat | x_t] = div b(t, x_t).
    cond_vel = x1 - x0
    phi = X.apply(params, t, xt, label, train=False, method=method, rngs=rng)
    rescale = jnp.asarray(X.config.rescale, dtype=jnp.float32)
    denominator = jnp.maximum(1.0 - t, 1e-3)
    div_hat = ((phi - cond_vel) * x0 / rescale**2 - 1.0) / denominator
    if not X.config.vectorized_divergence:
        div_hat = jnp.sum(div_hat)
    return phi, div_hat


def _get_div_loss_scaling(X: flow_map.FlowMap) -> float:
    if X.config.vectorized_divergence:
        return float(math.prod(X.config.input_dims))
    return 1.0


def _calc_div_loss(
    div_pred: jnp.ndarray,
    div_target: jnp.ndarray,
    t: float,
    X: flow_map.FlowMap,
) -> jnp.ndarray:
    if X.config.div_loss_weighting_type == "uniform":
        div_loss = jnp.sum((div_pred - div_target) ** 2)
    elif X.config.div_loss_weighting_type == "square":
        div_loss = jnp.sum((div_pred - div_target) ** 2) * (1.0 - t) ** 2
    elif X.config.div_loss_weighting_type == "linear":
        # (1 - t) * |div_pred - div_target|^2 up to a target-only constant, which
        # avoids the 1 / (1 - t) blow-up of the SCALLOP target near t = 1.
        div_loss = jnp.sum(div_pred**2) * (1.0 - t) - 2.0 * jnp.sum(
            div_pred * div_target
        ) * (1.0 - t)
    else:
        raise ValueError(
            f"Invalid div_loss_weighting_type: {X.config.div_loss_weighting_type}"
        )
    return div_loss / _get_div_loss_scaling(X)


def diagonal_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute the diagonal (s = t) term of the loss."""
    It = interp.calc_It(t, x0, x1)
    is_pretraining = X.config.divergence_mode == "pretrain_flowmap"

    bt_rslt = X.apply(
        params,
        t,
        It,
        label,
        train=True,
        method="calc_b",
        rngs=rng,
        return_div=not is_pretraining,
    )

    if is_pretraining:
        # flow matching against the interpolant velocity
        bt = bt_rslt
        bt_teacher = interp.calc_It_dot(t, x0, x1)
        div_loss = jnp.array(0.0, dtype=jnp.float32)
    else:
        # distill the teacher velocity and regress the divergence target
        bt, div_tt = bt_rslt
        bt_teacher, div_hat = _calc_divergence_with_phi(
            params=teacher_params,
            t=t,
            xt=It,
            x0=x0,
            x1=x1,
            label=label,
            rng=rng,
            X=X,
            method="calc_b",
        )
        bt_teacher = jax.lax.stop_gradient(bt_teacher)
        div_hat = jax.lax.stop_gradient(div_hat) / X.config.divergence_rescale
        div_loss = _calc_div_loss(div_pred=div_tt, div_target=div_hat, t=t, X=X)

    base_velocity_loss = jnp.sum((bt - bt_teacher) ** 2)
    velocity_loss = base_velocity_loss + div_loss

    # Diagonal uses s=t
    weight_tt = X.apply(params, t, t, method="calc_weight")
    loss_value = jnp.exp(-weight_tt) * velocity_loss + weight_tt
    metrics = {
        "diag/velocity_loss": jax.lax.stop_gradient(base_velocity_loss),
        "diag/div_loss": jax.lax.stop_gradient(div_loss),
    }
    return loss_value, metrics


def lsd_term(
    params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    stopgrad_type: str,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute the Lagrangian self-distillation (LSD) term of the loss."""
    if stopgrad_type != "convex":
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    Is = interp.calc_It(s, x0, x1)
    is_pretraining = X.config.divergence_mode == "pretrain_flowmap"

    # X_{s,t}(I_s) and its time derivative (plus the divergence head D_{s,t})
    Xst_Is, dt_Xst = X.apply(
        params,
        s,
        t,
        Is,
        label,
        train=False,
        method="partial_t",
        rngs=rng,
        return_div=not is_pretraining,
    )
    if not is_pretraining:
        Xst_Is, D_st = Xst_Is
        dt_Xst, dt_Dst = dt_Xst

    # convex stopgrad: the velocity at the transported point is the target
    Xst_Is = jax.lax.stop_gradient(Xst_Is)
    b_eval = X.apply(
        params,
        t,
        Xst_Is,
        label,
        train=False,
        method="calc_b",
        rngs=rng,
        return_div=not is_pretraining,
    )
    b_eval = jax.lax.stop_gradient(b_eval)

    if is_pretraining:
        b_eval_v = b_eval
        lsd_div_loss = jnp.array(0.0, dtype=jnp.float32)
    else:
        b_eval_v, b_eval_div = b_eval
        # d/dt [(t - s) D_{s,t}] = D_{s,t} + (t - s) d/dt D_{s,t}
        A_dot = D_st + (t - s) * jax.lax.stop_gradient(dt_Dst)
        lsd_div_loss = jnp.sum((A_dot - b_eval_div) ** 2) / _get_div_loss_scaling(X)

    lsd_v_loss = jnp.sum((b_eval_v - dt_Xst) ** 2)
    lsd_loss = lsd_v_loss + lsd_div_loss

    weight_st = X.apply(params, s, t, method="calc_weight")
    loss_value = jnp.exp(-weight_st) * lsd_loss + weight_st
    metrics = {
        "lsd/lsd_v_loss": jax.lax.stop_gradient(lsd_v_loss),
        "lsd/lsd_div_loss": jax.lax.stop_gradient(lsd_div_loss),
    }
    return loss_value, metrics


def setup_loss(
    cfg: config_dict.ConfigDict, net: flow_map.FlowMap, interp: interpolant.Interpolant
) -> Callable:
    """Setup the loss function."""
    if cfg.training.loss_type != "lsd":
        raise ValueError(f"Unknown loss_type: {cfg.training.loss_type}")
    if cfg.network.divergence_mode not in DIVERGENCE_MODES:
        raise ValueError(f"Invalid divergence_mode: {cfg.network.divergence_mode}")

    print(f"Setting up loss: {cfg.training.loss_type}")
    print(f"Stopgrad type: {cfg.training.stopgrad_type}")
    print(f"Divergence mode: {cfg.network.divergence_mode}")

    @mean_reduce
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0))
    def diagonal_only_loss(params, teacher_params, x0, x1, label, t, dropout_keys):
        return diagonal_term(
            params,
            teacher_params,
            x0,
            x1,
            label,
            t,
            {"dropout": dropout_keys},
            interp=interp,
            X=net,
        )

    @mean_reduce
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0))
    def offdiagonal_only_loss(params, x0, x1, label, s, t, dropout_keys):
        return lsd_term(
            params,
            x0,
            x1,
            label,
            s,
            t,
            {"dropout": dropout_keys},
            interp=interp,
            X=net,
            stopgrad_type=cfg.training.stopgrad_type,
        )

    def loss(
        params,
        teacher_params,
        x0,
        x1,
        label,
        s,
        t,
        dropout_keys,
    ):
        """Split batch into diagonal and off-diagonal portions."""
        total_bs = x0.shape[0]
        diag_bs, offdiag_bs = loss_args._get_diag_offdiag_bs(cfg, total_bs)

        total_loss = 0.0
        total_metrics = {}

        def _add_metrics(new_metrics, weight):
            for key, value in new_metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value * weight

        if diag_bs > 0:
            label_diag = None if label is None else label[:diag_bs]
            diag_loss, diag_metrics = diagonal_only_loss(
                params,
                teacher_params,
                x0[:diag_bs],
                x1[:diag_bs],
                label_diag,
                t[:diag_bs],
                dropout_keys[:diag_bs],
            )
            total_loss += diag_loss * diag_bs
            _add_metrics(diag_metrics, diag_bs)

        if offdiag_bs > 0:
            label_offdiag = None if label is None else label[diag_bs:]
            offdiag_loss, offdiag_metrics = offdiagonal_only_loss(
                params,
                x0[diag_bs:],
                x1[diag_bs:],
                label_offdiag,
                s[diag_bs:],
                t[diag_bs:],
                dropout_keys[diag_bs:],
            )
            total_loss += offdiag_loss * offdiag_bs
            _add_metrics(offdiag_metrics, offdiag_bs)

        total_loss = total_loss / total_bs
        total_metrics = jax.tree_util.tree_map(
            lambda x: jax.lax.stop_gradient(x / total_bs), total_metrics
        )
        return total_loss, total_metrics

    return loss
