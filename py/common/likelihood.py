"""
Log-likelihood and bits-per-dimension (BPD) for CelebA-64.

Data x_1 is mapped back to noise x_0 by integrating from t = 1 to t = 0 with
equally spaced steps, accumulating the change in log density:

    log p_1(x_1) = log p_0(x_0) - int_0^1 div b_t(x_t) dt.

Two estimators of the divergence are provided:
  - ``inverse_logp_div_head``: a likelihood flow map (F2D2 / SCALLOP) jumps
    t -> t + dt with phi_{t, t+dt} and predicts the averaged divergence with
    its divergence head; any number of steps (1, 2, 4, 8, ...).
  - ``inverse_logp_hutchinson``: Euler integration of the instantaneous
    velocity b_t = phi_{t, t} with Hutchinson's trace estimator; used for the
    teacher reference (1024 steps in the paper).
"""

import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from .datasets import base_log_prob


def _sum_non_batch(x: jnp.ndarray) -> jnp.ndarray:
    return x.reshape((x.shape[0], -1)).sum(axis=-1)


@functools.partial(jax.jit, static_argnums=(0, 3))
def inverse_logp_div_head(
    apply_fn: Callable,
    params: Dict,
    x1: jnp.ndarray,
    n_steps: int,
    rescale: float,
    divergence_rescale: float,
) -> jnp.ndarray:
    """log p_1(x1) using the flow map and its divergence head."""
    dt = -1.0 / n_steps
    t_curr = jnp.ones((x1.shape[0],), dtype=jnp.float32)
    delta_logp = jnp.zeros((x1.shape[0],), dtype=jnp.float32)

    def body(_, state):
        t_curr, x, delta_logp = state
        t_next = t_curr + dt
        phi, div = apply_fn(
            params,
            t_curr,
            t_next,
            x,
            None,
            train=False,
            method="calc_phi",
            return_div=True,
        )
        # vectorized heads predict per-pixel contributions; scalar heads predict div / d
        div = _sum_non_batch(div) * divergence_rescale
        x = x + dt * phi
        delta_logp = delta_logp - dt * div
        return t_next, x, delta_logp

    _, x0, delta_logp = jax.lax.fori_loop(0, n_steps, body, (t_curr, x1, delta_logp))
    return base_log_prob(x0, rescale) - delta_logp


@functools.partial(jax.jit, static_argnums=(0, 3, 6))
def inverse_logp_hutchinson(
    apply_fn: Callable,
    params: Dict,
    x1: jnp.ndarray,
    n_steps: int,
    rescale: float,
    rng_key: jnp.ndarray,
    n_probes: int = 1,
) -> jnp.ndarray:
    """log p_1(x1) using Euler steps of the velocity b_t and Hutchinson probes."""
    dt = -1.0 / n_steps
    t_curr = jnp.ones((x1.shape[0],), dtype=jnp.float32)
    delta_logp = jnp.zeros((x1.shape[0],), dtype=jnp.float32)

    def body(_, state):
        t_curr, x, delta_logp, rng_key = state
        t_next = t_curr + dt

        def b_fn(x_in):
            return apply_fn(params, t_curr, x_in, None, train=False, method="calc_b")

        b, vjp_fn = jax.vjp(b_fn, x)
        div = jnp.zeros((x.shape[0],), dtype=jnp.float32)
        for _ in range(n_probes):
            rng_key, eps_key = jax.random.split(rng_key)
            eps = jax.random.rademacher(eps_key, shape=x.shape, dtype=x.dtype)
            div = div + _sum_non_batch(vjp_fn(eps)[0] * eps)
        div = div / n_probes

        x = x + dt * b
        delta_logp = delta_logp - dt * div
        return t_next, x, delta_logp, rng_key

    _, x0, delta_logp, _ = jax.lax.fori_loop(
        0, n_steps, body, (t_curr, x1, delta_logp, rng_key)
    )
    return base_log_prob(x0, rescale) - delta_logp


def logp_to_bpd(
    logp: jnp.ndarray, d: int, *, dequantize: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert log p of [-1, 1]-scaled images to BPD of the 8-bit data.

    Mapping [0, 1] -> [-1, 1] contributes d * log 2; uniform dequantization of
    256 bins contributes -d * log 256.
    """
    total_logp = logp + d * jnp.log(2.0)
    if dequantize:
        total_logp = total_logp - d * jnp.log(256.0)
    bpd = -total_logp / (d * jnp.log(2.0))
    return bpd, total_logp
