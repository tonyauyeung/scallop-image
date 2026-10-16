"""
Nicholas M. Boffi
10/5/25

Utilities for storing training state.
"""

from copy import deepcopy
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import flax.serialization as flax_serialization
import optax
from flax import struct
from flax.serialization import from_bytes
from flax.training import train_state
from ml_collections import config_dict

from . import flow_map, interpolant

Parameters = Dict[str, Any]


def _print_model_summary(
    net: nn.Module,
    params: Dict[str, Any],
    ex_input: jnp.ndarray,
    prng_key: jnp.ndarray,
) -> jnp.ndarray:
    """Print model structure and parameter count once."""
    if jax.process_index() != 0:
        return prng_key

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Number of parameters: {param_count}")

    ex_s = ex_t = 0.0
    ex_label = 0
    prng_key, skey = jax.random.split(prng_key)
    try:
        print("Model structure:")
        print(
            net.tabulate(
                {"params": prng_key, "constants": skey},
                ex_s,
                ex_t,
                ex_input,
                ex_label,
                train=False,
                init_weights=True,
            )
        )
    except Exception as exc:
        print(f"Model tabulate failed: {exc}")

    return prng_key


class EMATrainState(train_state.TrainState):
    """Train state including EMA parameters."""

    ema_params: Dict[float, Any] = struct.field(default_factory=dict)


class StaticArgs(NamedTuple):
    net: nn.Module
    schedule: optax.Schedule
    loss: Callable
    get_loss_fn_args: Callable
    train_step: Callable
    update_ema_params: Callable
    ds: Any
    interp: interpolant.Interpolant
    sample_rho0: Callable
    inception_fn: Callable = None  # For FID computation
    teacher_params: Optional[Dict[str, Any]] = None  # Optional external teacher


def restore_state(path: str, train_state: EMATrainState) -> EMATrainState:
    """Restore a serialized training state into ``train_state``.

    Leaves whose shapes do not match (e.g. a new divergence head on top of a
    pretrained LSD flow map) keep their initialized values.
    """
    with open(path, "rb") as f:
        raw_bytes = f.read()
    try:
        restored_state = from_bytes(train_state, raw_bytes)
        if not _has_shape_mismatch(train_state, restored_state):
            return restored_state
        print("Checkpoint has param shape mismatch; falling back to partial restore.")
    except ValueError:
        # expected when adding a divergence head on top of the LSD teacher
        print("Checkpoint and network differ (e.g. new divergence head); restoring matching params.")
    return _load_checkpoint_partially(train_state, raw_bytes)


def _has_shape_mismatch(target_state: EMATrainState, source_state: EMATrainState) -> bool:
    """Return True if any shared params/ema_params leaf has a different shape."""
    target = flax_serialization.to_state_dict(target_state)
    source = flax_serialization.to_state_dict(source_state)
    for field in ("params", "ema_params"):
        if field not in target or field not in source:
            continue
        flat_t = flax.traverse_util.flatten_dict(target[field])
        flat_s = flax.traverse_util.flatten_dict(source[field])
        for key, tv in flat_t.items():
            if key not in flat_s:
                continue
            sv = flat_s[key]
            t_shape = getattr(tv, "shape", None)
            s_shape = getattr(sv, "shape", None)
            if t_shape is not None and s_shape is not None:
                if tuple(t_shape) != tuple(s_shape):
                    return True
    return False


def _merge_trees_keep_target_structure(target, source, path=()):
    """Merge `source` into `target` but keep `target` structure.

    - Dicts: recurse on overlapping keys, keep target values for missing keys.
    - Arrays/leaves: replace only if shapes are equal (and dtype compatible).
    """
    # dict-like
    if isinstance(target, dict) and isinstance(source, dict):
        out = {}
        for k, tv in target.items():
            child_path = path + (str(k),)
            if k in source:
                out[k] = _merge_trees_keep_target_structure(
                    tv, source[k], path=child_path
                )
            else:
                out[k] = tv
        return out

    # leaf: try to replace if compatible
    try:
        t_shape = getattr(target, "shape", None)
        s_shape = getattr(source, "shape", None)
        if t_shape is not None and s_shape is not None and tuple(t_shape) == tuple(s_shape):
            # dtype check: allow same dtype or numpy/jax casts handled by flax.
            return source
    except Exception:
        pass
    return target


def _load_checkpoint_partially(
    train_state: EMATrainState,
    raw_bytes: bytes,
) -> EMATrainState:
    """Partially restore checkpoint into `train_state`.

    This restores matching leaves under `params` and `ema_params` (by shape),
    leaving unmatched/new parameters at their initialized values.
    """
    ckpt_state = flax_serialization.msgpack_restore(raw_bytes)
    target_state = flax_serialization.to_state_dict(train_state)

    for field in ("params", "ema_params"):
        if field in ckpt_state and field in target_state:
            target_state[field] = _merge_trees_keep_target_structure(
                target_state[field], ckpt_state[field]
            )

    return flax_serialization.from_state_dict(train_state, target_state)


def load_teacher_params(
    cfg: config_dict.ConfigDict,
    train_state: EMATrainState,
) -> Optional[Dict[str, Any]]:
    """Load teacher params from checkpoint if configured."""
    teacher_cfg = getattr(cfg, "teacher", None)
    teacher_path = ""
    teacher_ema_fac = None
    if teacher_cfg is not None:
        teacher_path = getattr(teacher_cfg, "load_path", "")
        teacher_ema_fac = getattr(teacher_cfg, "ema_fac", None)
    if not teacher_path:
        return None

    print(f"Loading teacher checkpoint from {teacher_path}.")
    teacher_state = restore_state(teacher_path, train_state)
    print("Loaded teacher checkpoint.")
    if teacher_ema_fac is not None:
        ema_params = teacher_state.ema_params.get(teacher_ema_fac, None)
        if ema_params is None:
            print(f"Warning: teacher EMA {teacher_ema_fac} not found, using params.")
            return teacher_state.params
        return ema_params
    return teacher_state.params


def setup_schedule(
    cfg: config_dict.ConfigDict,
) -> optax.Schedule:
    """Set up the learning rate schedule."""
    if cfg.optimization.schedule_type == "cosine":
        return optax.cosine_decay_schedule(
            init_value=cfg.optimization.learning_rate,
            decay_steps=cfg.optimization.decay_steps,
            alpha=0.0,
        )
    elif cfg.optimization.schedule_type == "sqrt":
        return lambda step: cfg.optimization.learning_rate / jnp.sqrt(
            jnp.maximum(step / cfg.optimization.decay_steps, 1.0)
        )
    elif cfg.optimization.schedule_type == "constant":
        return lambda step: cfg.optimization.learning_rate
    else:
        raise ValueError(f"Unknown schedule type: {cfg.optimization.schedule_type}")


def setup_optimizer(cfg: config_dict.ConfigDict):
    """Set up the optimizer."""
    schedule = setup_schedule(cfg)

    # optimizer mask for positional embeddings (which do not have a constants key)
    def mask_fn(variables):
        masks = {
            "params": jax.tree_util.tree_map(lambda _: True, variables["params"]),
        }
        if "constants" in variables:  # network has Fourier tables
            masks["constants"] = jax.tree_util.tree_map(
                lambda _: False, variables["constants"]
            )
        return masks

    # define optimizer
    tx = optax.masked(
        optax.chain(
            optax.clip_by_global_norm(cfg.optimization.clip),
            optax.radam(
                learning_rate=schedule,
                b1=getattr(cfg.optimization, "b1", 0.9),
                b2=getattr(cfg.optimization, "b2", 0.999),
                eps=getattr(cfg.optimization, "eps", 1e-8),
            ),
        ),
        mask_fn,
    )

    return tx, schedule


def setup_training_state(
    cfg: config_dict.ConfigDict,
    ex_input: jnp.ndarray,
    prng_key: jnp.ndarray,
) -> Tuple[EMATrainState, flow_map.FlowMap, optax.Schedule, jnp.ndarray]:
    """Load flax training state."""

    # Initialize flow map network
    net, params, prng_key = flow_map.initialize_flow_map(
        cfg.network, ex_input, prng_key
    )
    ema_params = {ema_fac: deepcopy(params) for ema_fac in cfg.training.ema_facs}
    prng_key = _print_model_summary(net, params, ex_input, prng_key)

    # define training state
    tx, schedule = setup_optimizer(cfg)
    train_state = EMATrainState.create(
        apply_fn=net.apply,
        params=params,
        ema_params=ema_params,
        tx=tx,
    )

    # load training state from checkpoint, if desired
    if cfg.network.load_path != "":
        print("Loading full training state checkpoint.")
        train_state = restore_state(cfg.network.load_path, train_state)
        print("Loaded training state checkpoint.")

        init_from_ema_fac = getattr(cfg.network, "init_from_ema_factor", None)
        if init_from_ema_fac is not None:
            ema_params = train_state.ema_params.get(init_from_ema_fac, None)
            if ema_params is None:
                print(
                    f"Warning: EMA factor {init_from_ema_fac} not found for init. "
                    "Keeping checkpoint params."
                )
            else:
                print(f"Initializing params from EMA {init_from_ema_fac}.")
                train_state = train_state.replace(params=ema_params)

        if cfg.network.reset_optimizer:
            print("Resetting optimizer state.")
            train_state = train_state.replace(
                opt_state=tx.init(train_state.params),
                step=0,
            )

    return train_state, net, schedule, prng_key

def load_params_for_eval(
    cfg: config_dict.ConfigDict,
    ckpt_path: str,
    ema_fac: Optional[float],
    prng_key: jnp.ndarray,
) -> Tuple[flow_map.FlowMap, Parameters]:
    """Load (EMA) params for evaluation; the architecture must match exactly."""
    net, params, _ = flow_map.initialize_flow_map(
        cfg.network, jnp.zeros(cfg.problem.image_dims), prng_key
    )
    tx, _ = setup_optimizer(cfg)
    train_state = EMATrainState.create(
        apply_fn=net.apply,
        params=params,
        ema_params={fac: params for fac in cfg.training.ema_facs},
        tx=tx,
    )
    mismatch_msg = (
        f"{ckpt_path} does not match the network config "
        "(check --div_head: none / vector / scalar)."
    )
    with open(ckpt_path, "rb") as f:
        raw_bytes = f.read()
    ckpt_keys = flax.traverse_util.flatten_dict(
        flax_serialization.msgpack_restore(raw_bytes)["params"]
    ).keys()
    target_keys = flax.traverse_util.flatten_dict(
        flax_serialization.to_state_dict(train_state)["params"]
    ).keys()
    if set(ckpt_keys) != set(target_keys):
        raise ValueError(mismatch_msg)
    restored = from_bytes(train_state, raw_bytes)
    if _has_shape_mismatch(train_state, restored):
        raise ValueError(mismatch_msg)
    if ema_fac is None:
        return net, restored.params
    return net, restored.ema_params[ema_fac]
