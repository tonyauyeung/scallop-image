"""
Nicholas M. Boffi
10/5/25

Helper routines for neural network definitions.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import edm2_net as edm2_net


def setup_network(
    network_config: config_dict.ConfigDict,
) -> nn.Module:
    """Setup the neural network for the system.

    Args:
        config: Configuration dictionary.
    """
    if network_config.network_type == "edm2":
        return EDM2FlowMap(config=network_config)
    raise ValueError(f"Network type {network_config.network_type} not recognized.")


class EDM2FlowMap(nn.Module):
    """UNet architecture based on EDM2 (ported from flow-maps).

    Assumes inputs are images shaped as:
      - single: (C, H, W)
      - batch:  (B, C, H, W)
    """

    config: config_dict.ConfigDict

    def setup(self):
        self.one_hot_dim = (
            self.config.label_dim + 1 if self.config.use_cfg else self.config.label_dim
        )
        unet_kwargs = dict(self.config.unet_kwargs)
        unet_kwargs["predict_divergence"] = bool(self.config.predict_divergence)
        unet_kwargs["vectorized_divergence"] = bool(self.config.vectorized_divergence)
        self.net = edm2_net.PrecondFlowMap(
            img_resolution=self.config.img_resolution,
            img_channels=self.config.img_channels,
            label_dim=self.one_hot_dim,
            sigma_data=self.config.rescale,
            logvar_channels=self.config.logvar_channels,
            use_bfloat16=self.config.use_bfloat16,
            use_weight=self.config.use_weight,
            unet_kwargs=unet_kwargs,
        )

    def process_inputs(self, s: float, t: float, x: jnp.ndarray, label: float = None):
        s = jnp.asarray(s, dtype=jnp.float32)
        t = jnp.asarray(t, dtype=jnp.float32)

        if x.ndim == 3:
            x = x[None, ...]
            is_single = True
        elif x.ndim == 4:
            is_single = False
        else:
            raise ValueError(f"Unsupported x shape: {x.shape}")

        batch_size = x.shape[0]
        s = jnp.broadcast_to(s, (batch_size,))
        t = jnp.broadcast_to(t, (batch_size,))

        if label is not None:
            label = jnp.asarray(label)
            if label.ndim == 0:
                label = jnp.broadcast_to(label, (batch_size,))
            elif label.ndim == 1 and label.shape[0] != batch_size:
                label = jnp.broadcast_to(label, (batch_size,))
            label = jax.nn.one_hot(label, num_classes=self.one_hot_dim).reshape(
                (batch_size, -1)
            )

        return s, t, x, label, is_single

    def calc_weight(self, s: float, t: float) -> jnp.ndarray:
        s = jnp.asarray(s, dtype=jnp.float32)
        t = jnp.asarray(t, dtype=jnp.float32)
        return self.net.calc_weight(s, t)

    def calc_phi(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        init_weights: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        s, t, x, label, is_single = self.process_inputs(s, t, x, label)
        rslt = self.net.calc_phi(
            s, t, x, label, train, calc_weight, init_weights, return_div
        )

        div_st = None
        if calc_weight:
            if return_div and isinstance(rslt, tuple) and len(rslt) == 3:
                phi_st, div_st, logvar = rslt
            else:
                phi_st, logvar = rslt
            if return_div and div_st is not None:
                if is_single:
                    return phi_st[0], div_st[0], logvar[0]
                return phi_st, div_st, logvar
            if is_single:
                return phi_st[0], logvar[0]
            return phi_st, logvar

        if return_div and isinstance(rslt, tuple) and len(rslt) == 2:
            phi_st, div_st = rslt
        else:
            phi_st = rslt
        if return_div and div_st is not None:
            if is_single:
                return phi_st[0], div_st[0]
            return phi_st, div_st
        if is_single:
            return phi_st[0]
        return phi_st

    def calc_b(
        self,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        return_div: bool = False,
    ) -> jnp.ndarray:
        _, t, x, label, is_single = self.process_inputs(t, t, x, label)
        rslt = self.net.calc_b(t, x, label, train, calc_weight, return_div)

        div_st = None
        if calc_weight:
            if return_div and isinstance(rslt, tuple) and len(rslt) == 3:
                bt, div_st, logvar = rslt
            else:
                bt, logvar = rslt
            if return_div and div_st is not None:
                if is_single:
                    return bt[0], div_st[0], logvar[0]
                return bt, div_st, logvar
            if is_single:
                return bt[0], logvar[0]
            return bt, logvar

        if return_div and isinstance(rslt, tuple) and len(rslt) == 2:
            bt, div_st = rslt
        else:
            bt = rslt
        if return_div and div_st is not None:
            if is_single:
                return bt[0], div_st[0]
            return bt, div_st
        if is_single:
            return bt[0]
        return bt

    def __call__(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        return_X_and_phi: bool = False,
        init_weights: bool = False,
        return_div: bool = False,
    ):
        s, t, x, label, is_single = self.process_inputs(s, t, x, label)
        rslt = self.net(
            s, t, x, label, train, calc_weight, return_X_and_phi, init_weights, return_div
        )

        div_st = None
        if calc_weight:
            if return_div and isinstance(rslt, tuple) and len(rslt) == 3:
                Xst, div_st, logvar = rslt
            else:
                Xst, logvar = rslt
            if return_div and div_st is not None:
                if is_single:
                    return Xst[0], div_st[0], logvar[0]
                return Xst, div_st, logvar
            if is_single:
                return Xst[0], logvar[0]
            return Xst, logvar

        if return_X_and_phi:
            if return_div and isinstance(rslt, tuple) and len(rslt) == 3:
                Xst, phi_st, div_st = rslt
            else:
                Xst, phi_st = rslt
            if return_div and div_st is not None:
                if is_single:
                    return Xst[0], phi_st[0], div_st[0]
                return Xst, phi_st, div_st
            if is_single:
                return Xst[0], phi_st[0]
            return Xst, phi_st

        if return_div and isinstance(rslt, tuple) and len(rslt) == 2:
            Xst, div_st = rslt
        else:
            Xst = rslt
        if return_div and div_st is not None:
            if is_single:
                return Xst[0], div_st[0]
            return Xst, div_st
        if is_single:
            return Xst[0]
        return Xst