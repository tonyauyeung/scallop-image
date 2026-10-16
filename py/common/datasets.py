"""
Nicholas M. Boffi
10/5/25

CelebA-64 data pipeline and Gaussian base density.
"""

import functools
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import jax_utils
from ml_collections import config_dict

# Avoid TensorFlow grabbing GPUs (JAX is the compute backend).
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


def unnormalize_image(image: jnp.ndarray):
    """Unnormalize an image from [-1, 1] to [0, 1] by scaling and clipping."""
    image = (image + 1) / 2
    image = jnp.clip(image, 0.0, 1.0)
    return image


def preprocess_celeb_a(
    image: tf.Tensor, *, dequantize: bool = False, seed: tf.Tensor = None
) -> tf.Tensor:
    """Map uint8 HWC to [-1, 1], center-crop to 140x140, area-resize to 64x64, then CHW.

    With ``dequantize`` uniform noise is added before scaling, (x + u) / 256,
    which is the convention used for bits-per-dimension evaluation. A stateless
    ``seed`` (shape (2,)) makes the noise reproducible per image.
    """
    image = tf.cast(image, tf.float32)
    if dequantize:
        if seed is None:
            noise = tf.random.uniform(tf.shape(image), 0.0, 1.0)
        else:
            noise = tf.random.stateless_uniform(tf.shape(image), seed=seed)
        image = (image + noise) / 256.0
    else:
        image = image / 255.0
    image = 2.0 * image - 1.0
    crop = tf.image.resize_with_crop_or_pad(image, 140, 140)
    crop64 = tf.image.resize(crop, [64, 64], method="area", antialias=True)
    return tf.transpose(crop64, [2, 0, 1])


def load_celeb_a(data_dir: str, split: str, shuffle_files: bool) -> tf.data.Dataset:
    """Load the raw CelebA split through TensorFlow Datasets."""
    return tfds.load(
        "celeb_a",
        split=split,
        shuffle_files=shuffle_files,
        data_dir=data_dir,
    )


def get_image_dataset(cfg: config_dict.ConfigDict):
    """Infinite, shuffled iterator of {"image": (bs, 3, 64, 64)} training batches."""
    ds = load_celeb_a(cfg.problem.dataset_location, "train", shuffle_files=True)
    ds = (
        ds.shuffle(10_000, reshuffle_each_iteration=True)
        .map(
            lambda x: {"image": preprocess_celeb_a(x["image"])},
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .repeat()
        .batch(cfg.optimization.bs)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )
    return ds


def _shard_batch(cfg: config_dict.ConfigDict, batch: Dict):
    """Shard a batch along the leading axis for multi-device training."""
    return {
        key: value.reshape((cfg.training.ndevices, -1, *value.shape[1:]))
        for key, value in batch.items()
    }


def prefetch_to_device(cfg: config_dict.ConfigDict, ds, buffer_size: int = 2):
    """Prefetch dataset batches onto device(s)."""
    # For single-device training, avoid device_put_sharded which expects
    # a leading device axis (len(shards) == ndevices).
    if buffer_size <= 0 or cfg.training.ndevices <= 1:
        return ds

    def _iterator():
        for batch in ds:
            yield _shard_batch(cfg, batch)

    return jax_utils.prefetch_to_device(_iterator(), buffer_size)


def setup_base(cfg: config_dict.ConfigDict, ex_input: jnp.ndarray) -> Callable:
    """Gaussian base density N(0, rescale^2 I)."""

    @functools.partial(jax.jit, static_argnums=(0,))
    def sample_rho0(bs: int, key: jnp.ndarray):
        rescale = jnp.asarray(cfg.network.rescale)
        return rescale * jax.random.normal(key, shape=(bs, *ex_input.shape))

    return sample_rho0


def base_log_prob(x: jnp.ndarray, rescale: float) -> jnp.ndarray:
    """Log density of N(0, rescale^2 I) for a batch of images (B, C, H, W)."""
    sigma = jnp.asarray(rescale, dtype=x.dtype)
    dims = tuple(range(1, x.ndim))
    quad = jnp.sum((x / sigma) ** 2, axis=dims)
    d = x[0].size
    log_norm = 0.5 * d * (jnp.log(2 * jnp.pi) + jnp.log(sigma**2))
    return -0.5 * quad - log_norm


def setup_target(cfg: config_dict.ConfigDict):
    """Set up the CelebA-64 training stream."""
    if cfg.problem.target != "celeb_a":
        raise ValueError(f"Unsupported target: {cfg.problem.target}")
    ds = get_image_dataset(cfg)
    print("Loaded image dataset.")

    # images in [-1, 1] have std ~0.5; the base and preconditioning use the same scale
    cfg.network.rescale = 0.5
    return cfg, ds
