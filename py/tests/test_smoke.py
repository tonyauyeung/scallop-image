"""CPU smoke test: LSD pretraining -> SCALLOP / F2D2 fine-tuning -> BPD / FID evaluation.

Uses a tiny EDM2 UNet on random 16x16 images, so it checks plumbing, not quality.
Run with:  pytest py/tests -q
"""

import types

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import pytest
import wandb

import common.datasets as datasets
import common.fid_utils as fid_utils
import common.likelihood as likelihood
import common.losses as losses
import common.state_utils as state_utils
from configs import celeba64
from launchers import learn

IMAGE_DIMS = (3, 16, 16)
TINY_UNET = {
    "model_channels": 16,
    "channel_mult": [1, 2],
    "num_blocks": 1,
    "attn_resolutions": [8],
    "block_kwargs": {"dropout": 0.0},
}


def _tiny(cfg, output_folder):
    cfg.problem.image_dims = IMAGE_DIMS
    cfg.problem.d = int(np.prod(IMAGE_DIMS))
    cfg.network.img_resolution = IMAGE_DIMS[1]
    cfg.network.input_dims = IMAGE_DIMS
    cfg.network.unet_kwargs = TINY_UNET
    cfg.network.use_bfloat16 = False
    cfg.optimization.bs = 8
    cfg.optimization.total_steps = 3
    cfg.training.ndevices = 1
    cfg.logging.output_folder = str(output_folder)
    cfg.logging.save_freq = 1
    cfg.logging.fid_freq = 0
    cfg.logging.bpd_freq = 1
    cfg.logging.bpd_n_steps = [1, 2]
    cfg.logging.bpd_batch_size = 4
    cfg.logging.visual_freq = 2
    cfg.logging.visual_n_steps = [1, 2]
    cfg.logging.plot_bs = 4
    return cfg


@pytest.fixture(scope="module", autouse=True)
def fake_celeba():
    """Replace the CelebA stream by random images (no download needed)."""

    def _fake_dataset(cfg):
        rng = np.random.default_rng(0)
        while True:
            x = rng.uniform(-1, 1, size=(cfg.optimization.bs, *IMAGE_DIMS))
            yield {"image": x.astype(np.float32)}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(datasets, "get_image_dataset", _fake_dataset)
        wandb.init(mode="disabled")
        yield


def _train(cfg):
    prng_key = jax.random.PRNGKey(cfg.training.seed)
    cfg, statics, train_state, prng_key = learn.setup_state(cfg, prng_key)
    learn.train_loop(cfg, statics, train_state, prng_key)
    return cfg, statics


def _final_ckpt(cfg):
    return f"{cfg.logging.output_folder}/{cfg.logging.output_name}_{cfg.optimization.total_steps}.pkl"


@pytest.fixture(scope="module")
def lsd_ckpt(tmp_path_factory):
    out = tmp_path_factory.mktemp("lsd")
    cfg = _tiny(celeba64.get_config("", str(out), "pretrain_flowmap"), out)
    cfg.optimization.diag_fraction_switch = (2, 0.5)  # exercise the batch-split switch
    switches = []
    switch_fn = learn.switch_diag_fraction

    def _record(cfg, statics, diag_fraction):
        switches.append(diag_fraction)
        return switch_fn(cfg, statics, diag_fraction)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(learn, "switch_diag_fraction", _record)
        cfg, _ = _train(cfg)
    assert switches == [0.5]
    return _final_ckpt(cfg)


@pytest.mark.parametrize("mode", ["dlogpdt", "hutchinson"])
def test_finetune_and_eval(mode, lsd_ckpt, tmp_path):
    cfg = _tiny(celeba64.get_config("", str(tmp_path), mode), tmp_path)
    cfg.network.load_path = lsd_ckpt
    cfg.teacher.load_path = lsd_ckpt
    cfg.network.div_loss_weighting_type = "linear" if mode == "dlogpdt" else "uniform"
    cfg, _ = _train(cfg)
    ckpt = _final_ckpt(cfg)

    # strict eval loading: the architecture must match the checkpoint
    eval_cfg = _tiny(celeba64.get_eval_config("", "vector"), tmp_path)
    net, params = state_utils.load_params_for_eval(eval_cfg, ckpt, 0.9999, jax.random.PRNGKey(0))
    with pytest.raises(ValueError):
        bad_cfg = _tiny(celeba64.get_eval_config("", "none"), tmp_path)
        state_utils.load_params_for_eval(bad_cfg, ckpt, 0.9999, jax.random.PRNGKey(0))

    x1 = jax.random.uniform(jax.random.PRNGKey(1), (4, *IMAGE_DIMS), minval=-1, maxval=1)
    d = int(np.prod(IMAGE_DIMS))
    for n_steps in (1, 2, 4):
        logp = likelihood.inverse_logp_div_head(
            net.apply, params, x1, n_steps, eval_cfg.network.rescale, 1.0
        )
        bpd, _ = likelihood.logp_to_bpd(logp, d)
        assert bpd.shape == (4,) and np.all(np.isfinite(bpd))

    logp = likelihood.inverse_logp_hutchinson(
        net.apply, params, x1, 8, eval_cfg.network.rescale, jax.random.PRNGKey(2), 2
    )
    assert np.all(np.isfinite(logp))


def test_teacher_hutchinson_bpd(lsd_ckpt, tmp_path):
    cfg = _tiny(celeba64.get_eval_config("", "none"), tmp_path)
    net, params = state_utils.load_params_for_eval(cfg, lsd_ckpt, 0.9999, jax.random.PRNGKey(0))
    x1 = jnp.zeros((2, *IMAGE_DIMS))
    logp = likelihood.inverse_logp_hutchinson(
        net.apply, params, x1, 4, cfg.network.rescale, jax.random.PRNGKey(0)
    )
    assert logp.shape == (2,) and np.all(np.isfinite(logp))


def test_logp_of_identity_flow_is_base_density():
    """With phi = 0 and div = 0 the model density is the Gaussian base."""
    rescale = 0.5

    def apply_fn(params, s, t, x, label, train, method, return_div):
        return jnp.zeros_like(x), jnp.zeros_like(x)

    x1 = jax.random.normal(jax.random.PRNGKey(0), (3, *IMAGE_DIMS)) * rescale
    logp = likelihood.inverse_logp_div_head(apply_fn, None, x1, 4, rescale, 1.0)
    d = x1[0].size
    expected = -0.5 * jnp.sum((x1 / rescale) ** 2, axis=(1, 2, 3)) - 0.5 * d * jnp.log(
        2 * jnp.pi * rescale**2
    )
    np.testing.assert_allclose(logp, expected, rtol=1e-5)


def test_fid_pipeline_runs(lsd_ckpt, tmp_path):
    cfg = _tiny(celeba64.get_eval_config("", "none"), tmp_path)
    net, params = state_utils.load_params_for_eval(cfg, lsd_ckpt, 0.9999, jax.random.PRNGKey(0))
    inception = fid_utils.get_fid_network()
    sample_rho0 = datasets.setup_base(cfg, jnp.zeros(IMAGE_DIMS))
    import common.flow_map as flow_map

    stats = fid_utils.OnlineStats()
    for i in range(2):
        x0 = sample_rho0(3, jax.random.PRNGKey(i))
        imgs = flow_map.batch_sample(net.apply, params, x0, 2, None)
        imgs = jnp.clip(imgs, -1, 1).transpose(0, 2, 3, 1)
        stats.update(np.asarray(fid_utils.resize_and_incept(imgs, inception)))
    assert stats.n == 6 and stats.mu.shape == (2048,)
    fid = fid_utils.fid_from_stats(stats.mu, stats.sigma, stats.mu, stats.sigma)
    assert np.isfinite(fid) and abs(fid) < 0.05  # rank-deficient covariance (6 samples)


def test_online_stats_matches_numpy():
    feats = np.random.default_rng(0).normal(size=(37, 5))
    stats = fid_utils.OnlineStats()
    for chunk in np.array_split(feats, 4):
        stats.update(chunk)
    np.testing.assert_allclose(stats.mu, feats.mean(0))
    np.testing.assert_allclose(stats.sigma, np.cov(feats, rowvar=False))


def test_scallop_divergence_target_is_unbiased():
    """For Gaussian data the Hutchinson-free target averages to div b (1-D per pixel)."""
    a, r, t, n = 0.7, 0.5, 0.3, 2_000_000
    key0, key1 = jax.random.split(jax.random.PRNGKey(0))
    x0 = r * jax.random.normal(key0, (n,))
    x1 = a * jax.random.normal(key1, (n,))
    xt = (1 - t) * x0 + t * x1
    k = (t * a**2 - (1 - t) * r**2) / ((1 - t) ** 2 * r**2 + t**2 * a**2)  # b(x) = k x

    X = types.SimpleNamespace(
        config=ml_collections.ConfigDict(
            {"divergence_mode": "dlogpdt", "rescale": r, "vectorized_divergence": True}
        ),
        apply=lambda params, t, x, label, train, method, rngs: k * x,
    )
    _, div_hat = losses._calc_divergence_with_phi(
        None, t, xt, x0, x1, None, None, X=X, method="calc_b"
    )
    assert abs(float(jnp.mean(div_hat)) - k) < 5e-3
