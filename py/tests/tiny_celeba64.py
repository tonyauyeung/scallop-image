"""Tiny variant of configs.celeba64 for CPU smoke tests (real 64x64 CelebA pipeline)."""

from configs import celeba64

TINY_UNET = {
    "model_channels": 16,
    "channel_mult": [1, 2],
    "num_blocks": 1,
    "attn_resolutions": [32],
    "block_kwargs": {"dropout": 0.0},
}


def _shrink(cfg):
    cfg.network.unet_kwargs = TINY_UNET
    cfg.network.use_bfloat16 = False
    cfg.optimization.bs = 8
    cfg.optimization.total_steps = 2
    cfg.logging.save_freq = 1
    cfg.logging.visual_freq = 2
    cfg.logging.visual_n_steps = [1, 2]
    cfg.logging.plot_bs = 4
    cfg.logging.bpd_freq = 1
    cfg.logging.bpd_n_steps = [1, 2]
    cfg.logging.bpd_batch_size = 4
    cfg.logging.fid_freq = 2
    cfg.logging.fid_n_steps = [1]
    cfg.logging.fid_n_samples = 8
    cfg.logging.fid_batch_size = 4
    return cfg


def get_config(dataset_location, output_folder, mode="dlogpdt"):
    cfg = _shrink(celeba64.get_config(dataset_location, output_folder, mode))
    if mode == "pretrain_flowmap":
        cfg.optimization.diag_fraction_switch = (1, 0.5)
    return cfg


def get_eval_config(dataset_location, div_head="vector", divergence_rescale=None):
    return _shrink(celeba64.get_eval_config(dataset_location, div_head, divergence_rescale))
