"""
Nicholas M. Boffi
10/5/25

Unconditional CelebA-64 with an EDM2 UNet.

``mode`` selects the recipe:
  - ``pretrain_flowmap``: LSD flow map teacher trained from scratch (no divergence head).
  - ``hutchinson`` (F2D2) / ``dlogpdt`` (SCALLOP): likelihood flow map initialized
    from the LSD teacher, adding only a divergence head.
"""

import ml_collections


def get_config(
    dataset_location: str, output_folder: str, mode: str = "dlogpdt"
) -> ml_collections.ConfigDict:
    # ensure jax.device_count works (weird issue with importlib)
    import jax

    is_pretraining = mode == "pretrain_flowmap"
    config = ml_collections.ConfigDict()

    # training config
    config.training = ml_collections.ConfigDict()
    config.training.loss_type = "lsd"
    config.training.stopgrad_type = "convex"
    config.training.tmin = 0.0
    config.training.tmax = 1.0
    config.training.time_distr = "uniform"  # diagonal times; "linear" = 1/(1-t) importance sampling
    config.training.seed = 42
    config.training.ema_facs = [0.9999]
    config.training.ndevices = jax.device_count()

    # problem config
    config.problem = ml_collections.ConfigDict()
    config.problem.target = "celeb_a"
    config.problem.dataset_location = dataset_location
    config.problem.image_dims = (3, 64, 64)
    config.problem.d = 3 * 64 * 64
    config.problem.interp_type = "linear"

    # optimization config (RAdam, sqrt decay after decay_steps)
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 256
    config.optimization.clip = 1.0
    config.optimization.decay_steps = 35_000
    config.optimization.schedule_type = "sqrt"
    if is_pretraining:
        config.optimization.learning_rate = 1e-2
        config.optimization.total_steps = 350_000
        # fraction of the batch on the diagonal (flow matching); the rest is self-distillation
        config.optimization.diag_fraction = 0.75
        config.optimization.diag_fraction_switch = (200_000, 0.5)
    else:
        config.optimization.learning_rate = 1e-4
        config.optimization.total_steps = 60_000
        config.optimization.diag_fraction = 0.5
        config.optimization.diag_fraction_switch = None

    # logging config
    config.logging = ml_collections.ConfigDict()
    config.logging.wandb_project = "scallop"
    config.logging.wandb_entity = None  # defaults to the WANDB_ENTITY environment variable
    config.logging.wandb_name = f"celeba64_{mode}"
    config.logging.output_folder = output_folder
    config.logging.output_name = config.logging.wandb_name
    config.logging.log_freq = 1
    config.logging.save_freq = 20_000
    config.logging.visual_freq = 20_000
    config.logging.visual_n_steps = [1, 2, 4, 8, 16]
    config.logging.visual_ema_factor = 0.9999
    config.logging.plot_bs = 25

    # online BPD on a training batch (likelihood flow maps only)
    config.logging.bpd_freq = 1_000
    config.logging.bpd_n_steps = [1, 2, 4, 8, 16]
    config.logging.bpd_batch_size = 64
    config.logging.bpd_ema_factor = 0.9999

    # online FID; fid_stats_path comes from launchers/calc_dataset_fid_stats.py
    config.logging.fid_freq = 10_000
    config.logging.fid_stats_path = f"{dataset_location}/celeb_a/celeba_stats.npz"
    config.logging.fid_n_samples = 10_000
    config.logging.fid_batch_size = 256
    config.logging.fid_n_steps = [1, 2, 4, 8, 16]
    config.logging.fid_ema_factor = 0.9999

    # network config
    config.network = ml_collections.ConfigDict()
    config.network.network_type = "edm2"
    config.network.divergence_mode = mode
    config.network.predict_divergence = not is_pretraining
    config.network.vectorized_divergence = True
    config.network.divergence_rescale = 1.0
    config.network.div_loss_weighting_type = "uniform"
    config.network.load_path = ""  # checkpoint to initialize from
    config.network.reset_optimizer = True
    config.network.init_from_ema_factor = 0.9999
    config.network.img_resolution = config.problem.image_dims[1]
    config.network.img_channels = config.problem.image_dims[0]
    config.network.input_dims = config.problem.image_dims
    config.network.label_dim = 0
    config.network.use_cfg = False
    config.network.logvar_channels = 128
    config.network.use_bfloat16 = True
    config.network.use_weight = True
    config.network.rescale = 0.5  # std of [-1, 1] images; also the base std
    config.network.unet_kwargs = {
        "model_channels": 128,
        "channel_mult": [1, 2, 3, 4],
        "num_blocks": 3,
        "attn_resolutions": [16, 8],
        "block_kwargs": {"dropout": 0.0},
    }

    # external teacher (pretrained LSD flow map) for the diagonal velocity
    config.teacher = ml_collections.ConfigDict()
    config.teacher.load_path = ""
    config.teacher.ema_fac = 0.9999

    return config


def get_eval_config(
    dataset_location: str, div_head: str = "vector", divergence_rescale: float = None
) -> ml_collections.ConfigDict:
    """Network config matching a checkpoint: ``div_head`` in {none, vector, scalar}."""
    if div_head not in ("none", "vector", "scalar"):
        raise ValueError(f"Unknown div_head: {div_head}")
    mode = "pretrain_flowmap" if div_head == "none" else "dlogpdt"
    cfg = get_config(dataset_location, "", mode)
    cfg.training.ndevices = 1
    cfg.network.vectorized_divergence = div_head == "vector"
    if divergence_rescale is not None:
        cfg.network.divergence_rescale = divergence_rescale
    elif div_head == "scalar":
        cfg.network.divergence_rescale = float(cfg.problem.d)
    return cfg
