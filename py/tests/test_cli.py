"""End-to-end run of every launcher on mocked CelebA (tfds.testing.mock_data).

Chain: FID reference stats -> LSD teacher -> SCALLOP / F2D2 -> test BPD -> FID.
"""

import json
import os
import runpy
import sys

import numpy as np
import pytest
import tensorflow_datasets as tfds

LAUNCHERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "launchers")
CFG = "tests.tiny_celeba64"


def _run(script, *args):
    argv = sys.argv
    sys.argv = [script, *map(str, args)]
    try:
        runpy.run_path(os.path.join(LAUNCHERS, script), run_name="__main__")
    finally:
        sys.argv = argv


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    data = root / "data"
    data.mkdir()
    with tfds.testing.mock_data(num_examples=16, policy=tfds.testing.MockPolicy.USE_CODE):
        stats = data / "celeb_a" / "celeba_stats.npz"  # default cfg.logging.fid_stats_path
        _run("calc_dataset_fid_stats.py", "--dataset_location", data, "--out", stats,
             "--batch_size", 8, "--max_samples", 16)

        lsd = root / "lsd"
        _run("learn.py", "--cfg_path", CFG, "--dataset_location", data,
             "--output_folder", lsd, "--mode", "pretrain_flowmap")
        lsd_ckpt = lsd / "celeba64_pretrain_flowmap_2.pkl"

        ckpts = {}
        for mode, extra in [("dlogpdt", ["--use_importance_sample_for_diag_t"]), ("hutchinson", [])]:
            out = root / mode
            _run("learn.py", "--cfg_path", CFG, "--dataset_location", data,
                 "--output_folder", out, "--mode", mode,
                 "--network_load_path", lsd_ckpt, "--teacher_load_path", lsd_ckpt, *extra)
            ckpts[mode] = out / f"celeba64_{mode}_2.pkl"

        teacher_eval = root / "eval_teacher"
        _run("eval_bpd.py", "--cfg_path", CFG, "--dataset_location", data,
             "--ckpt_path", lsd_ckpt, "--out_dir", teacher_eval, "--div_head", "none",
             "--estimator", "hutchinson", "--n_steps", 4, "--max_samples", 6, "--batch_size", 4)

        scallop_eval = root / "eval_scallop"
        _run("eval_bpd.py", "--cfg_path", CFG, "--dataset_location", data,
             "--ckpt_path", ckpts["dlogpdt"], "--out_dir", scallop_eval,
             "--n_steps", 1, 2, "--max_samples", 6, "--batch_size", 4,
             "--reference_csv", teacher_eval / "bpd_hutchinson_4steps.csv")

        fid_out = root / "fid"
        _run("sample_fid.py", "--cfg_path", CFG, "--ckpt_path", ckpts["dlogpdt"],
             "--stats", stats, "--out_dir", fid_out, "--n_steps", 1, 2,
             "--n_images", 6, "--batch_size", 4, "--save_samples", 4)
    return locals()


def test_checkpoints_written(runs):
    assert runs["lsd_ckpt"].exists()
    for ckpt in runs["ckpts"].values():
        assert ckpt.exists()


def test_bpd_outputs(runs):
    teacher = np.loadtxt(runs["teacher_eval"] / "bpd_hutchinson_4steps.csv", delimiter=",", skiprows=1)
    assert teacher.shape == (6, 2) and np.all(np.isfinite(teacher[:, 1]))
    with open(runs["scallop_eval"] / "bpd_div_head_summary.json") as f:
        results = json.load(f)["results"]
    for key in ("1_steps", "2_steps"):
        assert results[key]["n_images"] == 6
        assert all(np.isfinite(results[key][m]) for m in ("bpd", "abs_delta", "err"))


def test_fid_outputs(runs):
    with open(runs["fid_out"] / "fid_summary.json") as f:
        fid = json.load(f)["fid"]
    assert set(fid) == {"1_steps", "2_steps"} and all(np.isfinite(v) for v in fid.values())
    samples = np.load(runs["fid_out"] / "samples_1steps.npz")["samples"]
    assert samples.shape == (4, 64, 64, 3)
