import torch

from train.loop import train


def _build_cfg(data, tmp_path, tag: str, total_steps: int, stop_after_step: int | None = None):
    dof = data["dof"]
    max_dof = max(data["embodiments"].values())
    cfg = {
        "data_root": str(data["root"]),
        "cache_dir": str(tmp_path / f"cache_{tag}"),
        "ckpt_dir": str(tmp_path / f"ckpt_{tag}"),
        "device": "cpu",
        "embodiments": data["embodiments"],
        "K": 16, "W": 4, "H": 10,
        "num_links": 8, "d_tok": 32, "d_model": 64, "d_h": 16, "h_T_dim": 16,
        "num_views": data["V"], "num_vision_queries": 4, "view_dropout_p": 0.0,
        "use_h_encoder": False, "use_adversary": False,
        "batch_size": 2, "num_groups": 2, "num_workers": 0,
        "lr_encoder": 1e-3, "lr_decoder": 1e-3, "lr_adversary": 1e-3,
        "wd_encoder": 0.0, "wd_decoder": 0.0,
        # total_steps fixes the schedule horizon (cosine decay curve) — it must be the
        # SAME for the interrupted run and its resumption. stop_after_step is what
        # actually ends the run early, independent of that horizon.
        "warmup_steps": 1, "total_steps": total_steps, "stop_after_step": stop_after_step,
        "log_every": 1000, "validate_every": 1000, "checkpoint_every": 1000,
        "seed": 0,
        "log": {"use_wandb": False},
    }
    cfg["model"] = {
        "num_links": cfg["num_links"], "W": cfg["W"], "d_tok": cfg["d_tok"], "d_model": cfg["d_model"],
        "d_h": cfg["d_h"], "h_T_dim": cfg["h_T_dim"], "max_dof": max_dof, "num_views": cfg["num_views"],
        "H": cfg["H"], "embodiments": cfg["embodiments"], "num_vision_queries": cfg["num_vision_queries"],
        "view_dropout_p": cfg["view_dropout_p"],
    }
    return cfg


def test_resume_matches_uninterrupted_run(synthetic_data_root, tmp_path):
    cfg_full = _build_cfg(synthetic_data_root, tmp_path, "full", total_steps=6)
    model_full = train(cfg_full)

    cfg_part = _build_cfg(synthetic_data_root, tmp_path, "part", total_steps=6, stop_after_step=3)
    train(cfg_part)
    ckpt_path = sorted((tmp_path / "ckpt_part").glob("step_*.pt"))[-1]

    cfg_resumed = _build_cfg(synthetic_data_root, tmp_path, "part", total_steps=6)
    model_resumed = train(cfg_resumed, resume_path=str(ckpt_path))

    for (name, p_full), (_, p_resumed) in zip(
        model_full.named_parameters(), model_resumed.named_parameters()
    ):
        assert torch.allclose(p_full, p_resumed, atol=1e-5), f"resume diverged at {name}"
