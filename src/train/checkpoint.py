"""Checkpoint save/load/rotation. implementation.md §7.4, §8.

A checkpoint missing action_norm, h_T_norm_constants, or format_registry is
not deployable — asserted on every save.
"""
import random
import shutil
from pathlib import Path

import numpy as np
import torch

REQUIRED_KEYS = ("action_norm", "h_T_norm_constants", "format_registry")


def get_rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def build_checkpoint(
    model, optimizer, scheduler, step, epoch, action_norm, h_T_norm_constants,
    format_registry, config, git_sha,
) -> dict:
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "action_norm": action_norm,
        "h_T_norm_constants": h_T_norm_constants,
        "format_registry": format_registry,
        "config": config,
        "git_sha": git_sha,
        "rng_state": get_rng_state(),
    }
    missing = [k for k in REQUIRED_KEYS if not ckpt.get(k)]
    if missing:
        raise ValueError(f"checkpoint missing required keys, not deployable: {missing}")
    return ckpt


def load_checkpoint(path, model=None, optimizer=None, scheduler=None, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


class CheckpointManager:
    """Keeps the last `keep_last` steps rolling, plus a separate best-by-val_heldout_format copy."""

    def __init__(self, ckpt_dir, keep_last: int = 3):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.rolling: list[Path] = []
        self.best_metric = float("inf")

    def save(self, step: int, ckpt: dict, heldout_l_action: float | None = None) -> Path:
        path = self.ckpt_dir / f"step_{step:07d}.pt"
        torch.save(ckpt, path)
        self.rolling.append(path)
        while len(self.rolling) > self.keep_last:
            self.rolling.pop(0).unlink(missing_ok=True)
        if heldout_l_action is not None and heldout_l_action < self.best_metric:
            self.best_metric = heldout_l_action
            shutil.copy(path, self.ckpt_dir / "best.pt")
        return path
