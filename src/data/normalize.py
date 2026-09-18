"""h_T and action normalization constants. implementation.md §2.3, §3.3."""
import json
import math
from pathlib import Path

import numpy as np

# Fixed field order — the model indexes positionally. implementation.md §2.3.
H_T_FIELDS = [
    "density", "mean_spacing", "coverage_area", "curvature", "modality",
    "gain", "nonlinearity", "saturation", "noise_sigma", "dropout_rate",
    "latency_frames", "stiffness_k",
]
H_T_DIM = 16  # 1+1+1+1+3+3+1+1+1+1+1+1


def _norm_density(x):
    return math.log(x) / math.log(600)


def _norm_stiffness(x):
    return math.log(x) / math.log(5000)


# field -> (dim, normalize_fn applied elementwise)
_SCALAR_NORM = {
    "density": (1, _norm_density),
    "mean_spacing": (1, lambda x: x / 0.01),
    "coverage_area": (1, lambda x: x / 0.001),
    "curvature": (1, lambda x: x / 0.02),
    "nonlinearity": (1, lambda x: x / 0.1),
    "saturation": (1, lambda x: x / 20.0),
    "noise_sigma": (1, lambda x: x / 0.05),
    "dropout_rate": (1, lambda x: x / 0.15),
    "latency_frames": (1, lambda x: x / 3.0),
    "stiffness_k": (1, _norm_stiffness),
}


def normalize_h_T(h_T: dict) -> np.ndarray:
    """h_T dict (raw, per implementation.md §2.3) -> (16,) float32, fixed field order."""
    missing = [k for k in H_T_FIELDS if k not in h_T]
    if missing:
        raise ValueError(f"h_T missing keys: {missing}")
    parts = []
    for field in H_T_FIELDS:
        val = h_T[field]
        if field == "modality":
            parts.append(np.asarray(val, dtype=np.float32))
        elif field == "gain":
            parts.append((np.asarray(val, dtype=np.float32) - 1.0) / 0.5)
        else:
            _, fn = _SCALAR_NORM[field]
            parts.append(np.asarray([fn(float(val))], dtype=np.float32))
    out = np.concatenate(parts).astype(np.float32)
    assert out.shape == (H_T_DIM,), out.shape
    return out



# Serializable record of the fixed normalization scheme, stored in every checkpoint
# so a real sensor's descriptor normalizes identically regardless of who deploys it.
H_T_NORM_CONSTANTS = {
    "density": {"op": "log_div", "base": 600},
    "mean_spacing": {"op": "div", "value": 0.01},
    "coverage_area": {"op": "div", "value": 0.001},
    "curvature": {"op": "div", "value": 0.02},
    "modality": {"op": "identity"},
    "gain": {"op": "affine", "sub": 1.0, "div": 0.5},
    "nonlinearity": {"op": "div", "value": 0.1},
    "saturation": {"op": "div", "value": 20.0},
    "noise_sigma": {"op": "div", "value": 0.05},
    "dropout_rate": {"op": "div", "value": 0.15},
    "latency_frames": {"op": "div", "value": 3.0},
    "stiffness_k": {"op": "log_div", "base": 5000},
}

H_T_DIM_NAMES = (
    ["density", "mean_spacing", "coverage_area", "curvature"]
    + [f"modality_{i}" for i in range(3)]
    + [f"gain_{i}" for i in range(3)]
    + ["nonlinearity", "saturation", "noise_sigma", "dropout_rate", "latency_frames", "stiffness_k"]
)
assert len(H_T_DIM_NAMES) == H_T_DIM


def load_h_T_norm(path: Path) -> np.ndarray:
    with open(path) as f:
        h_T = json.load(f)
    return normalize_h_T(h_T)


def compute_action_norm(actions_by_embodiment: dict[str, np.ndarray]) -> dict:
    """actions_by_embodiment[embodiment_id] -> (M, D_j) float32 target_q, over train split only.

    Returns {embodiment_id: {"mean": (D_j,), "std": (D_j,)}}, std clamped >= 1e-3.
    """
    out = {}
    for emb_id, actions in actions_by_embodiment.items():
        mean = actions.mean(axis=0).astype(np.float32)
        std = actions.std(axis=0).astype(np.float32)
        std = np.maximum(std, 1e-3).astype(np.float32)
        out[emb_id] = {"mean": mean, "std": std}
    return out


def normalize_actions(target_q: np.ndarray, action_norm: dict) -> np.ndarray:
    return (target_q - action_norm["mean"]) / action_norm["std"]


def denormalize_actions(pred: np.ndarray, action_norm: dict) -> np.ndarray:
    return pred * action_norm["std"] + action_norm["mean"]
