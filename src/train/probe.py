"""Contamination probe — the single most important diagnostic. implementation.md §7.2.

Ridge, 5-fold CV, on a fixed val slice. Near 0 R^2 -> z_t is format-agnostic.
Climbing while L_action looks healthy -> silent failure mode.
"""
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

from data.normalize import H_T_DIM_NAMES
from train.forward import forward_batch


@torch.no_grad()
def collect_probe_cache(model, probe_loader, cfg: dict, step: int, device, max_samples: int = 2000):
    """probe_loader must be a fixed (non-reshuffled) loader so the slice is stable across calls."""
    was_training = model.training
    model.eval()
    z_ts, h_Ts = [], []
    n = 0
    for batch in probe_loader:
        out = forward_batch(model, batch, cfg, step, device)
        z_ts.append(out["z_t"].float().cpu().numpy())
        h_Ts.append(out["h_T_norm"].float().cpu().numpy())
        n += out["z_t"].shape[0]
        if n >= max_samples:
            break
    model.train(was_training)
    z_t_cache = np.concatenate(z_ts, axis=0)[:max_samples]
    h_T_cache = np.concatenate(h_Ts, axis=0)[:max_samples]
    return z_t_cache, h_T_cache


def contamination_probe(z_t_cache: np.ndarray, h_T_cache: np.ndarray, alpha: float = 1.0) -> dict:
    """Returns per-field R^2 (dict) and mean R^2. Baseline (predict mean) = 0."""
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    r2_per_field = {}
    for j, name in enumerate(H_T_DIM_NAMES):
        y = h_T_cache[:, j]
        preds = cross_val_predict(Ridge(alpha=alpha), z_t_cache, y, cv=kf)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - np.sum((y - preds) ** 2) / ss_tot if ss_tot > 1e-12 else 0.0
        r2_per_field[name] = float(r2)
    return {"r2_per_field": r2_per_field, "mean_r2": float(np.mean(list(r2_per_field.values())))}
