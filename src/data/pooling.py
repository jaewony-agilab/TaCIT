"""Fixed-budget patch pooling: FPS centroids + ball-query grouping. implementation.md §3.1.

Steps 1-3 (centroids, N<K padding, grouping) are geometry-only — they depend
solely on a format's taxel layout, so they run once per format and cache to
disk. Step 4 (mean/OR aggregation of the actual per-timestep signal) runs
every batch via `aggregate_patch_signal`.
"""
import hashlib
from pathlib import Path

import numpy as np

MAX_MEMBERS = 16


def _seed_from_format_id(format_id: str) -> int:
    """Deterministic seed so patch layout is stable across epochs and runs."""
    return int(hashlib.sha256(format_id.encode()).hexdigest()[:8], 16)


def _farthest_point_sampling(positions: np.ndarray, num_samples: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n = positions.shape[0]
    chosen = np.empty(num_samples, dtype=np.int64)
    chosen[0] = rng.randint(n)
    dist = np.linalg.norm(positions - positions[chosen[0]], axis=1)
    for i in range(1, num_samples):
        nxt = int(np.argmax(dist))
        chosen[i] = nxt
        dist = np.minimum(dist, np.linalg.norm(positions - positions[nxt], axis=1))
    return chosen


def build_patches(
    positions: np.ndarray,
    normals: np.ndarray,
    link_id: np.ndarray,
    mean_spacing: float,
    format_id: str,
    K: int = 128,
) -> dict:
    """positions, normals: (N, 3) f32. link_id: (N,) i16. mean_spacing: metres.

    Returns dict with centroid_idx (K,), member_idx (K, 16), member_mask (K, 16),
    patch_pos (K, 3), patch_normal (K, 3), patch_link (K,), patch_mask (K,).
    """
    n = positions.shape[0]
    num_centroids = min(n, K)
    seed = _seed_from_format_id(format_id)

    if n <= K:
        centroid_idx = np.arange(n, dtype=np.int64)
    else:
        centroid_idx = _farthest_point_sampling(positions, K, seed)

    radius = 2.0 * mean_spacing
    centroid_pos = positions[centroid_idx]  # (num_centroids, 3)

    # (N, num_centroids) pairwise distance
    dist = np.linalg.norm(positions[:, None, :] - centroid_pos[None, :, :], axis=-1)
    within_radius = dist <= radius

    # Every taxel is assigned to its nearest centroid at minimum — none dropped.
    nearest = np.argmin(dist, axis=1)
    within_radius[np.arange(n), nearest] = True

    member_idx = np.full((K, MAX_MEMBERS), -1, dtype=np.int64)
    member_mask = np.zeros((K, MAX_MEMBERS), dtype=bool)
    patch_pos = np.zeros((K, 3), dtype=np.float32)
    patch_normal = np.zeros((K, 3), dtype=np.float32)
    patch_link = np.full((K,), -1, dtype=np.int16)
    patch_mask = np.zeros((K,), dtype=bool)

    for c in range(num_centroids):
        candidates = np.nonzero(within_radius[:, c])[0]
        order = np.argsort(dist[candidates, c])
        candidates = candidates[order][:MAX_MEMBERS]
        m = len(candidates)
        member_idx[c, :m] = candidates
        member_mask[c, :m] = True
        patch_pos[c] = centroid_pos[c]
        mean_normal = normals[candidates].mean(axis=0)
        norm = np.linalg.norm(mean_normal)
        patch_normal[c] = mean_normal / norm if norm > 1e-8 else mean_normal
        patch_link[c] = link_id[centroid_idx[c]]
        patch_mask[c] = True

    return {
        "centroid_idx": centroid_idx,
        "member_idx": member_idx,
        "member_mask": member_mask,
        "patch_pos": patch_pos,
        "patch_normal": patch_normal,
        "patch_link": patch_link,
        "patch_mask": patch_mask,
    }


def load_or_build_patches(
    cache_dir: str | Path,
    format_id: str,
    positions: np.ndarray,
    normals: np.ndarray,
    link_id: np.ndarray,
    mean_spacing: float,
    K: int = 128,
) -> dict:
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{format_id}.npz"
    if cache_path.exists():
        with np.load(cache_path) as f:
            return {k: f[k] for k in f.files}
    patches = build_patches(positions, normals, link_id, mean_spacing, format_id, K)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **patches)
    return patches


def aggregate_patch_signal(
    signal: np.ndarray,
    validity: np.ndarray,
    member_idx: np.ndarray,
    member_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-batch step 4: mean signal / OR validity over each patch's members.

    signal, validity: (..., N, C). member_idx, member_mask: (K, M).
    Returns patch_signal (..., K, C) f32, patch_validity (..., K, C) bool.
    Mean, not max — force is extensive; max would make dense sensors read
    systematically higher than sparse ones on identical contact.
    """
    n = signal.shape[-2]
    safe_idx = np.clip(member_idx, 0, n - 1)
    sig_members = signal[..., safe_idx, :]  # (..., K, M, C)
    val_members = validity[..., safe_idx, :].astype(bool)  # (..., K, M, C)
    valid_slot = member_mask[:, :, None] & val_members  # (..., K, M, C)

    count = valid_slot.sum(axis=-2)  # (..., K, C)
    total = np.where(valid_slot, sig_members, 0.0).sum(axis=-2)  # (..., K, C)
    patch_signal = np.divide(
        total, count, out=np.zeros_like(total, dtype=np.float32), where=count > 0
    ).astype(np.float32)
    patch_validity = valid_slot.any(axis=-2)
    return patch_signal, patch_validity
