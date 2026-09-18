"""Episode loading, window assembly, split handling. implementation.md §2-3."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.normalize import load_h_T_norm, normalize_actions
from data.pooling import aggregate_patch_signal, load_or_build_patches


class TactileDataset(Dataset):
    """One sample = one (episode, start-frame) window.

    embodiments: {embodiment_id: dof}. Proprio and actions are zero-padded to
    max(dof.values()) so a batch can mix embodiments; a `dof_mask` marks the
    real dims. The per-embodiment decoder head still outputs each sample's
    true `dof` — padding exists only so tactile/proprio/action tensors stack.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        splits: dict,
        embodiments: dict[str, int],
        cache_dir: str | Path,
        action_norm: dict,
        K: int = 128,
        W: int = 8,
        H: int = 100,
    ):
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir)
        self.embodiments = embodiments
        self.max_dof = max(embodiments.values())
        self.action_norm = action_norm
        self.K, self.W, self.H = K, W, H

        self.episode_ids = splits[split]
        self._metas = {eid: self._load_meta(eid) for eid in self.episode_ids}

        self.index: list[tuple[str, int]] = []
        for eid, meta in self._metas.items():
            last_start = meta["num_frames"] - H
            if last_start < 0:
                continue
            self.index.extend((eid, t) for t in range(last_start + 1))

        self._format_cache: dict[str, dict] = {}

    def _load_meta(self, episode_id: str) -> dict:
        with open(self.data_root / "episodes" / episode_id / "meta.json") as f:
            return json.load(f)

    @lru_cache(maxsize=16)
    def _load_episode_arrays(self, episode_id: str):
        ep_dir = self.data_root / "episodes" / episode_id
        tactile = np.load(ep_dir / "tactile.npz")
        proprio = np.load(ep_dir / "proprio.npz")
        actions = np.load(ep_dir / "actions.npz")
        return {
            "signal": tactile["signal"],
            "validity": tactile["validity"],
            "q": proprio["q"],
            "qd": proprio["qd"],
            "target_q": actions["target_q"],
        }

    def _load_frame(self, episode_id: str, t: int) -> np.ndarray:
        frame_path = self.data_root / "episodes" / episode_id / "frames" / f"{t:06d}.npy"
        return np.load(frame_path)  # (V, 224, 224, 3) uint8

    def _format_data(self, format_id: str) -> dict:
        if format_id in self._format_cache:
            return self._format_cache[format_id]
        fmt_dir = self.data_root / "formats" / format_id
        taxels = np.load(fmt_dir / "taxels.npz")
        with open(fmt_dir / "h_T.json") as f:
            h_T_raw = json.load(f)
        patches = load_or_build_patches(
            self.cache_dir,
            format_id,
            taxels["positions"],
            taxels["normals"],
            taxels["link_id"],
            mean_spacing=h_T_raw["mean_spacing"],
            K=self.K,
        )
        entry = {
            "patches": patches,
            "h_T_norm": load_h_T_norm(fmt_dir / "h_T.json"),
        }
        self._format_cache[format_id] = entry
        return entry

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        episode_id, t = self.index[idx]
        meta = self._metas[episode_id]
        arrays = self._load_episode_arrays(episode_id)
        fmt = self._format_data(meta["format_id"])
        patches = fmt["patches"]

        # Causal window [t - W + 1, t], padded by repeating frame 0 at episode start.
        window_idx = np.clip(np.arange(t - self.W + 1, t + 1), 0, None)
        sig_window = arrays["signal"][window_idx]       # (W, N, 3)
        val_window = arrays["validity"][window_idx]      # (W, N, 3)
        patch_signal, patch_validity = aggregate_patch_signal(
            sig_window, val_window, patches["member_idx"], patches["member_mask"]
        )  # (W, K, 3), (W, K, 3)
        patch_signal = np.ascontiguousarray(patch_signal.transpose(1, 0, 2).reshape(self.K, -1))  # (K, W*3)
        patch_validity = patch_validity[-1]  # (K, 3) — current-frame validity

        embodiment_id = meta["embodiment_id"]
        dof = self.embodiments[embodiment_id]
        dof_mask = np.zeros(self.max_dof, dtype=bool)
        dof_mask[:dof] = True

        q = np.zeros(self.max_dof, dtype=np.float32)
        qd = np.zeros(self.max_dof, dtype=np.float32)
        q[:dof] = arrays["q"][t]
        qd[:dof] = arrays["qd"][t]
        proprio = np.concatenate([q, qd])  # (2 * max_dof,)

        target_window = arrays["target_q"][t : t + self.H]  # (H, dof)
        target_norm = normalize_actions(target_window, self.action_norm[embodiment_id])
        actions_norm = np.zeros((self.H, self.max_dof), dtype=np.float32)
        actions_norm[:, :dof] = target_norm

        frames = self._load_frame(episode_id, t)  # (V, 224, 224, 3) uint8

        return {
            "patch_signal": torch.from_numpy(patch_signal),
            "patch_validity": torch.from_numpy(patch_validity),
            "patch_pos": torch.from_numpy(patches["patch_pos"]),
            "patch_normal": torch.from_numpy(patches["patch_normal"]),
            "patch_link": torch.from_numpy(patches["patch_link"].astype(np.int64)),
            "patch_mask": torch.from_numpy(patches["patch_mask"]),
            "proprio": torch.from_numpy(proprio),
            "actions_norm": torch.from_numpy(actions_norm),
            "dof_mask": torch.from_numpy(dof_mask),
            "frames": torch.from_numpy(frames),
            "h_T_norm": torch.from_numpy(fmt["h_T_norm"]),
            "embodiment_id": embodiment_id,
            "format_id": meta["format_id"],
        }
