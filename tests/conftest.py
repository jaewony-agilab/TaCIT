"""Synthetic data-contract fixture shared by test_contract, test_pooling, test_resume."""
import json

import numpy as np
import pytest


def _write_format(fmt_dir, num_taxels: int, seed: int):
    fmt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    positions = rng.uniform(-0.05, 0.05, size=(num_taxels, 3)).astype(np.float32)
    normals = rng.normal(size=(num_taxels, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    link_id = rng.randint(0, 4, size=num_taxels).astype(np.int16)
    np.savez(fmt_dir / "taxels.npz", positions=positions, normals=normals, link_id=link_id)

    h_T = {
        "density": float(num_taxels), "mean_spacing": 0.003, "coverage_area": 0.0005,
        "curvature": 0.01, "modality": [1, 0, 0], "gain": [1.0, 1.0, 1.0],
        "nonlinearity": 0.02, "saturation": 10.0, "noise_sigma": 0.01,
        "dropout_rate": 0.02, "latency_frames": 0, "stiffness_k": 2000.0,
    }
    with open(fmt_dir / "h_T.json", "w") as f:
        json.dump(h_T, f)


def _write_episode(ep_dir, format_id, embodiment_id, num_taxels, dof, T, V, seed):
    ep_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    signal = rng.normal(size=(T, num_taxels, 3)).astype(np.float32)
    validity = np.ones((T, num_taxels, 3), dtype=np.uint8)
    np.savez(ep_dir / "tactile.npz", signal=signal, validity=validity)

    q = rng.normal(size=(T, dof)).astype(np.float32)
    qd = rng.normal(size=(T, dof)).astype(np.float32)
    np.savez(ep_dir / "proprio.npz", q=q, qd=qd)

    target_q = rng.normal(size=(T, dof)).astype(np.float32)
    np.savez(ep_dir / "actions.npz", target_q=target_q)

    frames_dir = ep_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    for t in range(T):
        frame = rng.randint(0, 255, size=(V, 224, 224, 3)).astype(np.uint8)
        np.save(frames_dir / f"{t:06d}.npy", frame)

    meta = {
        "episode_id": ep_dir.name, "format_id": format_id, "embodiment_id": embodiment_id,
        "task_id": "pick_cup_stack", "num_frames": T, "num_views": V,
    }
    with open(ep_dir / "meta.json", "w") as f:
        json.dump(meta, f)


@pytest.fixture
def synthetic_data_root(tmp_path):
    """2 train formats x 1 embodiment, 4 episodes each; 1 heldout format, 2 episodes."""
    root = tmp_path / "data"
    embodiment_id = "test_bot"
    dof, T, V = 6, 30, 1

    _write_format(root / "formats" / "fmt_a", num_taxels=40, seed=1)
    _write_format(root / "formats" / "fmt_b", num_taxels=64, seed=2)
    _write_format(root / "formats" / "fmt_heldout", num_taxels=80, seed=3)

    splits = {"train": [], "val": [], "val_heldout_format": []}
    counter = 0
    for fmt_id, num_taxels, n_eps in [("fmt_a", 40, 4), ("fmt_b", 64, 4)]:
        for i in range(n_eps):
            eid = f"ep_{counter:04d}"
            counter += 1
            _write_episode(root / "episodes" / eid, fmt_id, embodiment_id, num_taxels, dof, T, V, seed=counter)
            (splits["train"] if i < 3 else splits["val"]).append(eid)

    for i in range(2):
        eid = f"ep_{counter:04d}"
        counter += 1
        _write_episode(root / "episodes" / eid, "fmt_heldout", embodiment_id, 80, dof, T, V, seed=counter)
        splits["val_heldout_format"].append(eid)

    with open(root / "splits.json", "w") as f:
        json.dump(splits, f)

    return {"root": root, "splits": splits, "embodiments": {embodiment_id: dof}, "dof": dof, "T": T, "V": V}
