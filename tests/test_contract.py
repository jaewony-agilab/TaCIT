import json

import numpy as np
import pytest

from data.contract import ContractError, validate_contract


def test_valid_synthetic_contract_passes(synthetic_data_root):
    splits = validate_contract(synthetic_data_root["root"])
    assert set(splits.keys()) == {"train", "val", "val_heldout_format"}
    assert len(splits["train"]) > 0


def test_episode_in_two_splits_fails(synthetic_data_root):
    root = synthetic_data_root["root"]
    with open(root / "splits.json") as f:
        splits = json.load(f)
    splits["val"].append(splits["train"][0])
    with open(root / "splits.json", "w") as f:
        json.dump(splits, f)

    with pytest.raises(ContractError, match="both"):
        validate_contract(root)


def test_missing_format_dir_fails(synthetic_data_root):
    root = synthetic_data_root["root"]
    with open(root / "splits.json") as f:
        splits = json.load(f)
    eid = splits["train"][0]
    with open(root / "episodes" / eid / "meta.json") as f:
        meta = json.load(f)
    meta["format_id"] = "does_not_exist"
    with open(root / "episodes" / eid / "meta.json", "w") as f:
        json.dump(meta, f)

    with pytest.raises(ContractError, match="not found under formats"):
        validate_contract(root)


def test_signal_taxel_count_mismatch_fails(synthetic_data_root):
    root = synthetic_data_root["root"]
    with open(root / "splits.json") as f:
        splits = json.load(f)
    eid = splits["train"][0]
    ep_dir = root / "episodes" / eid
    tactile = dict(np.load(ep_dir / "tactile.npz"))
    tactile["signal"] = tactile["signal"][:, :3, :]  # wrong taxel count
    np.savez(ep_dir / "tactile.npz", **tactile)

    with pytest.raises(ContractError, match="shape\\[1\\]"):
        validate_contract(root)


def test_num_frames_mismatch_fails(synthetic_data_root):
    root = synthetic_data_root["root"]
    with open(root / "splits.json") as f:
        splits = json.load(f)
    eid = splits["train"][0]
    with open(root / "episodes" / eid / "meta.json") as f:
        meta = json.load(f)
    meta["num_frames"] = meta["num_frames"] + 5
    with open(root / "episodes" / eid / "meta.json", "w") as f:
        json.dump(meta, f)

    with pytest.raises(ContractError, match="num_frames"):
        validate_contract(root)


def test_missing_h_T_key_fails(synthetic_data_root):
    root = synthetic_data_root["root"]
    with open(root / "formats" / "fmt_a" / "h_T.json") as f:
        h_T = json.load(f)
    del h_T["stiffness_k"]
    with open(root / "formats" / "fmt_a" / "h_T.json", "w") as f:
        json.dump(h_T, f)

    with pytest.raises(ContractError, match="missing keys"):
        validate_contract(root)


def test_heldout_format_overlap_with_train_fails(synthetic_data_root):
    root = synthetic_data_root["root"]
    with open(root / "splits.json") as f:
        splits = json.load(f)
    # Point a heldout episode's format at a training format -> overlap.
    eid = splits["val_heldout_format"][0]
    with open(root / "episodes" / eid / "meta.json") as f:
        meta = json.load(f)
    meta["format_id"] = "fmt_a"
    with open(root / "episodes" / eid / "meta.json", "w") as f:
        json.dump(meta, f)
    # Give it a taxel-count-compatible tactile array so the overlap check is what fails, not shape.
    tactile = dict(np.load(root / "episodes" / eid / "tactile.npz"))
    fmt_a_n = np.load(root / "formats" / "fmt_a" / "taxels.npz")["positions"].shape[0]
    tactile["signal"] = tactile["signal"][:, :fmt_a_n, :]
    tactile["validity"] = tactile["validity"][:, :fmt_a_n, :]
    np.savez(root / "episodes" / eid / "tactile.npz", **tactile)

    with pytest.raises(ContractError, match="disjoint"):
        validate_contract(root)
