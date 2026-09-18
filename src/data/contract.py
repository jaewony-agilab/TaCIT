"""Startup validation of the on-disk data contract. implementation.md §2.5.

Fails fast with the offending episode id. Never silently skips a bad episode.
"""
import json
from pathlib import Path

import numpy as np

from data.normalize import H_T_FIELDS


class ContractError(Exception):
    pass


def _load_meta(data_root: Path, episode_id: str) -> dict:
    with open(data_root / "episodes" / episode_id / "meta.json") as f:
        return json.load(f)


def validate_contract(data_root: str | Path) -> dict:
    """Runs every check in §2.5. Returns the parsed splits.json on success."""
    data_root = Path(data_root)

    with open(data_root / "splits.json") as f:
        splits = json.load(f)

    formats_dir = data_root / "formats"
    format_ids = {p.name for p in formats_dir.iterdir() if p.is_dir()}

    seen_in_split: dict[str, str] = {}
    train_format_ids: set[str] = set()
    heldout_format_ids: set[str] = set()

    for split_name, episode_ids in splits.items():
        for episode_id in episode_ids:
            if episode_id in seen_in_split:
                raise ContractError(
                    f"episode {episode_id!r} appears in both "
                    f"{seen_in_split[episode_id]!r} and {split_name!r}"
                )
            seen_in_split[episode_id] = split_name

            meta = _load_meta(data_root, episode_id)
            format_id = meta["format_id"]

            if format_id not in format_ids:
                raise ContractError(
                    f"episode {episode_id!r}: format_id {format_id!r} not found under formats/"
                )
            if split_name == "train":
                train_format_ids.add(format_id)
            elif split_name == "val_heldout_format":
                heldout_format_ids.add(format_id)

            ep_dir = data_root / "episodes" / episode_id
            tactile = np.load(ep_dir / "tactile.npz")
            taxels = np.load(formats_dir / format_id / "taxels.npz")
            if tactile["signal"].shape[1] != taxels["positions"].shape[0]:
                raise ContractError(
                    f"episode {episode_id!r}: tactile.signal.shape[1]="
                    f"{tactile['signal'].shape[1]} != taxels.positions.shape[0]="
                    f"{taxels['positions'].shape[0]}"
                )

            num_frames = meta["num_frames"]
            for key, arr in (
                ("tactile.signal", tactile["signal"]),
                ("tactile.validity", tactile["validity"]),
            ):
                if arr.shape[0] != num_frames:
                    raise ContractError(
                        f"episode {episode_id!r}: {key}.shape[0]={arr.shape[0]} "
                        f"!= meta.num_frames={num_frames}"
                    )
            proprio = np.load(ep_dir / "proprio.npz")
            actions = np.load(ep_dir / "actions.npz")
            for key, arr in (("proprio.q", proprio["q"]), ("actions.target_q", actions["target_q"])):
                if arr.shape[0] != num_frames:
                    raise ContractError(
                        f"episode {episode_id!r}: {key}.shape[0]={arr.shape[0]} "
                        f"!= meta.num_frames={num_frames}"
                    )

            h_T_path = formats_dir / format_id / "h_T.json"
            with open(h_T_path) as f:
                h_T = json.load(f)
            missing = [k for k in H_T_FIELDS if k not in h_T]
            if missing:
                raise ContractError(
                    f"episode {episode_id!r}: format {format_id!r} h_T.json missing keys {missing}"
                )

    overlap = train_format_ids & heldout_format_ids
    if overlap:
        raise ContractError(
            f"val_heldout_format formats not disjoint from train formats: {overlap}"
        )

    return splits
