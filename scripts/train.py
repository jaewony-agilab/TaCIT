#!/usr/bin/env python
"""Entry point: python scripts/train.py configs/stage1_baseline.yaml [--resume PATH]

Run inside the project venv (`uv pip install -e .` already puts `data`,
`models`, `train` on the path via the src/ package layout).
"""
import argparse

import torch
import yaml

from train.loop import train


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["device"] = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    max_dof = max(cfg["embodiments"].values())
    cfg["model"] = {
        "num_links": cfg["num_links"],
        "W": cfg["W"],
        "d_tok": cfg["d_tok"],
        "d_model": cfg["d_model"],
        "d_h": cfg["d_h"],
        "h_T_dim": cfg["h_T_dim"],
        "max_dof": max_dof,
        "num_views": cfg["num_views"],
        "H": cfg["H"],
        "embodiments": cfg["embodiments"],
        "num_vision_queries": cfg.get("num_vision_queries", 16),
        "view_dropout_p": cfg.get("view_dropout_p", 0.1),
    }
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(load_config(args.config), resume_path=args.resume)


if __name__ == "__main__":
    main()
