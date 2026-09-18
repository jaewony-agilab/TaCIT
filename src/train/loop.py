"""Training loop, setup, and logging. implementation.md §6, §7, §8."""
import json
import math
import subprocess
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from data.contract import validate_contract
from data.dataset import TactileDataset
from data.normalize import H_T_NORM_CONSTANTS, compute_action_norm
from data.sampler import FormatBalancedSampler
from models.policy import TactilePolicy
from train.checkpoint import CheckpointManager, build_checkpoint, load_checkpoint, set_rng_state
from train.forward import forward_batch
from train.losses import compute_losses
from train.probe import collect_probe_cache, contamination_probe
from train.validate import run_validation


class Logger:
    """wandb if configured and installed, else a no-op that prints occasionally."""

    def __init__(self, cfg: dict):
        self.use_wandb = False
        if cfg.get("log", {}).get("use_wandb", False):
            try:
                import wandb

                wandb.init(project=cfg["log"].get("project", "tactile-train"), config=cfg)
                self._wandb = wandb
                self.use_wandb = True
            except ImportError:
                print("wandb not installed; falling back to console logging")

    def log(self, metrics: dict, step: int):
        if self.use_wandb:
            self._wandb.log(metrics, step=step)
        elif step % 100 == 0:
            print(f"step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))))

    def log_image(self, key: str, fig, step: int):
        if self.use_wandb:
            self._wandb.log({key: self._wandb.Image(fig)}, step=step)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent).decode().strip()
    except Exception:
        return "unknown"


def _collate(batch):
    return default_collate(batch)


def build_format_registry(data_root: Path) -> dict:
    registry = {}
    for fmt_dir in (data_root / "formats").iterdir():
        if fmt_dir.is_dir():
            with open(fmt_dir / "h_T.json") as f:
                registry[fmt_dir.name] = json.load(f)
    return registry


def infinite_loader(loader: DataLoader):
    epoch = 0
    while True:
        for batch in loader:
            yield epoch, batch
        epoch += 1


def setup(cfg: dict):
    data_root = Path(cfg["data_root"])
    splits = validate_contract(data_root)

    train_actions = {}
    for eid in splits["train"]:
        with open(data_root / "episodes" / eid / "meta.json") as f:
            meta = json.load(f)
        actions = torch.from_numpy(__import__("numpy").load(data_root / "episodes" / eid / "actions.npz")["target_q"])
        train_actions.setdefault(meta["embodiment_id"], []).append(actions.numpy())
    import numpy as np

    action_norm = compute_action_norm({k: np.concatenate(v, axis=0) for k, v in train_actions.items()})

    cache_dir = Path(cfg["cache_dir"])
    embodiments = cfg["embodiments"]

    def make_dataset(split):
        return TactileDataset(
            data_root, split, splits, embodiments, cache_dir, action_norm,
            K=cfg["K"], W=cfg["W"], H=cfg["H"],
        )

    train_ds = make_dataset("train")
    val_ds = make_dataset("val")
    heldout_ds = make_dataset("val_heldout_format") if splits.get("val_heldout_format") else None

    batch_size = cfg["batch_size"]
    num_groups = cfg.get("num_groups", 8)
    train_sampler = FormatBalancedSampler(train_ds, batch_size, num_groups=num_groups)
    # Every fresh DataLoader iterator draws one value from torch's default RNG to
    # seed its (unused, since num_workers=0 in tests) worker-seeding logic. A dedicated
    # generator per loader keeps that draw off the model's RNG stream, so resuming
    # from a checkpoint doesn't shift every subsequent dropout mask by one step.
    loader_generator = torch.Generator().manual_seed(0)
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=_collate,
        num_workers=cfg.get("num_workers", 0), generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate,
        num_workers=cfg.get("num_workers", 0), generator=loader_generator,
    )
    heldout_loader = (
        DataLoader(
            heldout_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate,
            num_workers=cfg.get("num_workers", 0), generator=loader_generator,
        )
        if heldout_ds is not None else None
    )
    probe_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate,
        num_workers=0, generator=loader_generator,
    )

    model = TactilePolicy(cfg["model"]).to(cfg["device"])

    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.encoder_parameters()), "lr": cfg["lr_encoder"], "weight_decay": cfg["wd_encoder"]},
            {"params": list(model.decoder_parameters()), "lr": cfg["lr_decoder"], "weight_decay": cfg["wd_decoder"]},
            {"params": list(model.adversary_parameters()), "lr": cfg["lr_adversary"], "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            _cosine_warmup_lambda(cfg["lr_encoder"], cfg["warmup_steps"], cfg["total_steps"]),
            _cosine_warmup_lambda(cfg["lr_decoder"], cfg["warmup_steps"], cfg["total_steps"]),
            _cosine_warmup_lambda(cfg["lr_adversary"], cfg["warmup_steps"], cfg["total_steps"]),
        ],
    )

    format_registry = build_format_registry(data_root)

    return {
        "model": model, "optimizer": optimizer, "scheduler": scheduler,
        "train_loader": train_loader, "val_loader": val_loader,
        "heldout_loader": heldout_loader, "probe_loader": probe_loader,
        "action_norm": action_norm, "format_registry": format_registry,
    }


def _cosine_warmup_lambda(base_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        lr = min_lr + (base_lr - min_lr) * cos
        return lr / base_lr

    return fn


def film_diagnostics(model) -> dict:
    out = {}
    for i, layer in enumerate(model.state_encoder.layers):
        for name, film in (("film1", layer.film1), ("film2", layer.film2)):
            w = film.bias.detach()
            gamma, beta = w.chunk(2)
            out[f"film/layer{i}_{name}_gamma_minus_1_norm"] = (gamma - 1.0).norm().item()
            out[f"film/layer{i}_{name}_beta_norm"] = beta.norm().item()
    return out


def batch_adversary_r2(adv_out, h_T_norm) -> dict:
    from data.normalize import H_T_DIM_NAMES

    r2 = {}
    adv_out = adv_out.detach()
    for j, name in enumerate(H_T_DIM_NAMES):
        y = h_T_norm[:, j]
        pred = adv_out[:, j]
        ss_tot = ((y - y.mean()) ** 2).sum()
        ss_res = ((y - pred) ** 2).sum()
        r2[name] = (1.0 - ss_res / ss_tot.clamp(min=1e-12)).item()
    return r2


def train(cfg: dict, resume_path: str | None = None):
    if cfg.get("seed") is not None and resume_path is None:
        import random

        import numpy as np

        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])

    ctx = setup(cfg)
    model, optimizer, scheduler = ctx["model"], ctx["optimizer"], ctx["scheduler"]
    device = cfg["device"]
    logger = Logger(cfg)
    ckpt_mgr = CheckpointManager(cfg["ckpt_dir"], keep_last=3)

    start_step = 0
    if resume_path:
        ckpt = load_checkpoint(resume_path, model, optimizer, scheduler, map_location=device)
        set_rng_state(ckpt["rng_state"])
        start_step = ckpt["step"] + 1

    data_iter = infinite_loader(ctx["train_loader"])
    model.train()
    heldout_l_action = None

    for step in range(start_step, cfg["total_steps"]):
        epoch, batch = next(data_iter)
        out = forward_batch(model, batch, cfg, step, device)
        losses = compute_losses(
            out["pred"], out["actions_norm"], out["dof_mask"], out["h_T_norm"], step, cfg,
            out["attract_out"], out["adv_out"], out["mu"], out["logvar"], out["p_mu"], out["p_logvar"],
            out["w_att"], out["beta"],
        )
        optimizer.zero_grad()
        losses["L_total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        n_formats = len(set(out["format_id"]))
        metrics = {f"train/{k}": v.item() for k, v in losses.items()} | {
            "train/lambda": out["lam"], "train/beta_kl": out["beta"], "train/w_attract": out["w_att"],
            "train/grad_norm": grad_norm.item(), "train/formats_per_batch": n_formats,
            "train/lr_encoder": optimizer.param_groups[0]["lr"],
            "train/lr_decoder": optimizer.param_groups[1]["lr"],
            "train/lr_adversary": optimizer.param_groups[2]["lr"],
        }

        min_formats = cfg.get("num_groups", 8)
        if n_formats < min_formats:
            raise AssertionError(f"step {step}: only {n_formats} distinct formats in batch, need >= {min_formats}")

        if step % cfg.get("log_every", 500) == 0:
            metrics.update(film_diagnostics(model))
            if out["adv_out"] is not None:
                for name, r2 in batch_adversary_r2(out["adv_out"], out["h_T_norm"]).items():
                    metrics[f"adv_r2/{name}"] = r2
            if out["logvar"] is not None:
                for d in range(out["logvar"].shape[1]):
                    metrics[f"h_kl/dim{d}"] = out["logvar"][:, d].exp().mean().item()
            empty_frac = 1.0 - batch["patch_mask"].float().mean().item()
            metrics["data/empty_patch_fraction"] = empty_frac

        if step % cfg.get("validate_every", 2000) == 0 and step > 0:
            val_metrics = run_validation(model, ctx["val_loader"], cfg, step, device)
            metrics.update({f"val/{k}": v for k, v in val_metrics.items() if not isinstance(v, (list, dict))})
            heldout_l_action = None
            if ctx["heldout_loader"] is not None:
                heldout_metrics = run_validation(model, ctx["heldout_loader"], cfg, step, device)
                metrics.update({f"val_heldout/{k}": v for k, v in heldout_metrics.items() if not isinstance(v, (list, dict))})
                heldout_l_action = heldout_metrics["L_action"]

            z_t_cache, h_T_cache = collect_probe_cache(model, ctx["probe_loader"], cfg, step, device)
            probe_result = contamination_probe(z_t_cache, h_T_cache)
            metrics["probe/mean_r2"] = probe_result["mean_r2"]
            metrics.update({f"probe/r2_{k}": v for k, v in probe_result["r2_per_field"].items()})

        logger.log(metrics, step)

        stopping = cfg.get("stop_after_step") is not None and step >= cfg["stop_after_step"]
        if (step % cfg.get("checkpoint_every", 5000) == 0 and step > 0) or stopping:
            ckpt = build_checkpoint(
                model, optimizer, scheduler, step, epoch, ctx["action_norm"], H_T_NORM_CONSTANTS,
                ctx["format_registry"], cfg, git_sha(),
            )
            ckpt_mgr.save(step, ckpt, heldout_l_action=heldout_l_action)

        if stopping:
            break

    return model
