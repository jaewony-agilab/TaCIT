"""Loss-based validation on `val` and `val_heldout_format`. implementation.md §7.1."""
import torch

from train.forward import forward_batch
from train.losses import compute_losses


@torch.no_grad()
def run_validation(model, loader, cfg: dict, step: int, device, max_batches: int = 200) -> dict:
    was_training = model.training
    model.eval()

    h = cfg["H"]
    l_action_sum = l_attract_sum = l_kl_sum = 0.0
    n = 0
    per_timestep_abs_sum = torch.zeros(h)
    per_timestep_count = torch.zeros(h)
    per_format_sum: dict[str, float] = {}
    per_format_count: dict[str, int] = {}

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        out = forward_batch(model, batch, cfg, step, device)
        losses = compute_losses(
            out["pred"], out["actions_norm"], out["dof_mask"], out["h_T_norm"], step, cfg,
            out["attract_out"], out["adv_out"], out["mu"], out["logvar"], out["p_mu"], out["p_logvar"],
            out["w_att"], out["beta"],
        )
        b = out["pred"].shape[0]
        n += b
        l_action_sum += losses["L_action"].item() * b
        if "L_attract" in losses:
            l_attract_sum += losses["L_attract"].item() * b
        if "L_kl" in losses:
            l_kl_sum += losses["L_kl"].item() * b

        mask = out["dof_mask"][:, None, :].float()  # (B, 1, max_dof)
        diff = (out["pred"].float() - out["actions_norm"].float()).abs() * mask  # (B, H, max_dof)
        denom = mask.sum(dim=(1, 2)).sum() / h  # per-timestep valid-dim count, summed over batch
        per_timestep_abs_sum += diff.sum(dim=(0, 2)).cpu()
        per_timestep_count += denom.item()

        per_sample_l1 = diff.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1)  # (B,)
        for fmt, val in zip(out["format_id"], per_sample_l1.tolist()):
            per_format_sum[fmt] = per_format_sum.get(fmt, 0.0) + val
            per_format_count[fmt] = per_format_count.get(fmt, 0) + 1

    model.train(was_training)
    return {
        "L_action": l_action_sum / max(n, 1),
        "L_attract": (l_attract_sum / max(n, 1)) if cfg.get("use_h_encoder") else None,
        "L_kl": (l_kl_sum / max(n, 1)) if cfg.get("use_h_encoder") else None,
        "per_timestep_l1": (per_timestep_abs_sum / per_timestep_count.clamp(min=1)).tolist(),
        "per_format_l1": {f: per_format_sum[f] / per_format_count[f] for f in per_format_sum},
    }
