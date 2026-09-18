"""Shared forward pass — the single source of truth for one model call.

Used by the training step, validate.py, and probe.py so all three agree on
what the model computes. Not in implementation.md §9's file list, but
factored out to avoid duplicating this logic three times (which would drift).
"""
import torch

from train.schedules import grl_lambda, kl_warmup, w_attract


def forward_batch(model, batch: dict, cfg: dict, step: int, device) -> dict:
    patch_signal = batch["patch_signal"].to(device)
    patch_validity = batch["patch_validity"].to(device)
    patch_pos = batch["patch_pos"].to(device)
    patch_normal = batch["patch_normal"].to(device)
    patch_link = batch["patch_link"].to(device)
    patch_mask = batch["patch_mask"].to(device)
    proprio = batch["proprio"].to(device)
    frames = batch["frames"].to(device)
    h_T_norm = batch["h_T_norm"].to(device)
    actions_norm = batch["actions_norm"].to(device)
    dof_mask = batch["dof_mask"].to(device)
    embodiment_ids = batch["embodiment_id"]

    tokens = model.tokenizer(patch_signal, patch_validity, patch_pos, patch_normal, patch_link, patch_mask)
    vision_tokens, view_mask = model.vision(frames)

    mu = logvar = p_mu = p_logvar = attract_out = None
    if cfg.get("use_h_encoder", False):
        mu, logvar = model.h_encoder.encode(tokens, patch_mask)
        h = model.h_encoder.reparameterize(mu, logvar, sample=model.training)
        p_mu, p_logvar = model.h_encoder.prior(h_T_norm)
        attract_out = model.h_encoder.attract(h)
    else:
        h = h_T_norm

    z_t, z_seq, key_padding_mask = model.state_encoder(tokens, patch_mask, proprio, vision_tokens, view_mask, h)
    pred = model.decoder(z_seq, key_padding_mask, embodiment_ids)

    adv_out = None
    lam = grl_lambda(step, ramp=cfg.get("grl_ramp", 10_000), lam_max=cfg.get("lam_max", 0.3))
    if cfg.get("use_adversary", False):
        adv_out = model.adversary(z_t, lam)

    return {
        "pred": pred,
        "actions_norm": actions_norm,
        "dof_mask": dof_mask,
        "h_T_norm": h_T_norm,
        "mu": mu, "logvar": logvar, "p_mu": p_mu, "p_logvar": p_logvar,
        "attract_out": attract_out, "adv_out": adv_out,
        "z_t": z_t,
        "lam": lam,
        "beta": kl_warmup(step, warmup=cfg.get("kl_warmup_steps", 10_000), beta_kl=cfg.get("beta_kl", 1e-4)),
        "w_att": w_attract(step, switch=cfg.get("w_attract_switch", 5_000)),
        "format_id": batch["format_id"],
        "embodiment_id": embodiment_ids,
    }
