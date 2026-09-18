"""Loss terms. implementation.md §5."""
import torch
import torch.nn.functional as F

from models.h_encoder import kl_gaussian_per_dim


def masked_l1_action_loss(pred: torch.Tensor, target: torch.Tensor, dof_mask: torch.Tensor) -> torch.Tensor:
    """pred, target: (B, H, max_dof) fp32. dof_mask: (B, max_dof) bool, constant across H."""
    mask = dof_mask[:, None, :].float()
    diff = (pred.float() - target.float()).abs() * mask
    return diff.sum() / mask.sum().clamp(min=1.0)


def kl_loss(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    return kl_gaussian_per_dim(mu_q, logvar_q, mu_p, logvar_p).sum(dim=-1).mean()


def compute_losses(
    pred: torch.Tensor,
    actions_norm: torch.Tensor,
    dof_mask: torch.Tensor,
    h_T_norm: torch.Tensor,
    step: int,
    cfg: dict,
    attract_out: torch.Tensor | None = None,
    adv_out: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    logvar: torch.Tensor | None = None,
    p_mu: torch.Tensor | None = None,
    p_logvar: torch.Tensor | None = None,
    w_att: float = 1.0,
    beta: float = 1e-4,
) -> dict[str, torch.Tensor]:
    l_action = masked_l1_action_loss(pred, actions_norm, dof_mask)
    total = l_action
    out = {"L_action": l_action}

    if cfg.get("use_h_encoder", False):
        l_attract = F.mse_loss(attract_out, h_T_norm)
        l_kl = kl_loss(mu, logvar, p_mu, p_logvar)
        total = total + w_att * l_attract + beta * l_kl
        out["L_attract"] = l_attract
        out["L_kl"] = l_kl

    if cfg.get("use_adversary", False):
        l_adv = F.mse_loss(adv_out, h_T_norm)
        total = total + l_adv
        out["L_adv"] = l_adv

    out["L_total"] = total
    return out
