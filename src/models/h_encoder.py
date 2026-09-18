"""h-encoder + conditional prior + attraction head. implementation.md §4.2."""
import torch
import torch.nn as nn


class HEncoder(nn.Module):
    def __init__(
        self,
        d_tok: int = 256,
        d_h: int = 32,
        h_T_dim: int = 16,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 1024,
    ):
        super().__init__()
        self.d_h = d_h
        layer = nn.TransformerEncoderLayer(
            d_model=d_tok, nhead=num_heads, dim_feedforward=ffn_dim,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.pool_proj = nn.Linear(d_tok, 2 * d_h)
        self.prior_net = nn.Sequential(nn.Linear(h_T_dim, 128), nn.GELU(), nn.Linear(128, 2 * d_h))
        self.attract_head = nn.Sequential(nn.Linear(d_h, 128), nn.GELU(), nn.Linear(128, h_T_dim))

    def encode(self, tokens: torch.Tensor, patch_mask: torch.Tensor):
        """tokens (B, K, d_tok), patch_mask (B, K) bool -> mu, logvar each (B, d_h)."""
        key_padding_mask = ~patch_mask  # True = ignore, per nn.Transformer convention
        h_seq = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        mask_f = patch_mask.unsqueeze(-1).float()
        pooled = (h_seq * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        mu, logvar = self.pool_proj(pooled).chunk(2, dim=-1)
        return mu, logvar

    def prior(self, h_T_norm: torch.Tensor):
        """h_T_norm (B, h_T_dim) -> p_mu, p_logvar each (B, d_h)."""
        return self.prior_net(h_T_norm).chunk(2, dim=-1)

    def attract(self, h: torch.Tensor) -> torch.Tensor:
        return self.attract_head(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)


def kl_gaussian_per_dim(
    mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor
) -> torch.Tensor:
    """KL(q||p) between diagonal Gaussians, per dimension. Closed form, no sampling."""
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * (logvar_p - logvar_q) + (var_q + (mu_q - mu_p) ** 2) / (2 * var_p) - 0.5
