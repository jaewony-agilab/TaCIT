"""Canonical Tokenizer. implementation.md §4.1."""
import math

import torch
import torch.nn as nn

FOURIER_BANDS = 8
FOURIER_SCALE = 1.0 / 0.1
LINK_EMBED_DIM = 16


class FourierPositionEncoding(nn.Module):
    """p (..., 3) -> (..., 48): L=8 bands, sin+cos, 3 coords -> 3*8*2 = 48."""

    def __init__(self, num_bands: int = FOURIER_BANDS, scale: float = FOURIER_SCALE):
        super().__init__()
        self.register_buffer("bands", 2.0 ** torch.arange(num_bands).float() * math.pi, persistent=False)
        self.scale = scale

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        scaled = pos * self.scale  # (..., 3)
        args = scaled.unsqueeze(-1) * self.bands  # (..., 3, L)
        enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (..., 3, 2L)
        return enc.reshape(*pos.shape[:-1], -1)  # (..., 3 * 2L) = (..., 48)


class CanonicalTokenizer(nn.Module):
    def __init__(self, num_links: int = 32, W: int = 8, d_tok: int = 256):
        super().__init__()
        self.fourier = FourierPositionEncoding()
        # +1: index 0 reserved for padding (patch_link == -1 -> shifted to 0).
        self.link_embed = nn.Embedding(num_links + 1, LINK_EMBED_DIM, padding_idx=0)
        in_dim = W * 3 + 3 + 48 + 3 + LINK_EMBED_DIM  # = 94 for W=8
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_tok),
            nn.LayerNorm(d_tok),
            nn.GELU(),
            nn.Linear(d_tok, d_tok),
        )

    def forward(
        self,
        patch_signal: torch.Tensor,    # (B, K, W*3)
        patch_validity: torch.Tensor,  # (B, K, 3) bool
        patch_pos: torch.Tensor,       # (B, K, 3)
        patch_normal: torch.Tensor,    # (B, K, 3)
        patch_link: torch.Tensor,      # (B, K) long, -1 for empty
        patch_mask: torch.Tensor,      # (B, K) bool
    ) -> torch.Tensor:
        link_idx = (patch_link + 1).clamp(min=0)
        link_e = self.link_embed(link_idx)  # (B, K, 16)
        pos_f = self.fourier(patch_pos)  # (B, K, 48)
        token_in = torch.cat(
            [patch_signal, patch_validity.float(), pos_f, patch_normal, link_e], dim=-1
        )  # (B, K, 94)
        tokens = self.proj(token_in)  # (B, K, d_tok)
        return tokens * patch_mask.unsqueeze(-1).float()
