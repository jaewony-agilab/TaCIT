"""State Encoder: FiLM-conditioned transformer over tactile+proprio+vision. implementation.md §4.3."""
import torch
import torch.nn as nn

TYPE_TACTILE, TYPE_PROPRIO, TYPE_VISION = 0, 1, 2


class FiLMTransformerLayer(nn.Module):
    """FiLM applied after each sublayer's LayerNorm, before the sublayer op — on both
    attention and FFN sublayers. Zero-init FiLM generator: untrained, gamma=1/beta=0 exactly."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float, d_h: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.film1 = nn.Linear(d_h, 2 * d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model)
        )
        self.film2 = nn.Linear(d_h, 2 * d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.d_model = d_model
        for film in (self.film1, self.film2):
            nn.init.zeros_(film.weight)
            with torch.no_grad():
                film.bias.copy_(torch.cat([torch.ones(d_model), torch.zeros(d_model)]))

    @staticmethod
    def _film(film_layer: nn.Linear, x_normed: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gamma, beta = film_layer(h).chunk(2, dim=-1)  # each (B, d_model)
        return gamma[:, None, :] * x_normed + beta[:, None, :]

    def forward(self, x: torch.Tensor, h: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        modulated = self._film(self.film1, self.norm1(x), h)
        attn_out, _ = self.attn(modulated, modulated, modulated, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout1(attn_out)

        modulated2 = self._film(self.film2, self.norm2(x), h)
        x = x + self.dropout2(self.ffn(modulated2))
        return x


class StateEncoder(nn.Module):
    """d_h is config-driven, not fixed at 32: stages 1-2 feed h_T_norm (16-dim) straight
    into FiLM, so their configs set d_h=16; stage 3 samples a learned 32-dim h and sets
    d_h=32. See implementation.md §6.4 vs §4 — reconciled at the config layer."""

    def __init__(
        self,
        max_dof: int,
        d_tok: int = 256,
        d_model: int = 512,
        d_h: int = 32,
        num_layers: int = 8,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        num_views: int = 2,
        num_vision_queries: int = 16,
    ):
        super().__init__()
        self.num_vision_queries = num_vision_queries
        self.tactile_proj = nn.Linear(d_tok, d_model)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(2 * max_dof, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, 2 * d_model)
        )
        self.type_embed = nn.Embedding(3, d_model)
        self.view_embed = nn.Embedding(num_views, d_model)
        self.layers = nn.ModuleList(
            [FiLMTransformerLayer(d_model, num_heads, ffn_dim, dropout, d_h) for _ in range(num_layers)]
        )

    def forward(
        self,
        tokens: torch.Tensor,        # (B, K, d_tok)
        patch_mask: torch.Tensor,    # (B, K) bool
        proprio: torch.Tensor,       # (B, 2 * max_dof)
        vision_tokens: torch.Tensor, # (B, V, num_vision_queries, d_model)
        view_mask: torch.Tensor,     # (B, V) bool
        h: torch.Tensor,             # (B, d_h)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, k = tokens.shape[:2]
        v = vision_tokens.shape[1]
        nq = self.num_vision_queries
        device = tokens.device

        tactile = self.tactile_proj(tokens)  # (B, K, d_model)
        proprio_out = self.proprio_mlp(proprio).reshape(b, 2, -1)  # (B, 2, d_model)
        vision_flat = vision_tokens.reshape(b, v * nq, -1)  # (B, V*nq, d_model)

        seq = torch.cat([tactile, proprio_out, vision_flat], dim=1)  # (B, L, d_model)

        type_ids = torch.cat(
            [
                torch.full((k,), TYPE_TACTILE, dtype=torch.long, device=device),
                torch.full((2,), TYPE_PROPRIO, dtype=torch.long, device=device),
                torch.full((v * nq,), TYPE_VISION, dtype=torch.long, device=device),
            ]
        )
        seq = seq + self.type_embed(type_ids)[None]

        view_ids = torch.arange(v, device=device).repeat_interleave(nq)  # (V*nq,)
        view_e = self.view_embed(view_ids)[None]  # (1, V*nq, d_model)
        seq = torch.cat([seq[:, : k + 2], seq[:, k + 2 :] + view_e], dim=1)

        vision_key_pad = (~view_mask).repeat_interleave(nq, dim=1)  # (B, V*nq)
        proprio_key_pad = torch.zeros(b, 2, dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([~patch_mask, proprio_key_pad, vision_key_pad], dim=1)  # (B, L)

        for layer in self.layers:
            seq = layer(seq, h, key_padding_mask)

        valid_f = (~key_padding_mask).unsqueeze(-1).float()
        z_t = (seq * valid_f).sum(1) / valid_f.sum(1).clamp(min=1.0)
        return z_t, seq, key_padding_mask
