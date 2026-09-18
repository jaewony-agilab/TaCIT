"""ACT-style action decoder with per-embodiment output heads. implementation.md §4.5."""
import torch
import torch.nn as nn


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor) -> torch.Tensor:
        h1 = self.norm1(x)
        attn_out, _ = self.self_attn(h1, h1, h1, need_weights=False)
        x = x + self.dropout1(attn_out)

        h2 = self.norm2(x)
        cross_out, _ = self.cross_attn(
            h2, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=False
        )
        x = x + self.dropout2(cross_out)

        x = x + self.dropout3(self.ffn(self.norm3(x)))
        return x


class ActionDecoder(nn.Module):
    def __init__(
        self,
        embodiments: dict[str, int],
        H: int = 100,
        d_model: int = 512,
        num_layers: int = 7,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(H, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.heads = nn.ModuleDict({emb_id: nn.Linear(d_model, dof) for emb_id, dof in embodiments.items()})
        self.max_dof = max(embodiments.values())

    def forward(
        self, z_seq: torch.Tensor, memory_key_padding_mask: torch.Tensor, embodiment_ids: list[str]
    ) -> torch.Tensor:
        """Returns (B, H, max_dof) — zero-padded past each sample's true dof; use dof_mask for loss."""
        b = z_seq.shape[0]
        x = self.queries.unsqueeze(0).expand(b, -1, -1)
        for layer in self.layers:
            x = layer(x, z_seq, memory_key_padding_mask)

        out = torch.zeros(b, x.shape[1], self.max_dof, device=x.device, dtype=x.dtype)
        for emb_id in set(embodiment_ids):
            idx = torch.tensor(
                [i for i, e in enumerate(embodiment_ids) if e == emb_id], device=x.device
            )
            head_out = self.heads[emb_id](x[idx])  # (n_i, H, dof)
            out[idx, :, : head_out.shape[-1]] = head_out
        return out
