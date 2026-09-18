"""Vision backbone: frozen ViT-S/16 + trainable projection and attention pool.

implementation.md §4.4. Architecture matches DINOv3 ViT-S/16 (patch 16,
embed_dim 384, depth 12, 6 heads) so real DINOv3 weights load directly via
`load_dinov3_hub_weights`; falls back to random init (frozen regardless) so
this module is testable offline.
"""
import warnings

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_frames(frames_uint8: torch.Tensor) -> torch.Tensor:
    """(B, V, 224, 224, 3) uint8 -> (B, V, 3, 224, 224) float, ImageNet-normalized."""
    x = frames_uint8.float() / 255.0
    x = x.permute(0, 1, 4, 2, 3).contiguous()  # (B, V, 3, H, W)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 1, 3, 1, 1)
    return (x - mean) / std


class _ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTSmallPatch16(nn.Module):
    """ViT-S/16, patch-token output only (no CLS) — matches DINOv3 ViT-S/16 shapes."""

    def __init__(self, img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([_ViTBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # (B, embed_dim, H/16, W/16)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def load_dinov3_hub_weights(model: ViTSmallPatch16, repo: str = "facebookresearch/dinov3", entry: str = "dinov3_vits16"):
    """Best-effort: copy matching weights from the official DINOv3 hub checkpoint.

    Requires network access. Falls back to leaving `model` at its random init
    (still frozen) if the hub is unreachable — training can proceed with a
    fixed random projection while real weights are fetched separately.
    """
    try:
        hub_model = torch.hub.load(repo, entry, pretrained=True)
        state_dict = hub_model.state_dict()
    except Exception as exc:  # noqa: BLE001 - hub load has many failure modes, all non-fatal here
        warnings.warn(f"could not load DINOv3 weights from hub ({exc}); using random init")
        return model
    own_state = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in own_state and v.shape == own_state[k].shape}
    own_state.update(matched)
    model.load_state_dict(own_state)
    warnings.warn(f"loaded {len(matched)}/{len(own_state)} tensors from DINOv3 hub checkpoint")
    return model


class AttentionPool(nn.Module):
    """Learned query-attention pooling: 196 patch tokens -> 16 queries. Not average pooling."""

    def __init__(self, d_model: int = 512, num_queries: int = 16, num_heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, kv: torch.Tensor) -> torch.Tensor:
        b = kv.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return out


class VisionEncoder(nn.Module):
    def __init__(self, d_model: int = 512, backbone_dim: int = 384, num_queries: int = 16, view_dropout_p: float = 0.1):
        super().__init__()
        self.backbone = ViTSmallPatch16(embed_dim=backbone_dim)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.proj = nn.Linear(backbone_dim, d_model)
        self.pool = AttentionPool(d_model, num_queries)
        self.view_dropout_p = view_dropout_p

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # frozen backbone stays in eval mode regardless
        return self

    def forward(self, frames_uint8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """frames_uint8 (B, V, 224, 224, 3) -> tokens (B, V, 16, d_model), view_mask (B, V) bool."""
        b, v = frames_uint8.shape[:2]
        x = preprocess_frames(frames_uint8)
        x = x.reshape(b * v, *x.shape[2:])
        with torch.no_grad():
            feats = self.backbone(x)  # (B*V, 196, backbone_dim)
        feats = self.proj(feats)  # (B*V, 196, d_model)
        pooled = self.pool(feats)  # (B*V, num_queries, d_model)
        pooled = pooled.reshape(b, v, *pooled.shape[1:])

        view_mask = torch.ones(b, v, dtype=torch.bool, device=frames_uint8.device)
        if self.training and self.view_dropout_p > 0:
            drop = torch.rand(b, v, device=frames_uint8.device) < self.view_dropout_p
            view_mask = ~drop
            pooled = pooled * view_mask[:, :, None, None].float()
        return pooled, view_mask
