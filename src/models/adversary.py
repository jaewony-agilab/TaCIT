"""GRL adversary predicting the sensor descriptor from z_t. implementation.md §4.6, §7.1."""
import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambda_ * grad, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return GradientReversal.apply(x, lambda_)


class Adversary(nn.Module):
    def __init__(self, d_model: int = 512, h_T_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, h_T_dim),
        )

    def forward(self, z_t: torch.Tensor, lambda_: float) -> torch.Tensor:
        return self.net(grad_reverse(z_t, lambda_))
