"""Step-dependent coefficients. implementation.md §5."""
import math


def grl_lambda(step: int, ramp: int = 10_000, lam_max: float = 0.3) -> float:
    p = min(step / ramp, 1.0)
    return lam_max * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)


def kl_warmup(step: int, warmup: int = 10_000, beta_kl: float = 1e-4) -> float:
    return beta_kl * min(step / warmup, 1.0)


def w_attract(step: int, switch: int = 5_000, w_before: float = 1.0, w_after: float = 0.1) -> float:
    return w_before if step < switch else w_after
