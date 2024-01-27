import torch


def rescale(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-5,
    min: float = 0.0,
    max: float = 1.0,
) -> torch.Tensor:
    x_min, x_max = x.amin(dim=dim, keepdim=True), x.amax(dim=dim, keepdim=True)
    return min + (max - min) * (x - x_min + eps) / (x_max - x_min + 2 * eps)
