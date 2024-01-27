import torch


def standardize(
    x: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    x_mean = x.mean(dim=[-1, -2], keepdim=True)
    x_std = x.std(dim=[-1, -2], keepdim=True)
    return (x - x_mean) / (x_std + eps)
