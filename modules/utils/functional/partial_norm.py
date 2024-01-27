import torch


def partial_norm(
    x: torch.Tensor,
    fraction: float = 0.1,
    gamma: torch.Tensor = None,
    beta: torch.Tensor = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    sample_numel = x[0].numel()
    partial_size = int(sample_numel * fraction)
    partial_size = 1 if partial_size == 0 else partial_size
    indices = torch.randint(0, sample_numel, [x.shape[0], partial_size])
    indices = [indices[i].add_(sample_numel * i) for i in range(indices.shape[0])]
    indices = torch.cat(indices, dim=0)
    partial_x = x.reshape([-1])[indices].view([-1, partial_size])
    # It is better to use .view() above, but that may cause memory access errors.
    # I have to climb under the hood to fix this, but I don't want to. \/(o_O)\/
    norm_x = partial_x.norm(p="fro", dim=-1, keepdim=True)
    rms_x = norm_x * partial_size ** (-1.0 / 2)
    rms_x = rms_x.view([-1] + [1] * (len(x.shape) - 1))
    x = x / (rms_x + eps)
    x = x * gamma if gamma is not None else x
    x = x + beta if beta is not None else x
    return x
