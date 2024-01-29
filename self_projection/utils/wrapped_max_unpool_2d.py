import torch
import torch.nn as nn


# Dirty hack: nn.Sequential takes only one positional argument.
class WrappedMaxUnpool2d(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(WrappedMaxUnpool2d, self).__init__()
        self.unpool = nn.MaxUnpool2d(*args, **kwargs)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.unpool(x[0], x[1])
