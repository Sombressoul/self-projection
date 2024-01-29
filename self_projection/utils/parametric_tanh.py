import torch
import torch.nn as nn
import torch.nn.functional as F


class ParametricTanh(nn.Module):
    def __init__(
        self,
        gamma_min: float = +0.1,
        gamma_max: float = +1.1,
        std: float = 0.01,
        use_beta: bool = True,
    ) -> None:
        super(ParametricTanh, self).__init__()

        # The range [+0.1, +1.1] is enough for stability with enforcing of a non-linearity.
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        beta = torch.empty([1])
        beta = torch.nn.init.normal_(beta, mean=0.0, std=std)
        gamma = torch.empty([2])
        gamma = torch.nn.init.normal_(gamma, mean=0.5, std=std)
        self.beta = nn.Parameter(beta) if use_beta else 0.0
        self.gamma = nn.Parameter(gamma)

        self.register_parameter_constraint_hooks()
        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x * (1 - self.gamma[0]) + F.tanh(x) * self.gamma[1] + self.beta

    def enforce_gamma_constraints(self) -> None:
        with torch.no_grad():
            self.gamma.clamp_(self.gamma_min, self.gamma_max)
        pass

    def register_parameter_constraint_hooks(self) -> None:
        self.gamma.register_hook(lambda grad: self.enforce_gamma_constraints())
        pass
