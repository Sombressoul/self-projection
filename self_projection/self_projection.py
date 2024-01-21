import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
from collections.abc import Callable


class SelfProjection(nn.Module):
    # Configurable params.
    size_input: Union[torch.Size, list[int]]
    size_projection: int
    depth: int
    eps: float
    delta: float
    initializer: Callable[[torch.Tensor], torch.Tensor]
    activation: Callable[[torch.Tensor], torch.Tensor]

    # Normalizations params.
    gamma_o: nn.Parameter
    gamma_p: nn.Parameter
    gamma: nn.Parameter
    beta_o: nn.Parameter
    beta_p: nn.Parameter
    beta: nn.Parameter

    # Permutations params.
    original_xj_y: nn.Parameter
    original_xi_y: nn.Parameter
    permuted_xj_y: nn.Parameter
    permuted_xi_y: nn.Parameter

    def __init__(
        self,
        size_input: Union[torch.Size, list[int]],
        size_projection: int,
        depth: int = 1,
        initializer: Callable[[torch.Tensor], torch.Tensor] = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
        eps: float = 1e-5,
        delta: float = 5.0e-2,
        **kwargs,
    ) -> None:
        super(SelfProjection, self).__init__(**kwargs)

        # Define configurable parameters.
        self.size_input = (
            size_input if isinstance(size_input, torch.Size) else torch.Size(size_input)
        )
        self.size_projection = size_projection
        self.depth = depth
        self.initializer = (
            initializer if initializer is not None else self._default_initializer
        )
        self.activation = (
            activation if activation is not None else self._default_activation
        )
        self.eps = eps
        self.delta = delta

        # Define trainable parameters: normalization scale & bias.
        self.gamma_o = nn.Parameter(torch.ones([size_projection, size_projection]))
        self.gamma_p = nn.Parameter(torch.ones([size_projection, size_projection]))
        self.gamma = nn.Parameter(torch.ones([size_projection, size_projection]))

        self.beta_o = nn.Parameter(torch.zeros([size_projection, size_projection]))
        self.beta_p = nn.Parameter(torch.zeros([size_projection, size_projection]))
        self.beta = nn.Parameter(torch.zeros([size_projection, size_projection]))

        # Define trainable parameters: permutation matrices.
        original_xj_y = torch.empty(
            [
                self.depth,
                self.size_input[1],
                self.size_projection,
            ]
        )
        original_xj_y = self._initialize(original_xj_y)
        self.original_xj_y = nn.Parameter(original_xj_y)

        original_xi_y = torch.empty(
            [
                self.depth,
                self.size_input[0],
                self.size_projection,
            ]
        )
        original_xi_y = self._initialize(original_xi_y)
        self.original_xi_y = nn.Parameter(original_xi_y)

        permuted_xj_y = torch.empty(
            [
                self.depth,
                self.size_input[0],
                self.size_projection,
            ]
        )
        permuted_xj_y = self._initialize(permuted_xj_y)
        self.permuted_xj_y = nn.Parameter(permuted_xj_y)

        permuted_xi_y = torch.empty(
            [
                self.depth,
                self.size_input[1],
                self.size_projection,
            ]
        )
        permuted_xi_y = self._initialize(permuted_xi_y)
        self.permuted_xi_y = nn.Parameter(permuted_xi_y)

        # Define trainable parameters: accumulation matrices.
        accumulator_original = torch.zeros([self.size_projection, self.size_projection])
        self.accumulator_original = nn.Parameter(accumulator_original)

        accumulator_permuted = torch.zeros([self.size_projection, self.size_projection])
        self.accumulator_permuted = nn.Parameter(accumulator_permuted)

        # Init submodules.
        p = 1.0 - math.exp(-math.fabs(self.delta) * (self.depth - 1))
        self.dropout = nn.Dropout(p=p)

        pass

    def _default_initializer(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return nn.init.xavier_uniform_(x, gain=nn.init.calculate_gain("relu"))

    def _default_activation(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return F.tanh(x)

    def _initialize(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.initializer(x)

    def _normalize(
        self,
        x: torch.FloatTensor,
        dims: list[int],
        gamma: nn.Parameter,
        beta: nn.Parameter,
    ) -> torch.FloatTensor:
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True)
        y = (x - mean) / (std + self.eps)
        norm = gamma * y + beta
        return norm

    def _extract(
        self,
        x: torch.FloatTensor,
        mat_xj: nn.Parameter,
        mat_xi: nn.Parameter,
        accumulator: nn.Parameter,
    ) -> tuple[torch.FloatTensor]:
        for depth in range(self.depth):
            x_buf = x.clone()
            x_buf = self.dropout(x_buf)
            x_buf = x_buf @ mat_xj[depth]
            x_sum = x_buf.sum(dim=-2)
            x_sum = x_sum.sub(x_sum.min()).add(self.eps).log()
            x_sum = F.softmax(x_sum, dim=-1).unsqueeze(-1)
            x_buf = x_buf.permute([0, -1, -2]) @ mat_xi[depth]
            x_buf = x_buf.permute([0, -1, -2])
            x_buf = self.activation(x_buf)
            x_buf = x_buf.mul(x_sum)
            accumulator = accumulator.add(x_buf)
        x = accumulator
        return x

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Define initial projections.
        original = x
        permuted = x.permute([0, -1, -2])

        # Original projection.
        original_yy = self._extract(
            x=original,
            mat_xj=self.original_xj_y,
            mat_xi=self.original_xi_y,
            accumulator=self.accumulator_original,
        )
        original_yy = self._normalize(
            x=original_yy,
            dims=[-1, -2],
            gamma=self.gamma_o,
            beta=self.beta_o,
        )

        # Permuted projection.
        permuted_yy = self._extract(
            x=permuted,
            mat_xj=self.permuted_xj_y,
            mat_xi=self.permuted_xi_y,
            accumulator=self.accumulator_permuted,
        )
        permuted_yy = self._normalize(
            x=permuted_yy,
            dims=[-1, -2],
            gamma=self.gamma_p,
            beta=self.beta_p,
        )

        # Self-project.
        projected = original_yy.add(permuted_yy.permute([0, -1, -2]))
        projected = self._normalize(
            x=projected,
            dims=[-1, -2],
            gamma=self.gamma,
            beta=self.beta,
        )

        return projected
