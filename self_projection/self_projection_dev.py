import math
import torch
import torch.nn as nn

from typing import Union
from collections.abc import Callable


class SelfProjectionDev(nn.Module):
    # Configurable params.
    size_input: Union[torch.Size, list[int]]
    size_projection: int
    depth: int
    eps: float
    delta: float
    initializer: Callable[[torch.Tensor], torch.Tensor]

    # Trainable params.
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
        eps: float = 1e-5,
        delta: float = 5.0e-2,
        **kwargs,
    ) -> None:
        super(SelfProjectionDev, self).__init__(**kwargs)

        # Define configurable parameters.
        self.size_input = (
            size_input if isinstance(size_input, torch.Size) else torch.Size(size_input)
        )
        self.size_projection = size_projection
        self.depth = depth
        self.initializer = (
            initializer if initializer is not None else self._default_initializer
        )
        self.eps = eps
        self.delta = delta

        # Define trainable parameters: permutation matrices.
        t_src_shape_xi = [self.depth, self.size_input[0], self.size_projection]
        t_src_shape_xj = [self.depth, self.size_input[1], self.size_projection]

        original_xj_y = torch.empty(t_src_shape_xj)
        original_xj_y = self._initialize(original_xj_y)
        self.original_xj_y = nn.Parameter(original_xj_y)

        original_xi_y = torch.empty(t_src_shape_xi)
        original_xi_y = self._initialize(original_xi_y)
        self.original_xi_y = nn.Parameter(original_xi_y)

        permuted_xj_y = torch.empty(t_src_shape_xi)
        permuted_xj_y = self._initialize(permuted_xj_y)
        self.permuted_xj_y = nn.Parameter(permuted_xj_y)

        permuted_xi_y = torch.empty(t_src_shape_xj)
        permuted_xi_y = self._initialize(permuted_xi_y)
        self.permuted_xi_y = nn.Parameter(permuted_xi_y)

        # Define trainable parameters: relation matrices.
        t_src_shape_xi = [self.depth, self.size_input[0], self.size_projection]
        t_src_shape_xj = [self.depth, self.size_input[1], self.size_projection]

        original_rel_xj_y = torch.empty(t_src_shape_xj)
        original_rel_xj_y = self._initialize(original_rel_xj_y)
        self.original_rel_xj_y = nn.Parameter(original_rel_xj_y)

        original_rel_xi_y = torch.empty(t_src_shape_xi)
        original_rel_xi_y = self._initialize(original_rel_xi_y)
        self.original_rel_xi_y = nn.Parameter(original_rel_xi_y)

        t_src_shape_xi = [self.depth, self.size_projection, self.size_projection]
        t_src_shape_xj = [self.depth, self.size_projection, self.size_projection]

        permuted_rel_xj_y = torch.empty(t_src_shape_xi)
        permuted_rel_xj_y = self._initialize(permuted_rel_xj_y)
        self.permuted_rel_xj_y = nn.Parameter(permuted_rel_xj_y)

        permuted_rel_xi_y = torch.empty(t_src_shape_xj)
        permuted_rel_xi_y = self._initialize(permuted_rel_xi_y)
        self.permuted_rel_xi_y = nn.Parameter(permuted_rel_xi_y)

        # Init submodules.
        p = 1.0 - math.exp(-math.fabs(self.delta) * (self.depth - 1))
        self.dropout = nn.Dropout(p=p)

        pass

    def _default_initializer(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return nn.init.xavier_uniform_(x, gain=nn.init.calculate_gain("relu"))

    def _initialize(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.initializer(x)

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        projection = torch.zeros(
            [x.shape[0], self.size_projection, self.size_projection]
        ).to(device=x.device, dtype=torch.float32)
                
        rate_i = x.shape[-2] / self.size_projection
        rate_j = x.shape[-1] / self.size_projection
        scale = rate_i * rate_j
        x_origin_mean = x.mean(dim=[-1, -2], keepdim=True) * scale
        x_origin_std = x.std(dim=[-1, -2], keepdim=True) * scale
        
        for depth in range(self.depth):
            o_mat_xj = self.original_xj_y[depth]
            o_mat_xi = self.original_xi_y[depth]
            o_rel_mat_xj = self.original_rel_xj_y[depth]
            o_rel_mat_xi = self.original_rel_xi_y[depth]
            p_mat_xj = self.permuted_xj_y[depth]
            p_mat_xi = self.permuted_xi_y[depth]
            p_rel_mat_xj = self.permuted_rel_xj_y[depth]
            p_rel_mat_xi = self.permuted_rel_xi_y[depth]

            # Compute original relation matrices.
            o_rel_mat_xj_buf = o_rel_mat_xj
            o_rel_xj_buf = self.dropout(x) @ o_rel_mat_xj_buf
            o_rel_xj_buf = o_rel_xj_buf.flatten(1).softmax(dim=1).reshape(o_rel_xj_buf.shape)
            o_rel_xj_sum = o_rel_xj_buf.sum(dim=-2)
            
            o_rel_mat_xi_buf = o_rel_mat_xi
            o_rel_xi_buf = self.dropout(x).permute([0, -1, -2]) @ o_rel_mat_xi_buf
            o_rel_xi_buf = o_rel_xi_buf.flatten(1).softmax(dim=1).reshape(o_rel_xi_buf.shape)
            o_rel_xi_sum = o_rel_xi_buf.sum(dim=-2)

            # Transform original matrices.
            o_trans_xj_buf = self.dropout(x) @ o_mat_xj
            o_trans_xi_buf = self.dropout(x).permute([0, -1, -2]) @ o_mat_xi

            # Transform permuted matrices.
            p_trans_xj_buf = o_trans_xj_buf.permute([0, -1, -2]) @ p_mat_xj
            p_trans_xi_buf = o_trans_xi_buf.permute([0, -1, -2]) @ p_mat_xi

            # Compute permuted relation matrices.
            p_rel_mat_xj_buf = p_rel_mat_xj
            p_rel_xj_buf = p_trans_xj_buf @ p_rel_mat_xj_buf
            p_rel_xj_buf = p_rel_xj_buf.flatten(1).softmax(dim=1).reshape(p_rel_xj_buf.shape)
            p_rel_xj_sum = p_rel_xj_buf.sum(dim=-2)

            p_rel_mat_xi_buf = p_rel_mat_xi
            p_rel_xi_buf = p_trans_xi_buf @ p_rel_mat_xi_buf
            p_rel_xi_buf = p_rel_xi_buf.flatten(1).softmax(dim=1).reshape(p_rel_xi_buf.shape)
            p_rel_xi_sum = p_rel_xi_buf.sum(dim=-2)

            # Calculate feature-rescaling factors.
            f_scale_j = (o_rel_xj_sum / p_rel_xj_sum).sqrt()
            f_scale_i = (o_rel_xi_sum / p_rel_xi_sum).sqrt()

            # Rescale permuted matrices.
            xj_buf = p_trans_xj_buf * f_scale_j.unsqueeze(-1)
            xi_buf = p_trans_xi_buf * f_scale_i.unsqueeze(-1)

            # Combine and apply initial distribution.
            x_buf = xj_buf + xi_buf.permute([0, -1, -2])
            x_buf_mean = x_buf.mean(dim=[-1, -2], keepdim=True)
            x_buf_std = x_buf.std(dim=[-1, -2], keepdim=True)
            x_buf = (x_buf - x_buf_mean) / (x_buf_std + self.eps)
            x_buf = (x_buf * x_origin_std) + x_origin_mean

            # Accumulate values.
            projection = projection.add(x_buf)
        
        return projection
