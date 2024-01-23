import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    mat_original_xj_y: nn.Parameter
    mat_original_xi_y: nn.Parameter
    mat_permuted_xj_y: nn.Parameter
    mat_permuted_xi_y: nn.Parameter
    mat_original_rel_xj_y: nn.Parameter
    mat_original_rel_xi_y: nn.Parameter
    mat_permuted_rel_xj_y: nn.Parameter
    mat_permuted_rel_xi_y: nn.Parameter

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

        mat_original_xj_y = torch.empty(t_src_shape_xj)
        mat_original_xj_y = self._initialize(mat_original_xj_y)
        self.mat_original_xj_y = nn.Parameter(mat_original_xj_y)

        mat_original_xi_y = torch.empty(t_src_shape_xi)
        mat_original_xi_y = self._initialize(mat_original_xi_y)
        self.mat_original_xi_y = nn.Parameter(mat_original_xi_y)

        mat_permuted_xj_y = torch.empty(t_src_shape_xi)
        mat_permuted_xj_y = self._initialize(mat_permuted_xj_y)
        self.mat_permuted_xj_y = nn.Parameter(mat_permuted_xj_y)

        mat_permuted_xi_y = torch.empty(t_src_shape_xj)
        mat_permuted_xi_y = self._initialize(mat_permuted_xi_y)
        self.mat_permuted_xi_y = nn.Parameter(mat_permuted_xi_y)

        # Define trainable parameters: relation matrices.
        t_src_shape_xi = [self.depth, self.size_input[0], self.size_projection]
        t_src_shape_xj = [self.depth, self.size_input[1], self.size_projection]

        mat_original_rel_xj_y = torch.empty(t_src_shape_xj)
        mat_original_rel_xj_y = self._initialize(mat_original_rel_xj_y)
        self.mat_original_rel_xj_y = nn.Parameter(mat_original_rel_xj_y)

        mat_original_rel_xi_y = torch.empty(t_src_shape_xi)
        mat_original_rel_xi_y = self._initialize(mat_original_rel_xi_y)
        self.mat_original_rel_xi_y = nn.Parameter(mat_original_rel_xi_y)

        t_src_shape_xi = [self.depth, self.size_projection, self.size_projection]
        t_src_shape_xj = [self.depth, self.size_projection, self.size_projection]

        mat_permuted_rel_xj_y = torch.empty(t_src_shape_xi)
        mat_permuted_rel_xj_y = self._initialize(mat_permuted_rel_xj_y)
        self.mat_permuted_rel_xj_y = nn.Parameter(mat_permuted_rel_xj_y)

        mat_permuted_rel_xi_y = torch.empty(t_src_shape_xj)
        mat_permuted_rel_xi_y = self._initialize(mat_permuted_rel_xi_y)
        self.mat_permuted_rel_xi_y = nn.Parameter(mat_permuted_rel_xi_y)

        # Define trainable parameters: projection scaling.
        t_src_shape_ij = [self.depth, 1]

        mat_projected_gamma = torch.ones(t_src_shape_ij)
        self.mat_projected_gamma = nn.Parameter(mat_projected_gamma)

        mat_projected_beta = torch.zeros(t_src_shape_ij)
        self.mat_projected_beta = nn.Parameter(mat_projected_beta)

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

    def _standardize(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x_mean = x.mean(dim=[-1, -2], keepdim=True)
        x_std = x.std(dim=[-1, -2], keepdim=True)
        return (x - x_mean) / (x_std + self.eps)

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
            o_mat_xj = self.mat_original_xj_y[depth]
            o_mat_xi = self.mat_original_xi_y[depth]
            o_mat_rel_xj = self.mat_original_rel_xj_y[depth]
            o_mat_rel_xi = self.mat_original_rel_xi_y[depth]
            p_mat_xj = self.mat_permuted_xj_y[depth]
            p_mat_xi = self.mat_permuted_xi_y[depth]
            p_mat_rel_xj = self.mat_permuted_rel_xj_y[depth]
            p_mat_rel_xi = self.mat_permuted_rel_xi_y[depth]
            mat_projected_gamma = self.mat_projected_gamma[depth]
            mat_projected_beta = self.mat_projected_beta[depth]

            # Compute original relation matrices.
            o_mat_rel_xj_buf = o_mat_rel_xj
            o_rel_xj_buf = self.dropout(x) @ o_mat_rel_xj_buf
            o_rel_xj_buf = (
                o_rel_xj_buf.flatten(1).softmax(dim=1).reshape(o_rel_xj_buf.shape)
            )
            o_rel_xj_sum = o_rel_xj_buf.sum(dim=-2)

            o_mat_rel_xi_buf = o_mat_rel_xi
            o_rel_xi_buf = self.dropout(x).permute([0, -1, -2]) @ o_mat_rel_xi_buf
            o_rel_xi_buf = (
                o_rel_xi_buf.flatten(1).softmax(dim=1).reshape(o_rel_xi_buf.shape)
            )
            o_rel_xi_sum = o_rel_xi_buf.sum(dim=-2)

            # Transform original matrices.
            o_trans_xj_buf = self.dropout(x) @ o_mat_xj
            o_trans_xj_buf = F.tanh(o_trans_xj_buf)
            o_trans_xj_buf = self._standardize(o_trans_xj_buf)
            o_trans_xi_buf = self.dropout(x).permute([0, -1, -2]) @ o_mat_xi
            o_trans_xi_buf = F.tanh(o_trans_xi_buf)
            o_trans_xi_buf = self._standardize(o_trans_xi_buf)

            # Transform permuted matrices.
            p_trans_xj_buf = o_trans_xj_buf.permute([0, -1, -2]) @ p_mat_xj
            p_trans_xj_buf = F.tanh(p_trans_xj_buf)
            p_trans_xj_buf = self._standardize(p_trans_xj_buf)
            p_trans_xi_buf = o_trans_xi_buf.permute([0, -1, -2]) @ p_mat_xi
            p_trans_xi_buf = p_trans_xi_buf.permute([0, -1, -2])  # permute back
            p_trans_xi_buf = F.tanh(p_trans_xi_buf)
            p_trans_xi_buf = self._standardize(p_trans_xi_buf)

            # Compute permuted relation matrices.
            p_mat_rel_xj_buf = p_mat_rel_xj
            p_rel_xj_buf = p_trans_xj_buf @ p_mat_rel_xj_buf
            p_rel_xj_buf = (
                p_rel_xj_buf.flatten(1).softmax(dim=1).reshape(p_rel_xj_buf.shape)
            )
            p_rel_xj_sum = p_rel_xj_buf.sum(dim=-2)

            p_mat_rel_xi_buf = p_mat_rel_xi
            p_rel_xi_buf = p_trans_xi_buf @ p_mat_rel_xi_buf
            p_rel_xi_buf = (
                p_rel_xi_buf.flatten(1).softmax(dim=1).reshape(p_rel_xi_buf.shape)
            )
            p_rel_xi_sum = p_rel_xi_buf.sum(dim=-2)

            # Calculate feature-rescaling factors.
            f_scale_j = (o_rel_xj_sum / p_rel_xj_sum).sqrt()
            f_scale_i = (o_rel_xi_sum / p_rel_xi_sum).sqrt()

            # Rescale permuted matrices.
            xj_buf = p_trans_xj_buf * f_scale_j.unsqueeze(-1)
            xi_buf = p_trans_xi_buf * f_scale_i.unsqueeze(-1)

            # Combine, scale and apply initial distribution.
            x_buf = xj_buf * xi_buf.permute([0, -1, -2])
            x_buf = F.tanh(x_buf)
            x_buf = self._standardize(x_buf)
            x_buf = (x_buf * x_origin_std) + x_origin_mean
            x_buf = (x_buf * mat_projected_gamma) + mat_projected_beta

            # Scale down in accordance to overall depth.
            x_buf = x_buf * (1.0 / self.depth)

            # Accumulate values.
            projection = projection.add(x_buf)

        return projection
