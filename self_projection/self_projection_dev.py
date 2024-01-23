import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
from collections.abc import Callable


class ParametricTanh(nn.Module):
    def __init__(
        self,
        initial_gamma: float = 0.5,
        gamma_min: float = 0.0,
        gamma_max: float = 2.0,
    ) -> None:
        super(ParametricTanh, self).__init__()

        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gamma = nn.Parameter(torch.tensor([initial_gamma]))

        self.register_parameter_constraint_hooks()
        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x * (1 - self.gamma) + F.tanh(x) * self.gamma

    def enforce_gamma_constraints(self) -> None:
        with torch.no_grad():
            self.gamma.clamp_(self.gamma_min, self.gamma_max)
        pass

    def register_parameter_constraint_hooks(self) -> None:
        self.gamma.register_hook(lambda grad: self.enforce_gamma_constraints())
        pass


class SelfProjectionDev(nn.Module):
    # Configurable params.
    size_input: Union[torch.Size, list[int]]
    size_projection: int
    depth: int
    preserve_distribution: bool
    standardize_output: bool
    scale_and_bias: bool
    initializer: Callable[[torch.Tensor], torch.Tensor]
    activation: Callable[[torch.Tensor], torch.Tensor]
    eps: float
    delta: float

    # Trainable params.
    mat_original_xj: nn.Parameter
    mat_original_xi: nn.Parameter
    mat_permuted_xj: nn.Parameter
    mat_permuted_xi: nn.Parameter
    mat_original_rel_xj: nn.Parameter
    mat_original_rel_xi: nn.Parameter
    mat_permuted_rel_xj: nn.Parameter
    mat_permuted_rel_xi: nn.Parameter

    def __init__(
        self,
        size_input: Union[torch.Size, list[int]],
        size_projection: int,
        depth: int = 1,
        preserve_distribution: bool = False,
        standardize_output: bool = False,
        scale_and_bias: bool = False,
        initializer: Callable[[torch.Tensor], torch.Tensor] = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
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
        self.preserve_distribution = preserve_distribution
        self.standardize_output = standardize_output
        self.scale_and_bias = scale_and_bias
        self.initializer = (
            initializer if initializer is not None else self._default_initializer
        )
        self.activation = activation
        self.eps = eps
        self.delta = delta

        # Check allowed combinations of parameters.
        assert not all(
            [self.preserve_distribution, self.standardize_output]
        ), "Preserve distribution and standardize output are mutually exclusive."

        # Define trainable parameters: permutation matrices.
        t_src_shape_xi = [self.depth, self.size_input[0], self.size_projection]
        t_src_shape_xj = [self.depth, self.size_input[1], self.size_projection]

        mat_original_xj = torch.empty(t_src_shape_xj)
        mat_original_xj = self._initialize(mat_original_xj)
        self.mat_original_xj = nn.Parameter(mat_original_xj)

        mat_original_xi = torch.empty(t_src_shape_xi)
        mat_original_xi = self._initialize(mat_original_xi)
        self.mat_original_xi = nn.Parameter(mat_original_xi)

        mat_permuted_xj = torch.empty(t_src_shape_xi)
        mat_permuted_xj = self._initialize(mat_permuted_xj)
        self.mat_permuted_xj = nn.Parameter(mat_permuted_xj)

        mat_permuted_xi = torch.empty(t_src_shape_xj)
        mat_permuted_xi = self._initialize(mat_permuted_xi)
        self.mat_permuted_xi = nn.Parameter(mat_permuted_xi)

        # Define trainable parameters: relation matrices.
        t_src_shape_xi = [self.depth, self.size_input[0], self.size_projection]
        t_src_shape_xj = [self.depth, self.size_input[1], self.size_projection]

        mat_original_rel_xj = torch.empty(t_src_shape_xj)
        mat_original_rel_xj = self._initialize(mat_original_rel_xj)
        self.mat_original_rel_xj = nn.Parameter(mat_original_rel_xj)

        mat_original_rel_xi = torch.empty(t_src_shape_xi)
        mat_original_rel_xi = self._initialize(mat_original_rel_xi)
        self.mat_original_rel_xi = nn.Parameter(mat_original_rel_xi)

        t_src_shape_xi = [self.depth, self.size_projection, self.size_projection]
        t_src_shape_xj = [self.depth, self.size_projection, self.size_projection]

        mat_permuted_rel_xj = torch.empty(t_src_shape_xi)
        mat_permuted_rel_xj = self._initialize(mat_permuted_rel_xj)
        self.mat_permuted_rel_xj = nn.Parameter(mat_permuted_rel_xj)

        mat_permuted_rel_xi = torch.empty(t_src_shape_xj)
        mat_permuted_rel_xi = self._initialize(mat_permuted_rel_xi)
        self.mat_permuted_rel_xi = nn.Parameter(mat_permuted_rel_xi)

        # Init submodules.
        p = 1.0 - math.exp(-math.fabs(self.delta) * (self.depth - 1))
        self.dropout = nn.Dropout(p=p)

        # Create scale params.
        if self.scale_and_bias:
            t_src_shape_p = [self.size_projection, self.size_projection]
            self.scale: list[dict[str, nn.Parameter]] = [None] * self.depth
            for depth in range(self.depth):
                self.scale[depth] = dict(
                    o_rel_xj_buf=self._create_scale(
                        f"scale_o_rel_xj_buf_{depth}",
                        self.mat_original_rel_xj.shape[1::],
                    ),
                    o_rel_xi_buf=self._create_scale(
                        f"scale_o_rel_xi_buf_{depth}",
                        self.mat_original_rel_xi.shape[1::],
                    ),
                    o_trans_xj_buf=self._create_scale(
                        f"scale_o_trans_xj_buf_{depth}",
                        self.mat_original_xj.shape[1::],
                    ),
                    o_trans_xi_buf=self._create_scale(
                        f"scale_o_trans_xi_buf_{depth}",
                        self.mat_original_xi.shape[1::],
                    ),
                    p_trans_xj_buf=self._create_scale(
                        f"scale_p_trans_xj_buf_{depth}",
                        t_src_shape_p,
                    ),
                    p_trans_xi_buf=self._create_scale(
                        f"scale_p_trans_xi_buf_{depth}",
                        t_src_shape_p,
                    ),
                    p_rel_xj_buf=self._create_scale(
                        f"scale_p_rel_xj_buf_{depth}",
                        t_src_shape_p,
                    ),
                    p_rel_xi_buf=self._create_scale(
                        f"scale_p_rel_xi_buf_{depth}",
                        t_src_shape_p,
                    ),
                )

        # Create bias params.
        if self.scale_and_bias:
            t_src_shape_p = [self.size_projection, self.size_projection]
            self.bias: list[dict[str, nn.Parameter]] = [None] * self.depth
            for depth in range(self.depth):
                self.bias[depth] = dict(
                    o_rel_xj_buf=self._create_bias(
                        f"bias_o_rel_xj_buf_{depth}",
                        self.mat_original_rel_xj.shape[1::],
                    ),
                    o_rel_xi_buf=self._create_bias(
                        f"bias_o_rel_xi_buf_{depth}",
                        self.mat_original_rel_xi.shape[1::],
                    ),
                    o_trans_xj_buf=self._create_bias(
                        f"bias_o_trans_xj_buf_{depth}",
                        self.mat_original_xj.shape[1::],
                    ),
                    o_trans_xi_buf=self._create_bias(
                        f"bias_o_trans_xi_buf_{depth}",
                        self.mat_original_xi.shape[1::],
                    ),
                    p_trans_xj_buf=self._create_bias(
                        f"bias_p_trans_xj_buf_{depth}",
                        t_src_shape_p,
                    ),
                    p_trans_xi_buf=self._create_bias(
                        f"bias_p_trans_xi_buf_{depth}",
                        t_src_shape_p,
                    ),
                    p_rel_xj_buf=self._create_bias(
                        f"bias_p_rel_xj_buf_{depth}",
                        t_src_shape_p,
                    ),
                    p_rel_xi_buf=self._create_bias(
                        f"bias_p_rel_xi_buf_{depth}",
                        t_src_shape_p,
                    ),
                )

        # Create ParametricTahn activations.
        if self.activation is None:
            self.activations: list[dict[str, ParametricTanh]] = [None] * self.depth
            for depth in range(self.depth):
                self.activations[depth] = dict(
                    o_trans_xj_buf=self._create_activation(
                        f"activation_o_trans_xj_buf_{depth}"
                    ),
                    o_trans_xi_buf=self._create_activation(
                        f"activation_o_trans_xi_buf_{depth}"
                    ),
                    p_trans_xj_buf=self._create_activation(
                        f"activation_p_trans_xj_buf_{depth}"
                    ),
                    p_trans_xi_buf=self._create_activation(
                        f"activation_p_trans_xi_buf_{depth}"
                    ),
                    x_buf=self._create_activation(f"activation_x_buf_{depth}"),
                )

        pass

    def _create_bias(
        self,
        name: str,
        shape: list[int],
    ) -> nn.Parameter:
        p = nn.Parameter(torch.zeros(shape))
        self.register_parameter(name, p)
        return p

    def _create_scale(
        self,
        name: str,
        shape: list[int],
    ) -> nn.Parameter:
        p = nn.Parameter(torch.ones(shape))
        self.register_parameter(name, p)
        return p

    def _scale_and_bias(
        self,
        x: torch.Tensor,
        name: str,
        depth: int,
    ) -> torch.Tensor:
        if self.scale_and_bias:
            scale = self.scale[depth][name]
            bias = self.bias[depth][name]
            return x * scale + bias
        else:
            return x

    def _create_activation(
        self,
        name: str,
    ) -> ParametricTanh:
        p = ParametricTanh()
        self.register_module(name, p)
        return p

    def _activate(
        self,
        x: torch.Tensor,
        name: str,
        depth: int,
    ) -> torch.Tensor:
        if self.activation is not None:
            return self.activation(x)
        else:
            return self.activations[depth][name](x)

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
            o_mat_xj = self.mat_original_xj[depth]
            o_mat_xi = self.mat_original_xi[depth]
            o_mat_rel_xj = self.mat_original_rel_xj[depth]
            o_mat_rel_xi = self.mat_original_rel_xi[depth]
            p_mat_xj = self.mat_permuted_xj[depth]
            p_mat_xi = self.mat_permuted_xi[depth]
            p_mat_rel_xj = self.mat_permuted_rel_xj[depth]
            p_mat_rel_xi = self.mat_permuted_rel_xi[depth]

            # Compute original relation matrices.
            o_rel_xj_buf = self.dropout(x) @ o_mat_rel_xj
            o_rel_xj_buf = self._scale_and_bias(o_rel_xj_buf, "o_rel_xj_buf", depth)
            o_rel_xj_buf = (
                o_rel_xj_buf.flatten(1).softmax(dim=1).reshape(o_rel_xj_buf.shape)
            )
            o_rel_xj_sum = o_rel_xj_buf.sum(dim=-2)

            o_rel_xi_buf = self.dropout(x).permute([0, -1, -2]) @ o_mat_rel_xi
            o_rel_xi_buf = self._scale_and_bias(o_rel_xi_buf, "o_rel_xi_buf", depth)
            o_rel_xi_buf = (
                o_rel_xi_buf.flatten(1).softmax(dim=1).reshape(o_rel_xi_buf.shape)
            )
            o_rel_xi_sum = o_rel_xi_buf.sum(dim=-2)

            # Transform original matrices.
            o_trans_xj_buf = self.dropout(x) @ o_mat_xj
            o_trans_xj_buf = self._scale_and_bias(
                o_trans_xj_buf, "o_trans_xj_buf", depth
            )
            o_trans_xj_buf = self._activate(o_trans_xj_buf, "o_trans_xj_buf", depth)
            o_trans_xj_buf = self._standardize(o_trans_xj_buf)
            o_trans_xi_buf = self.dropout(x).permute([0, -1, -2]) @ o_mat_xi
            o_trans_xi_buf = self._scale_and_bias(
                o_trans_xi_buf, "o_trans_xi_buf", depth
            )
            o_trans_xi_buf = self._activate(o_trans_xi_buf, "o_trans_xi_buf", depth)
            o_trans_xi_buf = self._standardize(o_trans_xi_buf)

            # Transform permuted matrices.
            p_trans_xj_buf = o_trans_xj_buf.permute([0, -1, -2]) @ p_mat_xj
            p_trans_xj_buf = self._scale_and_bias(
                p_trans_xj_buf, "p_trans_xj_buf", depth
            )
            p_trans_xj_buf = self._activate(p_trans_xj_buf, "p_trans_xj_buf", depth)
            p_trans_xj_buf = self._standardize(p_trans_xj_buf)
            p_trans_xi_buf = o_trans_xi_buf.permute([0, -1, -2]) @ p_mat_xi
            p_trans_xi_buf = self._scale_and_bias(
                p_trans_xi_buf, "p_trans_xi_buf", depth
            )
            p_trans_xi_buf = p_trans_xi_buf.permute([0, -1, -2])  # permute back
            p_trans_xi_buf = self._activate(p_trans_xi_buf, "p_trans_xi_buf", depth)
            p_trans_xi_buf = self._standardize(p_trans_xi_buf)

            # Compute permuted relation matrices.
            p_rel_xj_buf = p_trans_xj_buf @ p_mat_rel_xj
            p_rel_xj_buf = self._scale_and_bias(p_rel_xj_buf, "p_rel_xj_buf", depth)
            p_rel_xj_buf = (
                p_rel_xj_buf.flatten(1).softmax(dim=1).reshape(p_rel_xj_buf.shape)
            )
            p_rel_xj_sum = p_rel_xj_buf.sum(dim=-2)

            p_rel_xi_buf = p_trans_xi_buf @ p_mat_rel_xi
            p_rel_xi_buf = self._scale_and_bias(p_rel_xi_buf, "p_rel_xi_buf", depth)
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
            x_buf = self._activate(x_buf, "x_buf", depth)

            if self.preserve_distribution:
                x_buf = self._standardize(x_buf)
                x_buf = (
                    (x_buf * x_origin_std) + x_origin_mean
                    if self.preserve_distribution
                    else x_buf
                )
            elif self.standardize_output:
                x_buf = self._standardize(x_buf)

            # Scale down in accordance to overall depth.
            x_buf = x_buf * (1.0 / self.depth)

            # Accumulate values.
            projection = projection.add(x_buf)

        return projection
