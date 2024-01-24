import torch
import torch.nn as nn

from self_projection import SelfProjection, SelfProjectionDev


class SimpleAutoencoderSPSingle(nn.Module):
    def __init__(
        self,
        input_size: int,
        self_projection_depth: int = 8,
        dev: bool = False,
        sp_params: dict = {},
    ):
        super(SimpleAutoencoderSPSingle, self).__init__()

        self.sp_class = SelfProjectionDev if dev else SelfProjection

        self.self_projection = self.sp_class(
            size_input=[input_size, input_size],
            size_projection=input_size,
            depth=self_projection_depth,
            **sp_params,
        )

        pass

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        return self.self_projection(x), torch.zeros([x.shape[0], 16, 16]).to(x.device, x.dtype)
