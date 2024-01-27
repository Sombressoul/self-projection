import torch
import torch.nn as nn

from modules.self_projection import SelfProjection


class SimpleAutoencoderSPSingle(nn.Module):
    def __init__(
        self,
        input_size: int,
        sp_params: dict = {},
    ):
        super(SimpleAutoencoderSPSingle, self).__init__()

        self.self_projection = SelfProjection(
            size_input=[input_size, input_size],
            size_projection=input_size,
            **sp_params,
        )

        pass

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        return self.self_projection(x), torch.zeros([x.shape[0], 16, 16]).to(
            x.device, x.dtype
        )
