import torch
import torch.nn as nn

from self_projection import SelfProjection, SelfProjectionDev


class SimpleAutoencoderSPSP(nn.Module):
    def __init__(
        self,
        input_size: int,
        network_depth: 3,
        self_projection_depth: int = 8,
        dev: bool = False,
    ):
        super(SimpleAutoencoderSPSP, self).__init__()

        self.sp_class = SelfProjectionDev if dev else SelfProjection

        dim = input_size
        dims = [dim]
        for _ in range(network_depth):
            dim = dim // 2
            dims.append(dim)

        encoder = nn.ModuleList()
        for depth_id in range(network_depth):
            encoder.append(
                self.sp_class(
                    size_input=[dims[depth_id]] * 2,
                    size_projection=dims[depth_id + 1],
                    depth=self_projection_depth,
                ),
            )
            encoder.append(nn.LayerNorm([dims[depth_id + 1]] * 2))
            print(f"encoder[{depth_id}]: {dims[depth_id]} -> {dims[depth_id + 1]}")
        self.add_module("encoder", nn.Sequential(*encoder))

        decoder = nn.ModuleList()
        for depth_id in range(network_depth - 1, -1, -1):
            decoder.append(
                self.sp_class(
                    size_input=[dims[depth_id + 1]] * 2,
                    size_projection=dims[depth_id],
                    depth=self_projection_depth,
                ),
            )
            if depth_id > 0:
                decoder.append(nn.LayerNorm([dims[depth_id]] * 2))
            print(f"decoder[{depth_id}]: {dims[depth_id + 1]} -> {dims[depth_id]}")
        self.add_module("decoder", nn.Sequential(*decoder))

        pass

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.encoder(x)
        return x

    def decode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.decoder(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = self.encode(x)
        latents = x.clone()
        x = self.decode(x)
        return x, latents
