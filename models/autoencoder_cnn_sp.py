import math
import torch
import torch.nn as nn

from self_projection import SelfProjection, SelfProjectionDev


class Checkerboard(nn.Module):
    def __init__(
        self,
        reverse: bool = False,
        channels: int = None,
    ) -> None:
        super(Checkerboard, self).__init__()

        if reverse:
            assert (
                channels is not None
            ), "The 'channels' argument must be specified when 'reverse' is True."
            assert not (
                channels & (channels - 1)
            ), "The 'channels' argument must be a power of 2."

        self.reverse = reverse
        self.channels = channels

        pass

    def _to_checkerboard(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[1] & (x.shape[1] - 1):
            raise ValueError("The 'blocks' dimension must be a power of 2.")

        num_blocks_side = int(math.sqrt(x.shape[1]))

        x_reshaped = x.view(
            [
                x.shape[0],
                num_blocks_side,
                num_blocks_side,
                x.shape[2],
                x.shape[3],
            ]
        )

        output_shape = [
            x.shape[0],
            x.shape[2] * num_blocks_side,
            x.shape[3] * num_blocks_side,
        ]
        output = torch.zeros(
            output_shape,
            dtype=x.dtype,
            device=x.device,
        )

        for row in range(num_blocks_side):
            for col in range(num_blocks_side):
                output[
                    :,
                    row * x.shape[2] : (row + 1) * x.shape[2],
                    col * x.shape[3] : (col + 1) * x.shape[3],
                ] = x_reshaped[:, row, col]

        return output

    def _from_checkerboard(
        self,
        x: torch.Tensor,
        channels: int,
    ) -> torch.Tensor:
        if len(x.shape) != 3:
            raise ValueError("Input tensor must have shape [batch, H, W]")

        batch, height, width = x.shape
        num_blocks_side = int(math.sqrt(channels))

        if height % num_blocks_side != 0 or width % num_blocks_side != 0:
            raise ValueError(
                "The height and width of the input tensor must be multiples of sqrt(channels)."
            )

        block_H, block_W = height // num_blocks_side, width // num_blocks_side

        output_shape = [
            batch,
            channels,
            block_H,
            block_W,
        ]

        output = torch.zeros(
            output_shape,
            dtype=x.dtype,
            device=x.device,
        )

        for i in range(num_blocks_side):
            for j in range(num_blocks_side):
                output[:, i * num_blocks_side + j, :, :] = x[
                    :, i * block_H : (i + 1) * block_H, j * block_W : (j + 1) * block_W
                ]

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reverse:
            return self._from_checkerboard(x, self.channels)
        else:
            return self._to_checkerboard(x)


class AutoencoderCNNSP(nn.Module):
    def __init__(
        self,
        input_size: int,
        network_depth: int,
        scale_factor: int = 2,
        dev: bool = False,
        sp_params: dict = {},
    ):
        super(AutoencoderCNNSP, self).__init__()

        scale_factor = int(scale_factor)

        assert scale_factor > 0, "The 'scale_factor' argument must be positive."
        assert (
            scale_factor and (scale_factor & (scale_factor - 1)) == 0
        ), "The 'scale_factor' argument must be a power of 2."

        self.sp_class = SelfProjectionDev if dev else SelfProjection
        self.scale_factor = scale_factor

        encoder_base = input_size
        encoder_dims = [encoder_base]
        for _ in range(network_depth):
            encoder_base = encoder_base // self.scale_factor
            encoder_dims.append(encoder_base)

        encoder: nn.ModuleList = [
            self._create_downsampling_block(
                size_input=encoder_dims[depth_id],
                size_output=encoder_dims[depth_id + 1],
                sp_params=sp_params,
            )
            for depth_id in range(network_depth)
        ]

        decoder_base = encoder_dims[-1]
        decoder_dims = [decoder_base]
        for _ in range(network_depth):
            decoder_base = decoder_base * self.scale_factor
            decoder_dims.append(decoder_base)

        decoder: nn.ModuleList = [
            self._create_upsampling_block(
                size_input=decoder_dims[depth_id],
                size_output=decoder_dims[depth_id + 1],
                sp_params=sp_params,
            )
            for depth_id in range(network_depth)
        ]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

        pass

    def _create_downsampling_block(
        self,
        size_input: int,
        size_output: int,
        sp_params: dict = {},
    ) -> nn.Sequential:
        block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            Checkerboard(),
            self.sp_class(
                size_input=[size_input, size_input],
                size_projection=size_output,
                **sp_params,
            ),
        )
        return block

    def _create_upsampling_block(
        self,
        size_input: int,
        size_output: int,
        sp_params: dict = {},
    ) -> nn.Sequential:
        block = nn.Sequential(
            self.sp_class(
                size_input=[size_input, size_input],
                size_projection=size_output,
                **sp_params,
            ),
            Checkerboard(
                reverse=True,
                channels=16,
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=False,
            ),
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        return block

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.view([x.shape[0], 1, *x.shape[1:]]) if len(x.shape) == 3 else x
        x = self.encoder(x)
        return x

    def decode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.decoder(x)
        x = x.view([x.shape[0], *x.shape[2:]]) if len(x.shape) == 4 else x
        return x

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        x = self.encode(x)
        latents = x.clone()
        x = self.decode(x)
        return x, latents
