import math
import torch
import torch.nn as nn


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
            assert (
                math.sqrt(channels) % 1 == 0
            ), "The 'channels' argument must be equal to a power of an integer."

        self.reverse = reverse
        self.channels = channels

        pass

    def _to_checkerboard(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if math.sqrt(x.shape[1]) % 1 != 0:
            raise ValueError("The number of channels must be a power of an integer.")

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
