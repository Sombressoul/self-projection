import math
import torch
import torch.nn as nn

from self_projection.self_projection import SelfProjection
from self_projection.utils import Checkerboard


class AutoencoderCNNSP(nn.Module):
    def __init__(
        self,
        input_size: int,
        network_depth: int,
        scale_factor: int = 2,
        channels_base: int = 16,
        compressor_depth: int = 1,
        extractor_depth: int = 1,
        use_compressor: bool = True,
        use_extractor: bool = True,
        baseline: bool = False,
        dropout_rate: float = 0.1,
        sp_params: dict = {},
    ):
        super(AutoencoderCNNSP, self).__init__()

        scale_factor = int(scale_factor)

        if network_depth > 1:
            raise NotImplementedError("The 'network_depth' argument must be 1 for now.")

        assert scale_factor > 0, "The 'scale_factor' argument must be positive."
        assert (
            scale_factor and (scale_factor & (scale_factor - 1)) == 0
        ), "The 'scale_factor' argument must be a power of 2."
        assert compressor_depth > 0, "The 'compressor_depth' argument must be positive."
        assert extractor_depth > 0, "The 'extractor_depth' argument must be positive."

        if baseline:
            use_compressor = False
            use_extractor = False

        # External parameters.
        self.scale_factor = scale_factor
        self.channels_base = channels_base
        self.compressor_depth = compressor_depth
        self.extractor_depth = extractor_depth
        self.use_compressor = use_compressor
        self.use_extractor = use_extractor
        self.baseline = baseline
        self.dropout_rate = dropout_rate

        # Internal parameters.
        self.baseline_bottleneck_c = (
            math.ceil(math.sqrt(self.scale_factor * self.channels_base * 4)) ** 2
        )

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

        self.checkerboard = Checkerboard()

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
                out_channels=self.channels_base,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.SELU(),
            nn.BatchNorm2d(
                num_features=self.channels_base,
            ),
            nn.Conv2d(
                in_channels=self.channels_base,
                out_channels=self.channels_base * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.SELU(),
            nn.BatchNorm2d(
                num_features=self.channels_base * 2,
            ),
            nn.Conv2d(
                in_channels=self.channels_base * 2,
                out_channels=self.channels_base * 4,
                kernel_size=self.scale_factor,
                stride=self.scale_factor,
                padding=0,
                bias=True,
            ),
            nn.SELU(),
            nn.BatchNorm2d(
                num_features=self.channels_base * 4,
            ),
            nn.Dropout(p=self.dropout_rate),
            (
                nn.Conv2d(
                    in_channels=self.channels_base * 4,
                    out_channels=self.scale_factor**2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
                if not self.baseline
                else nn.Identity()
            ),
            (nn.SELU() if not self.baseline else nn.Identity()),
            (
                nn.BatchNorm2d(
                    num_features=self.scale_factor**2,
                )
                if not self.baseline
                else nn.Identity()
            ),
            (Checkerboard() if not self.baseline else nn.Identity()),
            (
                SelfProjection(  # Extractor
                    size_input=[size_input, size_input],
                    size_projection=size_input,
                    **(
                        sp_params
                        | dict(
                            depth=self.extractor_depth,
                            preserve_distribution=False,
                            standardize_output=True,
                            scale_and_bias=False,
                        )
                    ),
                )
                if self.use_extractor
                else nn.Identity()
            ),
            (
                SelfProjection(  # Compressor
                    size_input=[size_input, size_input],
                    size_projection=size_output,
                    **(
                        sp_params
                        | dict(
                            depth=self.compressor_depth,
                        )
                    ),
                )
                if self.use_compressor
                else nn.Identity()
            ),
            (  # Baseline
                nn.Conv2d(
                    in_channels=self.channels_base * 4,
                    out_channels=self.baseline_bottleneck_c,
                    kernel_size=self.scale_factor,
                    stride=self.scale_factor,
                    padding=0,
                    bias=True,
                )
                if self.baseline
                else nn.Identity()
            ),
            (nn.SELU() if self.baseline else nn.Identity()),  # Baseline
        )
        return block

    def _create_upsampling_block(
        self,
        size_input: int,
        size_output: int,
        sp_params: dict = {},
    ) -> nn.Sequential:
        block = nn.Sequential(
            (  # Baseline
                nn.ConvTranspose2d(
                    in_channels=self.baseline_bottleneck_c,
                    out_channels=self.channels_base * 4,
                    kernel_size=self.scale_factor,
                    stride=self.scale_factor,
                    padding=0,
                    bias=True,
                )
                if self.baseline
                else nn.Identity()
            ),
            (nn.SELU() if self.baseline else nn.Identity()),
            (
                SelfProjection(  # Decompressor
                    size_input=[size_input, size_input],
                    size_projection=size_output,
                    **(
                        sp_params
                        | dict(
                            depth=self.compressor_depth,
                        )
                    ),
                )
                if self.use_compressor
                else nn.Identity()
            ),
            (
                SelfProjection(  # Extractor
                    size_input=[size_output, size_output],
                    size_projection=size_output,
                    **(
                        sp_params
                        | dict(
                            depth=self.extractor_depth,
                            preserve_distribution=False,
                            standardize_output=True,
                            scale_and_bias=False,
                        )
                    ),
                )
                if self.use_extractor
                else nn.Identity()
            ),
            (
                Checkerboard(
                    reverse=True,
                    channels=self.scale_factor**2,
                )
                if not self.baseline
                else nn.Identity()
            ),
            (
                nn.BatchNorm2d(
                    num_features=self.scale_factor**2,
                )
                if not self.baseline
                else nn.Identity()
            ),
            (
                nn.Conv2d(
                    in_channels=self.scale_factor**2,
                    out_channels=self.channels_base * 4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
                if not self.baseline
                else nn.Identity()
            ),
            (nn.SELU() if not self.baseline else nn.Identity()),
            nn.BatchNorm2d(
                num_features=self.channels_base * 4,
            ),
            nn.Dropout(p=self.dropout_rate),
            nn.ConvTranspose2d(
                in_channels=self.channels_base * 4,
                out_channels=self.channels_base * 2,
                kernel_size=self.scale_factor,
                stride=self.scale_factor,
                padding=0,
                bias=True,
            ),
            nn.SELU(),
            nn.BatchNorm2d(
                num_features=self.channels_base * 2,
            ),
            nn.ConvTranspose2d(
                in_channels=self.channels_base * 2,
                out_channels=self.channels_base,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.SELU(),
            nn.BatchNorm2d(
                num_features=self.channels_base,
            ),
            nn.ConvTranspose2d(
                in_channels=self.channels_base,
                out_channels=math.ceil(self.channels_base / 2),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.Conv2d(  # Output refiner.
                in_channels=math.ceil(self.channels_base / 2),
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
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
        latents = self.checkerboard(x) if self.baseline else x
        x = self.decode(x)
        return x, latents
