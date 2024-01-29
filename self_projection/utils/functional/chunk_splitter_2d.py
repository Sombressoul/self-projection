import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum


class ChunkSplitter2DMode(Enum):
    CROP = "crop"
    INTERPOLATE = "interpolate"
    PAD_RIGHT_BOTTOM = "pad_right_bottom"
    PAD_ALL = "pad_all"


class ChunkSplitter2D(nn.Module):
    def __init__(
        self,
        chunk_shape: list[int] = [8, 8],
        mode: ChunkSplitter2DMode = ChunkSplitter2DMode.PAD_ALL,
    ) -> None:
        """
        Args:
            chunk_shape: list[int] - two-dimensional chunk shape. Default: [8, 8].
            mode: ChunkSplitter2DPadMode - chunk splitting mode. Default: ChunkSplitter2DPadMode.PAD_ALL.
        """
        super().__init__()

        if len(chunk_shape) != 2:
            raise ValueError(
                "".join(
                    [
                        "Invalid chunk shape: chunk_size must be 2-dimensional.\n",
                        f"Got: {str(chunk_shape)}",
                    ]
                )
            )

        self.chunk_shape = chunk_shape
        self.mode = mode

        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        chunk_H, chunk_W = self.chunk_shape

        if self.mode in [
            ChunkSplitter2DMode.PAD_RIGHT_BOTTOM,
            ChunkSplitter2DMode.PAD_ALL,
        ]:
            pad_W_l = 0
            pad_W_r = (chunk_W - W % chunk_W) % chunk_W
            pad_H_t = 0
            pad_H_b = (chunk_H - H % chunk_H) % chunk_H

            if self.mode == ChunkSplitter2DMode.PAD_ALL:
                pad_W_l = pad_W_r // 2
                pad_W_r = pad_W_r - pad_W_l
                pad_H_t = pad_H_b // 2
                pad_H_b = pad_H_b - pad_H_t

            x = F.pad(
                x,
                [pad_W_l, pad_W_r, pad_H_t, pad_H_b],
                mode="constant",
                value=0,
            )
        elif self.mode == ChunkSplitter2DMode.INTERPOLATE:
            new_H = ((H - 1) // chunk_H + 1) * chunk_H
            new_W = ((W - 1) // chunk_W + 1) * chunk_W
            x = F.interpolate(
                x,
                size=[new_H, new_W],
                mode="bilinear",
                align_corners=False,
            )
        elif self.mode == ChunkSplitter2DMode.CROP:
            # Default unfold behavior.
            pass
        else:
            raise ValueError(
                "".join(
                    [
                        "Invalid mode: mode must be one of ChunkSplitter2DMode.\n",
                        f"Got: {str(self.mode)}",
                    ]
                )
            )

        unfolded_h = x.unfold(2, chunk_H, chunk_H)
        unfolded_w = unfolded_h.unfold(3, chunk_W, chunk_W)

        B, C, H_chunks, W_chunks, H, W = unfolded_w.shape

        chunks = unfolded_w
        chunks = chunks.permute([0, 2, 3, 1, 4, 5])
        chunks = chunks.reshape([B, H_chunks, W_chunks, C, H, W])

        return chunks
