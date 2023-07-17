from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from module import DownSample, ResBlock, Swish, TimeEmbedding, UpSample
from torch.nn import init


class UNet(nn.Module):
    def __init__(
        self,
        T: int = 1000,
        image_resolution: int = 64,
        ch: int = 128,
        ch_mult: List[int] = [1, 2, 2, 2],
        attn: List[int] = [1],
        num_res_blocks: int = 4,
        dropout: float = 0.1,
        use_cfg: bool = False,
        cfg_dropout: float = 0.1,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.image_resolution = image_resolution
    
        # TODO: Implement an architecture according to the provided architecture diagram.
        # You can use the modules in `module.py`.

    def forward(self, x, timestep, class_label=None):
        """
        Input:
            x (`torch.Tensor [B,C,H,W]`)
            timestep (`torch.Tensor [B]`)
            class_label (`torch.Tensor [B]`, optional)
        Output:
            out (`torch.Tensor [B,C,H,W]`): noise prediction.
        """
        assert (
            x.shape[-1] == x.shape[-2] == self.image_resolution
        ), f"The resolution of x ({x.shape[-2]}, {x.shape[-1]}) does not match with the image resolution ({self.image_resolution})."

        # TODO: Implement noise prediction network's forward function.
        
        out = torch.randn_like(x)
        return out
