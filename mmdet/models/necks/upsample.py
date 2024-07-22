# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig


@MODELS.register_module()
class UpSampler(BaseModule):
    """Up sample input image with a 2d conv
    Args:
    in_channels,
    out_channels,
    kernel_size
    """

    def __init__(
        self,
        init_cfg=None
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.downsize = nn.Conv2d(1024, 256, 3, 2)
        self.channel_map = nn.Conv2d(1024, 256, 1)
        self.first = nn.ConvTranspose2d(1024, 512, 2) # [bs, 25, 25, 1024] -> [bs, 50, 50, 512]
        self.channel_map2 = nn.Conv2d(512, 256, 1)
        self.second = nn.ConvTranspose2d(512, 256, 2) # [bs, 50, 50, 512] -> [bs, 100, 100, 256]
        self.third = nn.ConvTranspose2d(256, 256, 2) # [bs, 100, 100, 256] -> [bs, 200, 200, 128]

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function."""
        # return biggest feature map to smallest feature map
        outputs = []
        outputs.append(self.downsize(inputs[0]))
        outputs.append(self.channel_map(inputs[0]))
        first = self.first(inputs[0])
        outputs.append(self.channel_map2(first))
        outputs.append(self.second(first))
        outputs.append(self.third(outputs[-1]))
        return tuple(outputs[::-1])