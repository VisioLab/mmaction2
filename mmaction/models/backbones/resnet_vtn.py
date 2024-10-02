from torch import Tensor
from mmaction.registry import MODELS
from . import ResNet
import torch.nn as nn


@MODELS.register_module()
class RESNET_BACKBONE(ResNet):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x: Tensor):
        x = x.permute(0, 2, 1, 3, 4)
        B, C, F, H, W = x.shape  # Batch x Channel x Frame x Height x Width
        x = x.reshape(B * F, C, H, W)
        out = super().forward(x)
        if isinstance(out,tuple):
            out = out[-1]
        out = self.global_avg_pool(out).flatten(1)
        out = out.reshape(B, F, -1)
        return out
