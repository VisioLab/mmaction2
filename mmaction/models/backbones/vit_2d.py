from mmaction.registry import MODELS
from mmengine.model import BaseModule
from typing import Optional,Union,List,Dict
from timm.models.vision_transformer import vit_base_patch16_224
import torch

@MODELS.register_module()
class ViT2D(BaseModule):
    def __init__(
            self,
            pretrained:Optional[str] = None,
            freeze: bool = False,
            init_cfg: Optional[Union[Dict, List[Dict]]] = [
                     dict(
                         type='TruncNormal', layer='Linear', std=0.02,
                         bias=0.),
                     dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
                 ],

            
    ):
        self.pretrained = pretrained
        self.freeze = freeze
        super().__init__(init_cfg=init_cfg)

        self.backbone = vit_base_patch16_224(pretrained=bool(self.pretrained),num_classes=0)
        

    def init_weights(self):
        if self.pretrained :
            self.init_cfg = dict(type='Pretrained', checkpoint=self.pretrained)

        super().init_weights()

        if self.freeze:
            frozen_layers = []
            for name,weights in self.named_parameters():
                weights.requires_grad = False
                frozen_layers.append(name)
    

    def forward(self,x:torch.Tensor):
        B, C, F, H, W = x.shape  # Batch x Channel x Frame x Height x Width
        x = x.reshape(B * F, C, H, W)
        x = self.backbone(x)
        x = x.reshape(B, F, -1)
        return x