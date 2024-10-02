from typing import Literal

from mmaction.registry import MODELS
from mmengine.model import BaseModel

MODE = Literal['backbone','neck','cls_head']


@MODELS.register_module()
class VTN(BaseModel):
    def __init__(self,backbone,neck,cls_head,data_preprocessor):
        super().__init__(data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.cls_head = MODELS.build(cls_head)

    def extract_feat(self,input):
            x, position_id = input
            def _extract_backbone_feature(input):
                 return self.backbone(input)
            
            x = _extract_backbone_feature(x)
            x = (x,position_id)
            x = self.neck(x)
            return self.cls_head(x)

        

    def forward(self,inputs,data_samples,mode='loss',**kwargs):
        inputs = [i.to('cuda') for i in inputs]
    
        if mode == 'tensor':
            x = self.extract_feat(inputs)
            return x

        elif mode=='loss':
            x = self.loss(inputs,data_samples)
            return x
        
        elif mode=='predict':
             x = self.predict(inputs,data_samples)
             return x

    def loss(self,inputs,data_samples):
         feat = self.extract_feat(inputs)
         loss =  self.cls_head.loss(feat,data_samples)
         return loss
    
    def predict(self,inputs,data_samples):
         feat = self.extract_feat(inputs)
         pred = self.cls_head.predict(feat,data_samples)
         return pred
        
    


