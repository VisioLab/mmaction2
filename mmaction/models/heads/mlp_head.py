import torch
from torch import Tensor, nn
import torch.nn.functional as F
from mmengine.model.weight_init import normal_init

from mmaction.registry import MODELS
from .base import BaseModule, BaseHead


@MODELS.register_module()
class MLP(BaseHead):
    def __init__(
            self,
            in_channels: int,
            dropout_rate: float,
            num_classes:int,
            **kwargs,
    ):
        super().__init__(num_classes=num_classes,in_channels=in_channels,**kwargs)
        self.dropout_rate = dropout_rate

        self.features = nn.Sequential(
            nn.LayerNorm(self.in_channels),
            nn.Linear(self.in_channels,self.in_channels),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.in_channels,self.num_classes),
        )

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.features, std=0.02)


    def forward(self,x:Tensor,**kwargs):
        x = self.features(x[:,0])
        return x
    
    # def loss(self,x,data_samples):
    #     labels = torch.stack([x.gt_label for x in data_samples]).to('cuda')
        
    #     labels = labels.squeeze()
    #     if labels.shape == torch.Size([]):
    #         labels = labels.unsqueeze(0)
    #     loss_cls = self.loss_fn(x, labels)
    #     return dict(loss_cls=loss_cls)
    
    # def predict(self,x,data_samples):
    #     cls_score = F.softmax(x,dim=1)
    #     for ds,sc in zip(data_samples,cls_score):
    #         ds.pred_score = sc
    #     return data_samples
    

# @MODELS.register_module()
# class MLP(BaseModule):
#     def __init__(
#             self,
#             in_channels: int,
#             hidden_channels: int,
#             dropout_rate: float,
#             num_classes:int,
#             init_cfg=None
#     ):
#         super(MLP,self).__init__(init_cfg=init_cfg)
#         self.in_channels = in_channels
#         self.hidden_channels = hidden_channels
#         self.num_classes = num_classes
#         self.dropout_rate = dropout_rate
#         self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

#         self.features = nn.Sequential(
#             nn.LayerNorm(self.in_channels),
#             nn.Linear(self.in_channels,self.hidden_channels),
#             nn.GELU(),
#             nn.Dropout(self.dropout_rate),
#             nn.Linear(self.hidden_channels,self.num_classes),
#         )

#     def forward(self,x:Tensor):
#         x = self.features(x[:,0])
#         return x
    
#     def loss(self,x,data_samples):
#         labels = torch.stack([x.gt_label for x in data_samples]).to('cuda')
        
#         labels = labels.squeeze()
#         if labels.shape == torch.Size([]):
#             labels = labels.unsqueeze(0)
#         loss_cls = self.loss_fn(x, labels)
#         return dict(loss_cls=loss_cls)
    
#     def predict(self,x,data_samples):
#         cls_score = F.softmax(x,dim=1)
#         for ds,sc in zip(data_samples,cls_score):
#             ds.pred_score = sc
#         return data_samples