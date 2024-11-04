# Copyright (c) OpenMMLab. All rights reserved.
import math
import os.path as osp
import os
from typing import Optional, Sequence, Tuple

from mmengine.hooks import Hook
from mmengine.runner import  Runner

from mmaction.registry import HOOKS
from mmaction.structures import ActionDataSample
from collections import defaultdict

from torchvision.utils import make_grid, save_image
from PIL import ImageDraw, ImageFont
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from torch import Tensor
from tqdm import tqdm

from pathlib import Path
import json

import mlflow



@HOOKS.register_module()
class MisclassificationHook(Hook):
    priority = 'LOW'
    def __init__(self,
                 class_map:str,
                 out_dir: Optional[str] = None,
                 interval: int = 1,
                 **kwargs):

        assert class_map.endswith('.json')
        
        self.interval = interval
        self.out_dir = out_dir

        with open(class_map,'r') as f:
            self.class_map = json.load(f)

    def _create_misclassification_dict(self,
                      batch_idx: int,
                      data_batch: dict,
                      data_samples: Sequence[ActionDataSample],
                      ) -> None:
        
        batch_size = len(data_samples)
        videos = data_batch['inputs']
        start_idx = batch_size * batch_idx
        end_idx = start_idx + batch_size

        # The first index divisible by the interval, after the start index
        first_sample_id = math.ceil(start_idx / self.interval) * self.interval

        for sample_id in range(first_sample_id, end_idx, self.interval):
            data_sample = data_samples[sample_id - start_idx]
            if data_sample.gt_label.item() == data_sample.pred_label.item():
                continue

            video = videos[sample_id - start_idx]
            if video.shape[0]>1: ## inference with multi view
                video = video[0].unsqueeze(0) ## extract only one view

            video = video.reshape((-1, ) + video.shape[2:])
            video = video.permute(1, 0, 2, 3)

            self.misclassification[data_sample.gt_label.item()].append([video,data_sample])


    def before_test_epoch(self, runner:Runner) -> None:
        self.misclassification: defaultdict[int, list[Tuple[Tensor, ActionDataSample]]] = defaultdict(list)

        mlflow.log_param('checkpoint',runner._load_from)
   
    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[ActionDataSample]) -> None:
        """Visualize every ``self.interval`` samples during test.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): Outputs from model.
        """
        self._create_misclassification_dict(batch_idx, data_batch, outputs)

    def after_test(self, runner:Runner) -> None:
        self.out_dir = Path(self.out_dir) if self.out_dir else Path(runner.log_dir) / 'misclassifications'
        self._save_misclassifications()
        mlflow.log_artifact(self.out_dir)
        
        
        

    def _save_misclassifications(self):
        for label,values in tqdm(self.misclassification.items()):

            save_dir = self.out_dir / self.class_map[str(label)]
            for samples in values:
                frames,data_sample = samples
                grid = make_grid(frames,nrow=8)
                pil_image = to_pil_image(grid)
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.load_default()

                prediction = self.class_map[str(data_sample.pred_label.item())]
                score = data_sample.pred_score.max().item()

                draw.text((5,0),text=f"{prediction}",font=font,fill=(255,0,0))
                draw.text((5,10),text=f"{score:.2f}",font=font,fill=(255,0,0))
                img_tensor = pil_to_tensor(pil_image).float().div(255)


                filename = data_sample.get('filename').split('/')[-1]
                filename = filename.rsplit('.',1)[0] + '.jpg'

                save_dir.mkdir(parents=True,exist_ok=True)
                save_image(img_tensor,fp=save_dir/filename)
            

