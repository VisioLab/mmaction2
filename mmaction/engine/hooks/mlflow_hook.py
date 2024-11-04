from mmengine.hooks import Hook
from mmengine.runner import  Runner

from mmaction.registry import HOOKS

from pathlib import Path
import mlflow
from typing import Any
import os
import glob

import importlib.util as imu

NOT_LOG = [
    'default_hooks','custom_hooks','default_scope','env_cfg','file_client_args',
    'launcher','log_level','train_cfg','train_dataset_cfg','train_dataloader',
    'val_cfg','val_dataset_cfg','val_dataloader','val_evaluator',
    'vis_backends','visualizer','log_processor','tags'
    ]

def flatten_dict(d:dict, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list) and all(isinstance(i, dict) for i in v):
            for i, item in enumerate(v):
                list_key = f"{new_key}_{i}"
                items.extend(flatten_dict(item, list_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@HOOKS.register_module()
class MLflowHook(Hook):

    def __init__(self,
                 tags:dict[str,str],
                 ) -> None:
        self.tags = tags

    def before_run(self, runner:Runner) -> None:
        
        self.tags.update(experiment_id=runner.experiment_name)
        mlflow.start_run(tags=self.tags)

    def before_train(self, runner:Runner) -> None:
        params:dict[str,Any] = {}

        params.update(dict(train_batch_size=runner.train_dataloader.batch_size))
        params.update(dict(val_batch_size=runner.val_dataloader.batch_size))

        num_samples_train = len(runner.train_dataloader.dataset)
        params.update(dict(num_samples_train=num_samples_train))

        num_samples_val = len(runner.val_dataloader.dataset)
        params.update(dict(num_samples_val=num_samples_val))

        cfg = Path(runner.log_dir + '/vis_data' + '/config.py')
        mlflow.log_artifact(str(cfg))

        spec = imu.spec_from_file_location("config_module", cfg)
        config = imu.module_from_spec(spec)
        spec.loader.exec_module(config)


        for key, value in config.__dict__.items():
            if (key.startswith("__") and not callable(value) ) or (key in NOT_LOG):
                continue

            if isinstance(value, dict):
                params.update(flatten_dict(value,key))
            elif isinstance(value, list) and all(isinstance(i, dict) for i in value):
                for i, item in enumerate(value):
                    params.update(flatten_dict(item, f"{key}_{i}") )
            else:
                params[key] = value

        mlflow.log_params(params)

        dataset_path = Path(runner.train_dataloader.dataset.ann_file).parent
        mlflow.log_artifacts(str(dataset_path),artifact_path='dataset')
        

    def after_train_iter(self, runner:Runner, batch_idx: int, data_batch: dict, outputs: dict) -> None:
        iteration_per_epoch = len(runner.train_dataloader)


        if (batch_idx+1)%iteration_per_epoch == 0:
            mlflow.log_metrics(outputs,step=runner.epoch)
        

    def after_val_epoch(self, runner:Runner, metrics: dict) -> None:
        mlflow.log_metrics(metrics=metrics,step=runner.epoch)

    
    def after_test_epoch(self, runner:Runner, metrics: dict) -> None:
        mlflow.log_metrics(metrics=metrics,step=runner.epoch)


    def after_run(self, runner:Runner) -> None:
        if not runner.train_dataloader:
            mlflow.end_run()
            return
        work_dir = runner.work_dir
        patt = os.path.join(work_dir, 'best_*.pth')
        best_weights = glob.glob(patt)[0]
        
        mlflow.log_artifact(best_weights)
        last_epoch = os.path.join(work_dir,f'epoch_{runner.max_epochs}.pth')
        mlflow.log_artifact(last_epoch)

        mlflow.end_run()