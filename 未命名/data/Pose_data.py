import math
import os
import json
import re
import cv2
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from craftsman import register
from craftsman.utils.typing import *
from craftsman.utils.config import parse_structured

from streaming import StreamingDataLoader
from .base import BaseDataModuleConfig, BaseDataset
from .base_s3 import BaseS3DataModuleConfig, BaseS3Dataset
from .pose_dataset import PoseDataset, PoseDataModuleConfig

# @dataclass
# class PoseDataModuleConfig(BaseDataModuleConfig):
#     pass

# class PoseDataset(BaseDataset):
#     pass


@register("Pose-datamodule")
class PoseDataModule(pl.LightningDataModule):
    cfg: PoseDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(PoseDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = PoseDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = PoseDataset(self.cfg, "test")
            self.train_dataset = PoseDataset(self.cfg, "train")
        if stage in [None, "test", "predict"]:
            self.test_dataset = PoseDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None, num_workers=0) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
            num_workers=self.cfg.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, 
            batch_size=1,
            collate_fn=self.val_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, 
            batch_size=1,
            collate_fn=self.test_dataset.collate)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)
    
