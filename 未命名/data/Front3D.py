import math
import os
import json
import re
import cv2
import numpy as np
from dataclasses import dataclass, field
from PIL import Image
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from craftsman import register
from craftsman.utils.typing import *
from craftsman.utils.config import parse_structured
from .front3d_recon_dataset import Front3DReconDataset, Front3DReconDataModuleConfig

# @dataclass
# class Front3DReconDataModuleConfig(BaseDataModuleConfig):
#     pass

# class Front3DReconDataset(BaseDataset):
#     pass


@register("Front3D-datamodule")
class Front3DReconDataModule(pl.LightningDataModule):
    cfg: Front3DReconDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(Front3DReconDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit", "train"]:
            self.train_dataset = Front3DReconDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = Front3DReconDataset(self.cfg, "test")
            self.train_dataset = Front3DReconDataset(self.cfg, "train")
        if stage in [None, "test", "predict"]:
            self.test_dataset = Front3DReconDataset(self.cfg, "test")

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
    

    
