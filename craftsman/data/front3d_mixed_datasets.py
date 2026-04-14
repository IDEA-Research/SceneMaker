# Dataloader of InstPIFu.
# author: LiuHaolin
# date: Aug, 2022
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch.utils.data
from torchvision import transforms
import pickle
from PIL import Image
import numpy as np
import json
import random
from craftsman.data.net_utils.bins import *
import glob
from tqdm import tqdm
from utils.space_transform import get_rot_from_pitch, get_pitch_from_R, q2rot, get_yaw_from_R, get_rot_from_yaw
from utils.transforms import *
import trimesh
import fpsample
import rembg
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from craftsman import register
from craftsman.utils.typing import *
from craftsman.utils.config import parse_structured

import math
import re
import cv2
from dataclasses import dataclass, field

import random
import imageio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from craftsman.utils.typing import *
from .base_s3 import BaseS3DataModuleConfig
from .front3d_midi_dataset import Front3DMIDIDataModuleConfig, Front3DMIDIDataset
from .front3d_recon_dataset import Front3DReconDataModuleConfig, Front3DReconDataset

import pdb

class Front3DMixedDataset(Dataset):
    def __init__(self, config, mode):
        super(Front3DMixedDataset, self).__init__()
        self.mode = mode
        assert mode in ["train", "val", "test"]
        
        # split config
        instpifu_cfg_dict = dict(config.get("instpifu_cfg", {}))
        midi_cfg_dict = dict(config.get("midi_cfg", {}))
        public_cfg = {k: v for k, v in config.items() if k not in ["instpifu_cfg", "midi_cfg"]}
        
        # construct config
        instpifu_cfg_dict = {**instpifu_cfg_dict, **public_cfg}
        midi_cfg_dict = {**midi_cfg_dict, **public_cfg}
        self.instpifu_cfg = Front3DReconDataModuleConfig(**instpifu_cfg_dict)
        self.midi_cfg = Front3DMIDIDataModuleConfig(**midi_cfg_dict)
        
        # load datasets
        self.instpifu_datasets = Front3DReconDataset(self.instpifu_cfg, mode)
        self.midi_datasets = Front3DMIDIDataset(self.midi_cfg, mode)
        

    def __len__(self):
        return len(self.instpifu_datasets) + len(self.midi_datasets)
    
    def __getitem__(self, index):
        if index < len(self.instpifu_datasets):
            return self.instpifu_datasets[index]
        else:
            return self.midi_datasets[index - len(self.instpifu_datasets)]
    
    def collate(self, batch):
        from torch.utils.data._utils.collate import default_collate_fn_map
        ret = {}
        for key, value in batch[0].items():
            if isinstance(value, str):
                ret[key] = [b[key] for b in batch]
            elif isinstance(value, list):
                ret[key] = [b[key] for b in batch]
            elif isinstance(value, dict):
                ret[key] = [b[key] for b in batch]
            elif isinstance(value, bool):
                ret[key] = [b[key] for b in batch]
            elif isinstance(value, torch.Tensor):
                ret[key] = torch.stack([b[key] for b in batch])
            elif isinstance(value, np.ndarray):
                ret[key] = torch.stack([torch.from_numpy(b[key]) for b in batch])
            else:
                ret[key] = default_collate_fn_map[type(batch[0][key])](batch)
        return ret


# data loader  
@register("Front3D-mixed-datamodule")
class Front3DMixedDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = Front3DMixedDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = Front3DMixedDataset(self.cfg, "test")
            self.train_dataset = Front3DMixedDataset(self.cfg, "train")
        if stage in [None, "test", "predict"]:
            self.test_dataset = Front3DMixedDataset(self.cfg, "test")

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