from dataclasses import dataclass
import math

import torch
import numpy as np
import random
import time
import trimesh
import torch.nn as nn
from einops import repeat, rearrange
from tqdm import trange
from itertools import product

import craftsman
from craftsman.models.transformers.perceiver_1d import Perceiver
from craftsman.models.transformers.attention import ResidualCrossAttentionBlock
from craftsman.models.transformers.utils import init_linear, MLP
from craftsman.utils.checkpoint import checkpoint
from craftsman.utils.base import BaseModule
from craftsman.utils.typing import *
from craftsman.utils.misc import get_world_size, get_device
from craftsman.utils.ops import generate_dense_grid_points
from craftsman.models.geometry.utils import FlexiCubes

import pdb

###################### Utils 
@craftsman.register("pose-ae")
class Pose_autoencoder(BaseModule):
    r"""
    A VAE model for encoding shapes into latents and decoding latent representations into shapes.
    """

    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        num_latents: int = 5
        embed_point_feats: bool = False
        in_dim: int = 6
        out_dim: int = 6
        embed_dim: int = 64
        embed_type: str = "fourier"
        num_freqs: int = 8
        include_pi: bool = True
        init_scale: float = 0.25
        num_tokens: int = 1
        enable_translation: bool = False
        encode_size_translation: bool = False
        context_dim: int = 1024
        enable_ln_affine: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()
        
        init_scale = self.cfg.init_scale
        self.latent_shape = (self.cfg.num_latents, self.cfg.num_tokens, self.cfg.embed_dim)
        
        # rotation
        self.input_proj = nn.Sequential(
            nn.Linear(self.cfg.in_dim, self.cfg.embed_dim),
            nn.LayerNorm(self.cfg.embed_dim),
            MLP(width=self.cfg.embed_dim, init_scale=init_scale),
            nn.LayerNorm(self.cfg.embed_dim, elementwise_affine=self.cfg.enable_ln_affine),
        )
        self.out_proj = nn.Sequential(
            MLP(width=self.cfg.embed_dim, init_scale=init_scale),
            nn.LayerNorm(self.cfg.embed_dim),
            nn.Linear(self.cfg.embed_dim, self.cfg.out_dim),
        )
        
        # translation
        self.input_proj_trans = nn.Sequential(
            nn.Linear(3, self.cfg.embed_dim),
            nn.LayerNorm(self.cfg.embed_dim),
            MLP(width=self.cfg.embed_dim, init_scale=init_scale),
            nn.LayerNorm(self.cfg.embed_dim, elementwise_affine=self.cfg.enable_ln_affine),
        )
        self.out_proj_trans = nn.Sequential(
            MLP(width=self.cfg.embed_dim, init_scale=init_scale),
            nn.LayerNorm(self.cfg.embed_dim),
            nn.Linear(self.cfg.embed_dim, 3),
        )
        
        # size
        self.input_proj_size = nn.Sequential(
            nn.Linear(1, self.cfg.embed_dim),
            nn.LayerNorm(self.cfg.embed_dim),
            MLP(width=self.cfg.embed_dim, init_scale=init_scale),
            nn.LayerNorm(self.cfg.embed_dim, elementwise_affine=self.cfg.enable_ln_affine),
        )
        self.out_proj_size = nn.Sequential(
            MLP(width=self.cfg.embed_dim, init_scale=init_scale),
            nn.LayerNorm(self.cfg.embed_dim),
            nn.Linear(self.cfg.embed_dim, 1),
        )

        if self.cfg.pretrained_model_name_or_path != "":
            print(f"Loading pretrained pose model from {self.cfg.pretrained_model_name_or_path}")
            pretrained_ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")
            if 'state_dict' in pretrained_ckpt:
                _pretrained_ckpt = {}
                for k, v in pretrained_ckpt['state_dict'].items():
                    if k.startswith('pose_model.'):
                        _pretrained_ckpt[k.replace('pose_model.', '')] = v
                pretrained_ckpt = _pretrained_ckpt
                self.load_state_dict(pretrained_ckpt, strict=True)
            else:
                _pretrained_ckpt = {}
                for k, v in pretrained_ckpt.items():
                    if k.startswith('pose_model.'):
                        _pretrained_ckpt[k.replace('pose_model.', '')] = v
                pretrained_ckpt = _pretrained_ckpt
                self.load_state_dict(pretrained_ckpt, strict=True)
            
    def forward(self, rotation: torch.FloatTensor, translation: torch.FloatTensor=None, size: torch.FloatTensor=None, \
                sample_posterior: bool = True):
        # encode
        pose_latents = self.encode(rotation=rotation, translation=translation, size=size) # [B, num_latents, embed_dim]
        # decode
        rotation, translation, size = self.decode(pose_latents) # [B, num_latents, width]

        return rotation, translation, size, pose_latents
    
    
    def encode(self, rotation: torch.FloatTensor, translation: torch.FloatTensor=None, size: torch.FloatTensor=None, \
                sample_posterior: bool = True):
        """
        Args:
            surface (torch.FloatTensor): [B, N, 6]
            sample_posterior (bool):

        Returns:
            shape_latents (torch.FloatTensor): [B, num_latents, width]
        """
        assert rotation.shape[-1] == 6, f"\
            Expected {6} channels, got {rotation.shape[-1]}"

        bs, N, D = rotation.shape
        data = self.input_proj(rotation)
        translation = self.input_proj_trans(translation)
        size = self.input_proj_size(size)
        data = torch.stack([data, translation, size], dim=-2)
            
        return data
    
    
    def encode_size_translation(self, translation: torch.FloatTensor=None, size: torch.FloatTensor=None, sample_posterior: bool = True):
        """
        Args:
            surface (torch.FloatTensor): [B, N, 6]
            sample_posterior (bool):

        Returns:
            shape_latents (torch.FloatTensor): [B, num_latents, width]
        """
        
        translation = self.input_proj_trans(translation)
        size = self.input_proj_size(size)
        data = torch.stack([translation, size], dim=-2)
            
        return data


    def decode(self, latents: torch.FloatTensor):
        """
        Args:
            latents (torch.FloatTensor): [B, T, N, embed_dim]

        Returns:
            latents (torch.FloatTensor): [B, T, out_dim]
        """
        rotation = self.out_proj(latents[:,:,0])
        translation = self.out_proj_trans(latents[:,:,1])
        size = self.out_proj_size(latents[:,:,2])
        
        return torch.cat([rotation, translation, size], dim=-1)

