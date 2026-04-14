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
class FlexiCubesPlus(FlexiCubes):
    @torch.no_grad()
    def _get_case_id(self, occ_fx8, surf_cubes, res):
        return (occ_fx8[surf_cubes] * self.cube_corners_idx.to(self.device).unsqueeze(0)).sum(-1)


VALID_EMBED_TYPES = ["identity", "fourier", "learned_fourier", "siren"]

class FourierEmbedder(nn.Module):
    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x

class LearnedFourierEmbedder(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        per_channel_dim = half_dim // input_dim
        self.weights = nn.Parameter(torch.randn(per_channel_dim))

        self.out_dim = self.get_dims(input_dim)

    def forward(self, x):
        # [b, t, c, 1] * [1, d] = [b, t, c, d] -> [b, t, c * d]
        freqs = (x[..., None] * self.weights[None] * 2 * np.pi).view(*x.shape[:-1], -1)
        fouriered = torch.cat((x, freqs.sin(), freqs.cos()), dim=-1)
        return fouriered
    
    def get_dims(self, input_dim):
        return input_dim * (self.weights.shape[0] * 2 + 1)

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
    
class Siren(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        w0 = 1.,
        c = 6.,
        is_first = False,
        use_bias = True,
        activation = None,
        dropout = 0.
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_first = is_first

        weight = torch.zeros(out_dim, in_dim)
        bias = torch.zeros(out_dim) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)
    
    def init_(self, weight, bias, c, w0):
        dim = self.in_dim

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out
    
def get_embedder(embed_type="fourier", num_freqs=-1, input_dim=3, include_pi=True):
    if embed_type == "identity" or (embed_type == "fourier" and num_freqs == -1):
        return nn.Identity(), input_dim

    elif embed_type == "fourier":
        embedder_obj = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

    elif embed_type == "learned_fourier":
        embedder_obj = LearnedFourierEmbedder(in_channels=input_dim, dim=num_freqs)
    
    elif embed_type == "siren":
        embedder_obj = Siren(in_dim=input_dim, out_dim=num_freqs * input_dim * 2 + input_dim)

    else:
        raise ValueError(f"{embed_type} is not valid. Currently only supprts {VALID_EMBED_TYPES}")
    return embedder_obj




@craftsman.register("pcd-encoder")
class Pcdencoder(BaseModule):
    r"""
    A VAE model for encoding shapes into latents and decoding latent representations into shapes.
    """

    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        embed_point_feats: bool = False
        embed_type: str = "fourier"
        num_freqs: int = 8
        include_pi: bool = True
        out_dim: int = 768
        init_scale: float = 0.25
        num_latents: int = 2048

    cfg: Config

    def configure(self) -> None:
        super().configure()
        
        init_scale = self.cfg.init_scale
        
        # encoder
        self.embedder = get_embedder(embed_type=self.cfg.embed_type, num_freqs=self.cfg.num_freqs, include_pi=self.cfg.include_pi)
        self.input_proj = nn.Sequential(
            nn.Linear(self.embedder.out_dim, self.cfg.out_dim),
            nn.RMSNorm(self.cfg.out_dim),
            nn.GELU(),
            MLP(width=self.cfg.out_dim, init_scale=init_scale),
            nn.RMSNorm(self.cfg.out_dim),
            nn.GELU()
        )

        if self.cfg.pretrained_model_name_or_path != "":
            print(f"Loading pretrained VAE model from {self.cfg.pretrained_model_name_or_path}")
            pretrained_ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")
            if 'state_dict' in pretrained_ckpt:
                _pretrained_ckpt = {}
                for k, v in pretrained_ckpt['state_dict'].items():
                    if k.startswith('pcd_model.'):
                        _pretrained_ckpt[k.replace('pcd_model.', '')] = v
                pretrained_ckpt = _pretrained_ckpt
                self.load_state_dict(pretrained_ckpt, strict=True)
            else:
                _pretrained_ckpt = {}
                for k, v in pretrained_ckpt.items():
                    if k.startswith('pcd_model.'):
                        _pretrained_ckpt[k.replace('pcd_model.', '')] = v
                pretrained_ckpt = _pretrained_ckpt
                self.load_state_dict(pretrained_ckpt, strict=True)
            
    
    def encode(self,
               pc: torch.FloatTensor,
               ):
        """
        Args:
            surface (torch.FloatTensor): [B, N, 3+C]
            sample_posterior (bool):

        Returns:
            shape_latents (torch.FloatTensor): [B, num_latents, out_dim]
        """
        assert pc.shape[-1] == 3, f"\
            Expected {3} channels, got {pc.shape[-1]}"

        bs, N, D = pc.shape
        data = self.embedder(pc)
        data = self.input_proj(data)

        return data




