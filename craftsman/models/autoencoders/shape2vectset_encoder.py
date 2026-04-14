from dataclasses import dataclass
import math
import numpy as np

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath
from functools import wraps

import craftsman
from craftsman.models.autoencoders.michelangelo_autoencoder import AutoEncoder
from craftsman.utils.checkpoint import checkpoint
from craftsman.utils.base import BaseModule
from craftsman.utils.typing import *

import pdb


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)
    
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)
    
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))

class PointEmbed(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+input_dim, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed
    
@craftsman.register("shape2vectset-encoder")
class Shape2VectSetAutoencoder(AutoEncoder):
    r"""
    A VAE model for encoding shapes into latents and decoding latent representations into shapes.
    Structure borrowed from the: https://github.com/1zb/3DShape2VecSet/blob/master/models_ae.py
    """

    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        num_latents: int = 512
        point_feats: int = 0
        out_dim: int = 1
        embed_dim: int = 8
        width: int = 512
        heads: int = 8
        dim_head: int = 64
        num_decoder_layers: int = 24
        init_scale: float = 0.25
        qkv_bias: bool = True
        weight_tie_layers: bool = False
        use_flash: bool = False
        use_checkpoint: bool = True
        use_fps: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(self.cfg.width, Attention(self.cfg.width, self.cfg.width + self.cfg.point_feats, heads = 1, dim_head = self.cfg.width), context_dim = self.cfg.width + self.cfg.point_feats),
            PreNorm(self.cfg.width, FeedForward(self.cfg.width))
        ])
        self.point_embed = PointEmbed(dim=self.cfg.width)

        get_latent_attn = lambda: PreNorm(self.cfg.width, Attention(self.cfg.width, heads = self.cfg.heads, dim_head =self.cfg.dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(self.cfg.width, FeedForward(self.cfg.width, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': self.cfg.weight_tie_layers}

        for i in range(self.cfg.num_decoder_layers):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(self.cfg.width, Attention(self.cfg.width, self.cfg.width, heads = 1, dim_head = self.cfg.width), context_dim = self.cfg.width)

        self.to_outputs = nn.Linear(self.cfg.width, self.cfg.out_dim) if exists(self.cfg.out_dim) else nn.Identity()
        
        self.mean_fc = nn.Linear(self.cfg.width, self.cfg.embed_dim)
        self.logvar_fc = nn.Linear(self.cfg.width, self.cfg.embed_dim)

        self.proj = nn.Linear(self.cfg.embed_dim, self.cfg.width)
        
        if self.cfg.pretrained_model_name_or_path != "":
            print(f"Loading pretrained Pcd model from {self.cfg.pretrained_model_name_or_path}")
            pretrained_ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")
            if 'model' in pretrained_ckpt:
                _pretrained_ckpt = {}
                for k, v in pretrained_ckpt['model'].items():
                    if k.startswith('shape_model.'):
                        _pretrained_ckpt[k.replace('shape_model.', '')] = v
                    else:
                        _pretrained_ckpt[k] = v
                pretrained_ckpt = _pretrained_ckpt
                self.load_state_dict(pretrained_ckpt, strict=True)
            else:
                if "hunyuan" in self.cfg.pretrained_model_name_or_path:
                    _pretrained_ckpt = {}
                    for k, v in pretrained_ckpt['vae'].items():
                        if k.startswith('geo_decoder.'):
                            _pretrained_ckpt[k.replace('geo_decoder.', 'decoder.')] = v
                        else:
                            _pretrained_ckpt[k] = v
                    pretrained_ckpt = _pretrained_ckpt
                    self.load_state_dict(pretrained_ckpt, strict=False)
                else:
                    _pretrained_ckpt = {}
                    for k, v in pretrained_ckpt.items():
                        if k.startswith('model.'):
                            _pretrained_ckpt[k.replace('model.', '')] = v
                    pretrained_ckpt = _pretrained_ckpt
                    self.load_state_dict(pretrained_ckpt, strict=True)

    def pre_kl(self, x):
        return torch.cat([self.mean_fc(x), self.logvar_fc(x)], dim=-1)


    def encode(self,
               surface: torch.FloatTensor,
               sample_posterior: bool = True):
        """
        Args:
            surface (torch.FloatTensor): [B, N, 3+C]
            sample_posterior (bool):

        Returns:
            latents (torch.FloatTensor)
            center_pos (torch.FloatTensor or None):
            posterior (DiagonalGaussianDistribution or None):
        """
        assert surface.shape[-1] == 3 + self.cfg.point_feats, f"\
            Expected {3 + self.cfg.point_feats} channels, got {surface.shape[-1]}"
        
        pc, feats = surface[..., :3], surface[..., 3:]

        B, N, D = pc.shape

        if self.cfg.use_fps:
            ###### fps
            from torch_cluster import fps
            flattened = pc.view(B*N, D)

            batch = torch.arange(B).to(pc.device)
            batch = torch.repeat_interleave(batch, N)

            pos = flattened

            # ratio = 1.0 * self.cfg.num_latents / self.num_inputs
            ratio = 1.0 * self.cfg.num_latents / N
            idx = fps(pos.float(), batch, ratio=ratio)

            sampled_pc = pos[idx]
            sampled_pc = sampled_pc.view(B, -1, 3)
            ######
        else:
            sampled_pc = pc

        sampled_pc_embeddings = self.point_embed(sampled_pc)

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks
        latents = cross_attn(sampled_pc_embeddings, context = torch.cat([pc_embeddings, feats], dim=-1), mask = None) + sampled_pc_embeddings
        latents = cross_ff(latents) + latents

        # vae
        # latents, posterior = self.encode_kl_embed(latents, sample_posterior)
        
        # remove vae
        latents = self.mean_fc(latents)
        posterior = None

        return latents, posterior


    def decode(self, 
               latents: torch.FloatTensor):
        """
        Args:
            latents (torch.FloatTensor): [B, E-1]

        Returns:
            latents (torch.FloatTensor): [B, E-1]
        """
        latents = self.proj(latents)

        for self_attn, self_ff in self.layers:
            latents = self_attn(latents) + latents
            latents = self_ff(latents) + latents

        return latents


    def query(self, 
              queries: torch.FloatTensor, 
              latents: torch.FloatTensor):
        """
        Args:
            queries (torch.FloatTensor): [B, N, 3]
            latents (torch.FloatTensor): [B, E-1]

        Returns:
            logits (torch.FloatTensor): [B, N], occupancy logits
        """
        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context = latents)
        
        return self.to_outputs(latents).squeeze(-1)