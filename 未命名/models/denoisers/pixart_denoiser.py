from dataclasses import dataclass

import torch
import torch.nn as nn
import math
import importlib
import craftsman
import re

from typing import Optional
from craftsman.utils.base import BaseModule
from craftsman.models.denoisers.utils import *

import pdb

@craftsman.register("pixart-denoiser")
class PixArtDenoiser(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: Optional[str] = None
        input_channels: int = 32
        output_channels: int = 32
        n_ctx: int = 512
        width: int = 768
        layers: int = 28
        pre_heads: int = 16
        curr_heads: int = 16
        context_dim: int = 1024
        n_views: int = 1
        context_ln: bool = True
        skip_ln: bool = False
        init_scale: float = 0.25
        use_checkpoint: bool = False
        drop_path: float = 0.
        qkv_bias: bool = False
        condition_type: str = "clip_dinov2"
        clip_weight: float = 1.0
        dino_weight: float = 1.0

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # timestep embedding
        self.time_embed = TimestepEmbedder(self.cfg.width)

        # x embedding
        self.x_embed = nn.Linear(self.cfg.input_channels, self.cfg.width, bias=True)

        # context embedding
        if self.cfg.context_ln:
            self.clip_embed = nn.Sequential(
                nn.RMSNorm(self.cfg.context_dim),
                nn.Linear(self.cfg.context_dim, self.cfg.width),
            )

            self.dino_embed = nn.Sequential(
                nn.RMSNorm(self.cfg.context_dim),
                nn.Linear(self.cfg.context_dim, self.cfg.width),
            )
        else:
            self.clip_embed = nn.Linear(self.cfg.context_dim, self.cfg.width)
            self.dino_embed = nn.Linear(self.cfg.context_dim, self.cfg.width)

        init_scale = self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width)
        drop_path = [x.item() for x in torch.linspace(0, self.cfg.drop_path, self.cfg.layers)]
        self.blocks = nn.ModuleList([
            DiTBlock(
                    width=self.cfg.width, 
                    heads=self.cfg.curr_heads, 
                    init_scale=init_scale, 
                    qkv_bias=self.cfg.qkv_bias, 
                    use_flash=True,
                    drop_path=drop_path[i]
            )
            for i in range(self.cfg.layers)
        ])

        self.t_block = nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(self.cfg.width, 6 * self.cfg.width, bias=True)
                    )
        # final layer
        self.final_layer = T2IFinalLayer(self.cfg.width, self.cfg.output_channels)

        self.identity_initialize()

        if self.cfg.pretrained_model_name_or_path:
            print(f"Loading pretrained DiT model from {self.cfg.pretrained_model_name_or_path}")
            ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")
            if "state_dict" in ckpt:
                ckpt = ckpt['state_dict']
            self.denoiser_ckpt = {}
            for k, v in ckpt.items():
                if k.startswith('denoiser_model.'):
                    module_keys = k.replace('denoiser_model.', '').split(".")
                    final_module = self
                    for key in module_keys:
                        final_module = getattr(final_module, key)
                    data = final_module.data
                    data_zero = torch.zeros_like(data).to(v)

                    if data.dim() == 1:
                        if data.shape[0] != v.shape[0]:
                            if re.match(r".*t_block\.1\.bias", k.replace('denoiser_model.', '')):
                                v = v.reshape(6, -1)
                                data_zero = data_zero.reshape(6, -1)
                                data_zero[:, :v.shape[1]] = v
                                data = data_zero.reshape(-1)
                            elif re.match(r".*norm.*\.weight", k.replace('denoiser_model.', '')):   
                                data_zero[:v.shape[0]] = v * (math.sqrt(v.shape[0]) / math.sqrt(data_zero.shape[0]))
                                data = data_zero
                            elif re.match(r"clip_embed.0.weight", k.replace('denoiser_model.', '')):   
                                data_zero[:v.shape[0]] = v * (math.sqrt(v.shape[0]) / math.sqrt(data_zero.shape[0]))
                                data = data_zero
                            elif re.match(r"dino_embed.0.weight", k.replace('denoiser_model.', '')):   
                                data_zero[:v.shape[0]] = v * (math.sqrt(v.shape[0]) / math.sqrt(data_zero.shape[0]))
                                data = data_zero
                            elif re.match(r".*\.bias", k.replace('denoiser_model.', '')):
                                data_zero[:v.shape[0]] = v
                                data = data_zero
                            else:
                                data[:v.shape[0]] = v
                        else:
                            data = v
                    elif data.dim() == 2:
                        if data.shape[0] != v.shape[0] and data.shape[1] != v.shape[1]:
                            if re.match(r".*t_block\.1\.weight", k.replace('denoiser_model.', '')):
                                v = v.reshape(6, -1, v.shape[1])
                                data_zero = data_zero.reshape(6, -1, data.shape[1])
                                data_zero[:, :v.shape[1], :v.shape[1]] = v
                                data = data_zero.reshape(6*data_zero.shape[1], data_zero.shape[1])
                            elif re.match(r".*c_k\.weight", k.replace('denoiser_model.', '')):
                                pre_head_dim = v.shape[0] // self.cfg.pre_heads
                                curr_head_dim = data_zero.shape[0] // self.cfg.curr_heads
                                data_zero[:v.shape[0], :v.shape[1]] = v * (math.sqrt(curr_head_dim) / math.sqrt(pre_head_dim))
                                data = data_zero
                            elif re.match(r".*c_proj\.weight", k.replace('denoiser_model.', '')):
                                data_zero[:v.shape[0], :v.shape[1]] = v
                                data = data_zero
                            else:
                                data[:v.shape[0], :v.shape[1]] = v
                        elif data.shape[0] != v.shape[0] and data.shape[1] == v.shape[1]:
                            data_zero[:v.shape[0], :v.shape[1]] = v
                            data = data_zero
                        elif data.shape[0] == v.shape[0] and data.shape[1] != v.shape[1]:
                            data_zero[:v.shape[0], :v.shape[1]] = v
                            data = data_zero
                        else:
                            data = v
                    self.denoiser_ckpt[k.replace('denoiser_model.', '')] = data
            self.load_state_dict(self.denoiser_ckpt, strict=True)


    def identity_initialize(self):
        for block in self.blocks:
            nn.init.constant_(block.attn.c_proj.weight, 0)
            nn.init.constant_(block.attn.c_proj.bias, 0)
            nn.init.constant_(block.cross_attn.c_proj.weight, 0)
            nn.init.constant_(block.cross_attn.c_proj.bias, 0)
            nn.init.constant_(block.mlp.c_proj.weight, 0)
            nn.init.constant_(block.mlp.c_proj.bias, 0)

    def forward(self,
                model_input: torch.FloatTensor,
                timestep: torch.LongTensor,
                context: torch.FloatTensor):

        r"""
        Args:
            model_input (torch.FloatTensor): [bs, n_data, c]
            timestep (torch.LongTensor): [bs,]
            context (torch.FloatTensor): [bs, context_tokens, c]

        Returns:
            sample (torch.FloatTensor): [bs, n_data, c]

        """

        B, n_data, _ = model_input.shape

        # 1. time
        t_emb = self.time_embed(timestep)

        # 2. conditions projector
        context = context.view(B, self.cfg.n_views, -1, self.cfg.context_dim)
        if self.cfg.condition_type == "clip_dinov2":
            clip_feat, dino_feat = context.chunk(2, dim=2)
            clip_cond = self.clip_embed(clip_feat.contiguous().view(B, -1, self.cfg.context_dim))
            dino_cond = self.dino_embed(dino_feat.contiguous().view(B, -1, self.cfg.context_dim))
            visual_cond = self.cfg.clip_weight * clip_cond + self.cfg.dino_weight * dino_cond
        elif self.cfg.condition_type == "clip":
            clip_cond = self.clip_embed(context.contiguous().view(B, -1, self.cfg.context_dim))
            visual_cond = clip_cond
        elif self.cfg.condition_type == "dinov2":
            dino_cond = self.dino_embed(context.contiguous().view(B, -1, self.cfg.context_dim))
            visual_cond = dino_cond
        else:
            raise NotImplementedError(f"condition type {self.cfg.condition_type} not implemented")
            
        # 4. denoiser
        latent = self.x_embed(model_input)
        
        t0 = self.t_block(t_emb).unsqueeze(dim=1)
        for block in self.blocks:
            latent = auto_grad_checkpoint(block, latent, visual_cond, t0)

        latent = self.final_layer(latent, t_emb)

        return latent

