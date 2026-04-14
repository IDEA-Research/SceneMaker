from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import math
import importlib
import craftsman
import re

from typing import Optional
from craftsman.utils.base import BaseModule
from craftsman.models.denoisers.utils import *
from .pixart_denoiser import PixArtDenoiser

@craftsman.register("dit-pose-denoiser")
class PoseDiTDenoiser(PixArtDenoiser):
    @dataclass
    class Config(PixArtDenoiser.Config):
        pretrained_model_name_or_path: Optional[str] = None
        input_channels: int = 32
        output_channels: int = 32
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
        use_flash: bool = True
        
        num_img_latents: int = 257
        num_shape_latents: int = 512
        num_pose_latents: int = 1
        num_pcd_latents: int = 512
        num_text_latents: int = 77
        use_rope: bool = False
        use_cross_rope: bool = False
        use_pe: bool = False
        use_scene_img: bool = True
        use_scene_mask: bool = True
        use_scene_pcd: bool = True
        use_caption: bool = False
        attn_mode: str = "scene"
        block_mode: str = "full"
        proj_mode: str = "full"

    cfg: Config

    def configure(self) -> None:
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
     
        if self.cfg.proj_mode == "sep":
            self.pose_emb = nn.Sequential(
                nn.RMSNorm(self.cfg.context_dim),
                nn.Linear(self.cfg.context_dim, self.cfg.width),
            )
        elif self.cfg.proj_mode == "linear":
            self.pose_emb = nn.Linear(self.cfg.context_dim, self.cfg.width)

        # pose embedding
        if self.cfg.use_pe:
            self.pose_positional_embedding = nn.Parameter(torch.zeros(1, 1, self.cfg.num_pose_latents, self.cfg.width))
            nn.init.normal_(self.pose_positional_embedding, std=0.02)
        
        # blocks
        init_scale = self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width)
        drop_path = [x.item() for x in torch.linspace(0, self.cfg.drop_path, self.cfg.layers)]
        if self.cfg.block_mode == "full":
            self.blocks = nn.ModuleList([
                DiTBlock_pose(
                        width=self.cfg.width, 
                        heads=self.cfg.curr_heads, 
                        init_scale=init_scale, 
                        qkv_bias=self.cfg.qkv_bias, 
                        use_flash=self.cfg.use_flash,
                        drop_path=drop_path[i],
                        use_rope=self.cfg.use_rope,
                        attn_mode=self.cfg.attn_mode,
                )
                for i in range(self.cfg.layers)
            ])
        
        elif self.cfg.block_mode == "sep":
            self.blocks = nn.ModuleList([
                DiTBlock_pose_sep(
                        width=self.cfg.width, 
                        heads=self.cfg.curr_heads, 
                        init_scale=init_scale, 
                        qkv_bias=self.cfg.qkv_bias, 
                        use_flash=self.cfg.use_flash,
                        drop_path=drop_path[i],
                        use_rope=self.cfg.use_rope,
                        attn_mode=self.cfg.attn_mode,
                        num_img_latents=self.cfg.num_img_latents,
                        num_pcd_latents=self.cfg.num_pcd_latents,
                        num_pose_latents=self.cfg.num_pose_latents,
                        num_text_latents=self.cfg.num_text_latents,
                        use_caption=self.cfg.use_caption,
                        use_scene_img=self.cfg.use_scene_img,
                        use_scene_mask=self.cfg.use_scene_mask,
                        use_scene_pcd=self.cfg.use_scene_pcd,
                )
                for i in range(self.cfg.layers)
            ])
            
        elif self.cfg.block_mode == "sep_split":
            self.blocks = nn.ModuleList([
                DiTBlock_pose_sep_split(
                        width=self.cfg.width, 
                        heads=self.cfg.curr_heads, 
                        init_scale=init_scale, 
                        qkv_bias=self.cfg.qkv_bias, 
                        use_flash=self.cfg.use_flash,
                        drop_path=drop_path[i],
                        use_rope=self.cfg.use_rope,
                        attn_mode=self.cfg.attn_mode,
                        num_img_latents=self.cfg.num_img_latents,
                        num_pcd_latents=self.cfg.num_pcd_latents,
                        num_pose_latents=self.cfg.num_pose_latents,
                        num_text_latents=self.cfg.num_text_latents,
                        use_caption=self.cfg.use_caption,
                        use_scene_img=self.cfg.use_scene_img,
                        use_scene_mask=self.cfg.use_scene_mask,
                        use_scene_pcd=self.cfg.use_scene_pcd,
                )
                for i in range(self.cfg.layers)
            ])
        
        elif self.cfg.block_mode == "sep_mlp":
            self.blocks = nn.ModuleList([
                DiTBlock_pose_sep_mlp(
                        width=self.cfg.width, 
                        heads=self.cfg.curr_heads, 
                        init_scale=init_scale, 
                        qkv_bias=self.cfg.qkv_bias, 
                        use_flash=self.cfg.use_flash,
                        drop_path=drop_path[i],
                        use_rope=self.cfg.use_rope,
                        use_cross_rope=self.cfg.use_cross_rope,
                        attn_mode=self.cfg.attn_mode,
                        num_img_latents=self.cfg.num_img_latents,
                        num_pcd_latents=self.cfg.num_pcd_latents,
                        num_pose_latents=self.cfg.num_pose_latents,
                        num_text_latents=self.cfg.num_text_latents,
                        use_caption=self.cfg.use_caption,
                        use_scene_img=self.cfg.use_scene_img,
                        use_scene_mask=self.cfg.use_scene_mask,
                        use_scene_pcd=self.cfg.use_scene_pcd,
                )
                for i in range(self.cfg.layers)
            ])
            
        elif self.cfg.block_mode == "sep_mlp_rot":
            self.blocks = nn.ModuleList([
                DiTBlock_pose_sep_mlp_rot(
                        width=self.cfg.width, 
                        heads=self.cfg.curr_heads, 
                        init_scale=init_scale, 
                        qkv_bias=self.cfg.qkv_bias, 
                        use_flash=self.cfg.use_flash,
                        drop_path=drop_path[i],
                        use_rope=self.cfg.use_rope,
                        use_cross_rope=self.cfg.use_cross_rope,
                        attn_mode=self.cfg.attn_mode,
                        num_img_latents=self.cfg.num_img_latents,
                        num_pcd_latents=self.cfg.num_pcd_latents,
                        num_pose_latents=self.cfg.num_pose_latents,
                        num_text_latents=self.cfg.num_text_latents,
                        use_caption=self.cfg.use_caption,
                        use_scene_img=self.cfg.use_scene_img,
                        use_scene_mask=self.cfg.use_scene_mask,
                        use_scene_pcd=self.cfg.use_scene_pcd,
                )
                for i in range(self.cfg.layers)
            ])
        
        elif self.cfg.block_mode == "sep_split_all":
            self.blocks = nn.ModuleList([
                DiTBlock_pose_sep_split_all(
                        width=self.cfg.width, 
                        heads=self.cfg.curr_heads, 
                        init_scale=init_scale, 
                        qkv_bias=self.cfg.qkv_bias, 
                        use_flash=self.cfg.use_flash,
                        drop_path=drop_path[i],
                        use_rope=self.cfg.use_rope,
                        attn_mode=self.cfg.attn_mode,
                        num_img_latents=self.cfg.num_img_latents,
                        num_pcd_latents=self.cfg.num_pcd_latents,
                        num_pose_latents=self.cfg.num_pose_latents,
                        num_text_latents=self.cfg.num_text_latents,
                        use_caption=self.cfg.use_caption,
                        use_scene_img=self.cfg.use_scene_img,
                        use_scene_mask=self.cfg.use_scene_mask,
                        use_scene_pcd=self.cfg.use_scene_pcd,
                )
                for i in range(self.cfg.layers)
            ])
        
        # both scene-level and object-level attention
        elif self.cfg.block_mode == "sep_both":
            self.blocks = nn.ModuleList([
                DiTBlock_pose_sep_both(
                        width=self.cfg.width, 
                        heads=self.cfg.curr_heads, 
                        init_scale=init_scale, 
                        qkv_bias=self.cfg.qkv_bias, 
                        use_flash=self.cfg.use_flash,
                        drop_path=drop_path[i],
                        use_rope=self.cfg.use_rope,
                        attn_mode=self.cfg.attn_mode,
                        num_img_latents=self.cfg.num_img_latents,
                        num_pcd_latents=self.cfg.num_pcd_latents,
                        num_pose_latents=self.cfg.num_pose_latents,
                        num_text_latents=self.cfg.num_text_latents,
                        use_caption=self.cfg.use_caption,
                        use_scene_img=self.cfg.use_scene_img,
                        use_scene_mask=self.cfg.use_scene_mask,
                        use_scene_pcd=self.cfg.use_scene_pcd,
                )
                for i in range(self.cfg.layers)
            ])

        # both scene-level and object-level attention and cross-attn containing rotation
        elif self.cfg.block_mode == "sep_both_only_gca":
            self.blocks = nn.ModuleList([
                DiTBlock_pose_sep_both_only_gca(
                        width=self.cfg.width, 
                        heads=self.cfg.curr_heads, 
                        init_scale=init_scale, 
                        qkv_bias=self.cfg.qkv_bias, 
                        use_flash=self.cfg.use_flash,
                        drop_path=drop_path[i],
                        use_rope=self.cfg.use_rope,
                        attn_mode=self.cfg.attn_mode,
                        num_img_latents=self.cfg.num_img_latents,
                        num_pcd_latents=self.cfg.num_pcd_latents,
                        num_pose_latents=self.cfg.num_pose_latents,
                        num_text_latents=self.cfg.num_text_latents,
                        use_caption=self.cfg.use_caption,
                        use_scene_img=self.cfg.use_scene_img,
                        use_scene_mask=self.cfg.use_scene_mask,
                        use_scene_pcd=self.cfg.use_scene_pcd,
                )
                for i in range(self.cfg.layers)
            ])
            
        # both scene-level and object-level attention and cross-attn containing rotation
        elif self.cfg.block_mode == "sep_only_gsa":
            self.blocks = nn.ModuleList([
                DiTBlock_pose_sep_only_gsa(
                        width=self.cfg.width, 
                        heads=self.cfg.curr_heads, 
                        init_scale=init_scale, 
                        qkv_bias=self.cfg.qkv_bias, 
                        use_flash=self.cfg.use_flash,
                        drop_path=drop_path[i],
                        use_rope=self.cfg.use_rope,
                        attn_mode=self.cfg.attn_mode,
                        num_img_latents=self.cfg.num_img_latents,
                        num_pcd_latents=self.cfg.num_pcd_latents,
                        num_pose_latents=self.cfg.num_pose_latents,
                        num_text_latents=self.cfg.num_text_latents,
                        use_caption=self.cfg.use_caption,
                        use_scene_img=self.cfg.use_scene_img,
                        use_scene_mask=self.cfg.use_scene_mask,
                        use_scene_pcd=self.cfg.use_scene_pcd,
                )
                for i in range(self.cfg.layers)
            ])
            
        # both scene-level and object-level attention and cross-attn containing rotation
        elif self.cfg.block_mode == "sep_both_rot":
            self.blocks = nn.ModuleList([
                DiTBlock_pose_sep_both_rot(
                        width=self.cfg.width, 
                        heads=self.cfg.curr_heads, 
                        init_scale=init_scale, 
                        qkv_bias=self.cfg.qkv_bias, 
                        use_flash=self.cfg.use_flash,
                        drop_path=drop_path[i],
                        use_rope=self.cfg.use_rope,
                        attn_mode=self.cfg.attn_mode,
                        num_img_latents=self.cfg.num_img_latents,
                        num_pcd_latents=self.cfg.num_pcd_latents,
                        num_pose_latents=self.cfg.num_pose_latents,
                        num_text_latents=self.cfg.num_text_latents,
                        use_caption=self.cfg.use_caption,
                        use_scene_img=self.cfg.use_scene_img,
                        use_scene_mask=self.cfg.use_scene_mask,
                        use_scene_pcd=self.cfg.use_scene_pcd,
                )
                for i in range(self.cfg.layers)
            ])
            
        elif self.cfg.block_mode == "sep_both_scene":
            self.blocks = nn.ModuleList([
                DiTBlock_pose_sep_both_scene(
                        width=self.cfg.width, 
                        heads=self.cfg.curr_heads, 
                        init_scale=init_scale, 
                        qkv_bias=self.cfg.qkv_bias, 
                        use_flash=self.cfg.use_flash,
                        drop_path=drop_path[i],
                        use_rope=self.cfg.use_rope,
                        attn_mode=self.cfg.attn_mode,
                        num_img_latents=self.cfg.num_img_latents,
                        num_pcd_latents=self.cfg.num_pcd_latents,
                        num_pose_latents=self.cfg.num_pose_latents,
                        num_text_latents=self.cfg.num_text_latents,
                        use_caption=self.cfg.use_caption,
                        use_scene_img=self.cfg.use_scene_img,
                        use_scene_mask=self.cfg.use_scene_mask,
                        use_scene_pcd=self.cfg.use_scene_pcd,
                )
                for i in range(self.cfg.layers)
            ])
            
        
        elif self.cfg.block_mode == "trans":
            self.blocks = nn.ModuleList([
                DiTBlock_pose_trans(
                        width=self.cfg.width, 
                        heads=self.cfg.curr_heads, 
                        init_scale=init_scale, 
                        qkv_bias=self.cfg.qkv_bias, 
                        use_flash=self.cfg.use_flash,
                        drop_path=drop_path[i],
                        use_rope=self.cfg.use_rope,
                        attn_mode=self.cfg.attn_mode,
                        num_img_latents=self.cfg.num_img_latents,
                        num_pcd_latents=self.cfg.num_pcd_latents,
                        num_pose_latents=self.cfg.num_pose_latents,
                        num_text_latents=self.cfg.num_text_latents,
                        use_caption=self.cfg.use_caption,
                        use_scene_img=self.cfg.use_scene_img,
                        use_scene_mask=self.cfg.use_scene_mask,
                        use_scene_pcd=self.cfg.use_scene_pcd,
                )
                for i in range(self.cfg.layers)
            ])
        
        elif self.cfg.block_mode == "trans2":
            self.blocks = nn.ModuleList([
                DiTBlock_pose_trans2(
                        width=self.cfg.width, 
                        heads=self.cfg.curr_heads, 
                        init_scale=init_scale, 
                        qkv_bias=self.cfg.qkv_bias, 
                        use_flash=self.cfg.use_flash,
                        drop_path=drop_path[i],
                        use_rope=self.cfg.use_rope,
                        attn_mode=self.cfg.attn_mode,
                        num_img_latents=self.cfg.num_img_latents,
                        num_pcd_latents=self.cfg.num_pcd_latents,
                        num_pose_latents=self.cfg.num_pose_latents,
                        num_text_latents=self.cfg.num_text_latents,
                        use_caption=self.cfg.use_caption,
                        use_scene_img=self.cfg.use_scene_img,
                        use_scene_mask=self.cfg.use_scene_mask,
                        use_scene_pcd=self.cfg.use_scene_pcd,
                )
                for i in range(self.cfg.layers)
            ])

        self.t_block = nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(self.cfg.width, 6 * self.cfg.width, bias=True)
                    )
        
         # final layer
        self.final_layer = T2IFinalLayer(self.cfg.width, self.cfg.output_channels)

        # self.identity_initialize()

        if self.cfg.pretrained_model_name_or_path:
            print(f"Loading pretrained DiT model from {self.cfg.pretrained_model_name_or_path}")
            ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")
            if "state_dict" in ckpt:
                ckpt = ckpt['state_dict']
            self.denoiser_ckpt = {}

            for k, v in ckpt.items():
                if k.startswith('denoiser_model.'):
                    # print(k.replace('denoiser_model.', ''))
                    
                    module_keys = k.replace('denoiser_model.', '').split(".")
                    final_module = self
                    try:
                        for key in module_keys:
                            final_module = getattr(final_module, key)
                    except:
                        continue
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
            self.load_state_dict(self.denoiser_ckpt, strict=False)


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
                context: torch.FloatTensor,
                attention_kwargs: Dict[str, torch.Tensor] = None,):

        r"""
        Args:
            model_input (torch.FloatTensor): [bs, T, N, c]
            timestep (torch.LongTensor): [bs,]
            context (torch.FloatTensor): [bs, context_tokens, c]

        Returns:
            sample (torch.FloatTensor): [bs, n_data, c]

        """

        B, T, N, C = model_input.shape

        # 1. time
        t_emb = self.time_embed(timestep)

        # 2. conditions projector
        context = context.view(B, T, -1, self.cfg.context_dim)
        if self.cfg.proj_mode == "sep":
            context_pose = context[:, :, -2:, :]
            context = context[:, :, :-2, :]
            context_pose = self.pose_emb(context_pose)
            
        if self.cfg.condition_type == "clip_dinov2":
            clip_feat, dino_feat = context.chunk(2, dim=2)
            clip_cond = self.clip_embed(clip_feat)
            dino_cond = self.dino_embed(dino_feat)
            visual_cond = self.cfg.clip_weight * clip_cond + self.cfg.dino_weight * dino_cond
        elif self.cfg.condition_type == "clip":
            clip_cond = self.clip_embed(context)
            visual_cond = clip_cond
        elif self.cfg.condition_type == "dinov2":
            dino_cond = self.dino_embed(context)
            visual_cond = dino_cond
        else:
            raise NotImplementedError(f"condition type {self.cfg.condition_type} not implemented")
        
        if self.cfg.proj_mode == "sep":
            visual_cond = torch.cat([visual_cond, context_pose], dim=2)
            
        # 4. denoiser
        latent = self.x_embed(model_input)
        
        # add position embedding for pose latents
        if self.cfg.use_pe:
            pose_latents = latent[:, :, :self.cfg.num_pose_latents, :]
            pose_latents = pose_latents + self.pose_positional_embedding  # Add positional embedding
            # Update latent with pose latents
            latent[:, :, :self.cfg.num_pose_latents, :] = pose_latents
        
        
        t0 = self.t_block(t_emb).unsqueeze(dim=1)
        for block in self.blocks:
            latent = auto_grad_checkpoint(block, latent, visual_cond, t0, **attention_kwargs)

        latent = self.final_layer(latent.reshape(B,-1,self.cfg.width), t_emb).reshape(B, T, N, C)

        return latent

