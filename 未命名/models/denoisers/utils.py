import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from itertools import repeat
from collections.abc import Iterable
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from timm.models.layers import DropPath
from craftsman.models.transformers.utils import MLP
from craftsman.models.transformers.attention_diffusion import MultiheadAttention, MultiheadCrossAttention, MultiheadAttention_pose, MultiheadCrossAttention_split

import pdb

def expand_attn_mask(mask_legal: torch.Tensor, n: int) -> torch.Tensor:
    # mask_legal: (b, t)
    b, t = mask_legal.shape
    tn = t * n

    # 将 (b, t) 扩展成 (b, t*n)
    mask_token = mask_legal.unsqueeze(-1).repeat(1, 1, n)  # (b, t, n)
    mask_token = mask_token.reshape(b, tn)  # (b, t*n)

    # 构造 attention mask (b, t*n, t*n)
    attn_mask = mask_token.unsqueeze(1) & mask_token.unsqueeze(2)  # broadcast and逻辑
    return attn_mask  # bool tensor


class MMDiTBlock(nn.Module):
    """
    A MMDiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0):
        super().__init__()
        self.norm1_shape = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.norm1_image = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.attn = MultiheadAttention(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash
        )
        self.norm2_shape = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.norm2_image = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp_shape = MLP(width=width, init_scale=init_scale)
        self.mlp_image = MLP(width=width, init_scale=init_scale)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(12, width) / width ** 0.5)

    def forward(self, x, y, t, **kwargs):
        B, N, C = x.shape

        shift_msa_shape, scale_msa_shape, gate_msa_shape, shift_mlp_shape, scale_mlp_shape, gate_mlp_shape, \
            shift_msa_image, scale_msa_image, gate_msa_image, shift_mlp_image, scale_mlp_image, gate_mlp_image = (self.scale_shift_table[None] + t.reshape(B, 6, -1).repeat(1, 2, 1)).chunk(12, dim=1)
        
        x_out = t2i_modulate(self.norm1_shape(x), shift_msa_shape, scale_msa_shape)
        y_out = t2i_modulate(self.norm1_image(y), shift_msa_image, scale_msa_image)

        # concate the shape and image features
        xy = self.attn(torch.cat([x_out, y_out], dim=1))

        # skip connection
        x = x + self.drop_path(gate_msa_shape * xy[:, :N])
        y = y + self.drop_path(gate_msa_image * xy[:, N:])

        x = x + self.drop_path(gate_mlp_shape * self.mlp_shape(t2i_modulate(self.norm2_shape(x), shift_mlp_shape, scale_mlp_shape)))
        y = y + self.drop_path(gate_mlp_image * self.mlp_image(t2i_modulate(self.norm2_image(y), shift_mlp_image, scale_mlp_image)))

        return x, y

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
        )
        self.norm2 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)

    def forward(self, x, y, t, **kwargs):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.cross_attn(x, y)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x
 
    
class DiTBlock_scene(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
        )
        self.cross_attn_obj = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
        )
        self.norm2 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)

    def forward(self, x, y, t, **kwargs):
        # split scene shape and object shape
        x, x_obj = torch.chunk(x, 2, dim=-2)

        # reshape
        B, T, N, C = x.shape
        x = x.reshape(B*T, N, C)  # reshape for shape attention
        x_obj = x_obj.reshape(B*T, N, C)

        # self-attn + cross-attn for scene shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B*T, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B*T, N, C))
        x = x + self.cross_attn(x, y)

        # cross-attn for object shape
        x = x + self.cross_attn_obj(x, x_obj)

        # mlp for scene shape
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        # concat scene shape and object shape
        x = torch.cat([x, x_obj], dim=1)
        x = x.reshape(B, T, N*2, C)  # reshape back

        return x
    

class DiTBlock_pose(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0, use_rope=False, attn_mode="scene", use_caption=False):
        super().__init__()
        self.norm1 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.norm2 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        
        self.attn_mode = attn_mode

    def forward(self, x, y, t, **kwargs):
        B, T, N, C = x.shape
        
        if self.attn_mode == "scene":
            # reshape for self-attn
            x = rearrange(x, 'b t n c -> b (t n) c')
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
            x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, T*N, C))
            # reshape for cross-attn
            x = x.reshape(B*T, N, C)
            y = y.reshape(B*T, *y.shape[-2:])
            x = x + self.cross_attn(x, y)
            x = x + self.drop_path(gate_mlp.repeat_interleave(T, dim=0) * self.mlp(t2i_modulate(self.norm2(x), shift_mlp.repeat_interleave(T, dim=0), scale_mlp.repeat_interleave(T, dim=0))))
        else:
            # reshape for self-attn
            x = rearrange(x, 'b t n c -> (b t) n c')
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1).repeat_interleave(T, dim=0)).chunk(6, dim=1)
            x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B*T, N, C))
            # reshape for cross-attn
            x = x.reshape(B*T, N, C)
            y = y.reshape(B*T, *y.shape[-2:])
            x = x + self.cross_attn(x, y)
            x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        
        # reshape
        x = x.reshape(B, T, N, C)

        return x
    

class DiTBlock_pose_sep(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0, use_rope=False, attn_mode="scene", \
        num_img_latents=257, num_pcd_latents=512, num_pose_latents=3, num_text_latents=77, freeze_rot=False, \
        use_caption=False, use_scene_img=False, use_scene_pcd=False, use_scene_mask=False):
        
        super().__init__()
        self.norm1 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.norm2 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        
        # translation estimation
        self.cross_attn_trans = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        
        self.use_caption = use_caption
        self.attn_mode = attn_mode
        self.num_img_latents = num_img_latents
        self.num_pose_latents = num_pose_latents
        self.num_pcd_latents = num_pcd_latents
        self.num_text_latents = num_text_latents
        
        if freeze_rot:
            self.norm1.requires_grad_(False)
            self.attn.requires_grad_(False)
            self.cross_attn.requires_grad_(False)
            self.norm2.requires_grad_(False)
            self.mlp.requires_grad_(False)
            self.scale_shift_table.requires_grad_(False)
            

    def forward(self, x, y, t, **kwargs):
        B, T, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        # repeat for mlp
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.repeat_interleave(T, dim=0), scale_mlp.repeat_interleave(T, dim=0), gate_mlp.repeat_interleave(T, dim=0)
        
        # self-attn
        if self.attn_mode == "scene":
            # attn mask for legal object
            mask_legal = kwargs.get("mask_legal")
            attn_mask = expand_attn_mask(mask_legal.bool(), n=N).to(x) 
            # scene-level attn
            x = rearrange(x, 'b t n c -> b (t n) c')
            x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask).reshape(B, T*N, C))
        else:
            # object-level attn
            x = rearrange(x, 'b t n c -> (b t) n c')
            shift_msa, scale_msa, gate_msa = (shift_msa.repeat_interleave(T, dim=0), scale_msa.repeat_interleave(T, dim=0), gate_msa.repeat_interleave(T, dim=0))
            x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=torch.ones((B*T,N,N)).bool().to(x)).reshape(B*T, N, C))
        
        # reshape for cross-attn
        x = x.reshape(B*T, N, C)
        y = y.reshape(B*T, *y.shape[-2:])
        
        # rotation cross-attn
        x_rot = torch.cat([x[:, :1], x[:, self.num_pose_latents:]], dim=1)
        y_rot = y[:,:self.num_img_latents+self.num_pcd_latents]
        if self.use_caption:
            y_rot = torch.cat([y_rot, y[:,-self.num_text_latents:]], dim=1)
        x[:, :1] = x[:, :1] + self.cross_attn(x_rot, y_rot)[:, :1]
        
        # translation cross-attn
        x_trans = x[:,1:]
        y_trans = y[:,self.num_img_latents+self.num_pcd_latents:]
        x[:,1:self.num_pose_latents] = x[:,1:self.num_pose_latents] + self.cross_attn_trans(x_trans, y_trans)[:,:self.num_pose_latents-1]
        
        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        
        # reshape
        x = x.reshape(B, T, N, C)

        return x
    
class DiTBlock_pose_sep_split(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0, use_rope=False, attn_mode="scene", \
        num_img_latents=257, num_pcd_latents=512, num_pose_latents=3, num_text_latents=77, freeze_rot=False, \
        use_caption=False, use_scene_img=False, use_scene_pcd=False, use_scene_mask=False):
        
        super().__init__()
        self.norm1 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.norm2 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        
        # translation estimation
        self.cross_attn_trans = MultiheadCrossAttention_split(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        
        self.use_caption = use_caption
        self.attn_mode = attn_mode
        self.num_img_latents = num_img_latents
        self.num_pose_latents = num_pose_latents
        self.num_pcd_latents = num_pcd_latents
        self.num_text_latents = num_text_latents
        
        if freeze_rot:
            self.norm1.requires_grad_(False)
            self.attn.requires_grad_(False)
            self.cross_attn.requires_grad_(False)
            self.norm2.requires_grad_(False)
            self.mlp.requires_grad_(False)
            self.scale_shift_table.requires_grad_(False)
            

    def forward(self, x, y, t, **kwargs):
        B, T, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        # repeat for mlp
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.repeat_interleave(T, dim=0), scale_mlp.repeat_interleave(T, dim=0), gate_mlp.repeat_interleave(T, dim=0)
        
        # self-attn
        if self.attn_mode == "scene":
            # attn mask for legal object
            mask_legal = kwargs.get("mask_legal")
            attn_mask = expand_attn_mask(mask_legal.bool(), n=N).to(x) 
            # scene-level attn
            x = rearrange(x, 'b t n c -> b (t n) c')
            x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask).reshape(B, T*N, C))
        else:
            # object-level attn
            x = rearrange(x, 'b t n c -> (b t) n c')
            shift_msa, scale_msa, gate_msa = (shift_msa.repeat_interleave(T, dim=0), scale_msa.repeat_interleave(T, dim=0), gate_msa.repeat_interleave(T, dim=0))
            x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=torch.ones((B*T,N,N)).bool().to(x)).reshape(B*T, N, C))
        
        # reshape for cross-attn
        x = x.reshape(B*T, N, C)
        y = y.reshape(B*T, *y.shape[-2:])
        
        # rotation cross-attn
        x_rot = torch.cat([x[:, :1], x[:, self.num_pose_latents:]], dim=1)
        y_rot = y[:,:self.num_img_latents+self.num_pcd_latents]
        if self.use_caption:
            y_rot = torch.cat([y_rot, y[:,-self.num_text_latents:]], dim=1)
        x[:, :1] = x[:, :1] + self.cross_attn(x_rot, y_rot)[:, :1]
        
        # translation cross-attn
        x_trans = x[:,1:]
        y_trans = y[:,self.num_img_latents+self.num_pcd_latents:]
        x[:,1:self.num_pose_latents] = x[:,1:self.num_pose_latents] + self.cross_attn_trans(x_trans, y_trans)[:,:self.num_pose_latents-1]
        
        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        
        # reshape
        x = x.reshape(B, T, N, C)

        return x
    
class DiTBlock_pose_sep_split_all(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0, use_rope=False, attn_mode="scene", \
        num_img_latents=257, num_pcd_latents=512, num_pose_latents=3, num_text_latents=77, freeze_rot=False, \
        use_caption=False, use_scene_img=False, use_scene_pcd=False, use_scene_mask=False):
        
        super().__init__()
        self.norm1 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.norm2 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        
        # translation estimation
        self.cross_attn_trans = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        
        # size estimation
        self.cross_attn_size = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        
        self.use_caption = use_caption
        self.attn_mode = attn_mode
        self.num_img_latents = num_img_latents
        self.num_pose_latents = num_pose_latents
        self.num_pcd_latents = num_pcd_latents
        self.num_text_latents = num_text_latents
        
        if freeze_rot:
            self.norm1.requires_grad_(False)
            self.attn.requires_grad_(False)
            self.cross_attn.requires_grad_(False)
            self.norm2.requires_grad_(False)
            self.mlp.requires_grad_(False)
            self.scale_shift_table.requires_grad_(False)
            

    def forward(self, x, y, t, **kwargs):
        B, T, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        # repeat for mlp
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.repeat_interleave(T, dim=0), scale_mlp.repeat_interleave(T, dim=0), gate_mlp.repeat_interleave(T, dim=0)
        
        # self-attn
        if self.attn_mode == "scene":
            # attn mask for legal object
            mask_legal = kwargs.get("mask_legal")
            attn_mask = expand_attn_mask(mask_legal.bool(), n=N).to(x) 
            # scene-level attn
            x = rearrange(x, 'b t n c -> b (t n) c')
            x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask).reshape(B, T*N, C))
        else:
            # object-level attn
            x = rearrange(x, 'b t n c -> (b t) n c')
            shift_msa, scale_msa, gate_msa = (shift_msa.repeat_interleave(T, dim=0), scale_msa.repeat_interleave(T, dim=0), gate_msa.repeat_interleave(T, dim=0))
            x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=torch.ones((B*T,N,N)).bool().to(x)).reshape(B*T, N, C))
        
        # reshape for cross-attn
        x = x.reshape(B*T, N, C)
        y = y.reshape(B*T, *y.shape[-2:])
        
        # rotation cross-attn
        x_rot = torch.cat([x[:, :1], x[:, self.num_pose_latents:]], dim=1)
        y_rot = y[:,:self.num_img_latents+self.num_pcd_latents]
        if self.use_caption:
            y_rot = torch.cat([y_rot, y[:,-self.num_text_latents:]], dim=1)
        x_rot = x[:, :1] + self.cross_attn(x_rot, y_rot)[:, :1]
        
        # translation cross-attn
        x_trans = x[:,1:2]
        mark = self.num_img_latents + self.num_pcd_latents
        y_trans = torch.cat([y[:,mark:mark+1], y[:,mark+2:]], dim=1) if y.shape[1] > (mark + 2) else y[:,mark:mark+1]
        x_trans = x_trans + self.cross_attn_trans(x_trans, y_trans)[:,:1]
        
        # size cross-attn
        x_size = x[:,2:3]
        y_size = y[:,mark+1:]
        x_size = x_size + self.cross_attn_size(x_size, y_size)[:,:1]
        
        # concatenate the features
        x = torch.cat([x_rot, x_trans, x_size, x[:,3:]], dim=1)
        
        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        
        # reshape
        x = x.reshape(B, T, N, C)

        return x
    
class DiTBlock_pose_sep_mlp(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0, use_rope=False, use_cross_rope=False, attn_mode="scene", \
        num_img_latents=257, num_pcd_latents=512, num_pose_latents=3, num_text_latents=77, freeze_rot=False, \
        use_caption=False, use_scene_img=False, use_scene_pcd=False, use_scene_mask=False):
        
        super().__init__()
        self.norm1 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_cross_rope,
        )
        self.norm2 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        self.scale_shift_table_trans = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        
        # translation estimation
        self.norm_trans = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp_trans = MLP(width=width, init_scale=init_scale)
        self.norm_size = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp_size = MLP(width=width, init_scale=init_scale)
        
        self.cross_attn_trans = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_cross_rope,
        )
        
        self.use_caption = use_caption
        self.attn_mode = attn_mode
        self.num_img_latents = num_img_latents
        self.num_pose_latents = num_pose_latents
        self.num_pcd_latents = num_pcd_latents
        self.num_text_latents = num_text_latents
        
        if freeze_rot:
            self.norm1.requires_grad_(False)
            self.attn.requires_grad_(False)
            self.cross_attn.requires_grad_(False)
            self.norm2.requires_grad_(False)
            self.mlp.requires_grad_(False)
            self.scale_shift_table.requires_grad_(False)
            

    def forward(self, x, y, t, **kwargs):
        B, T, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        # repeat for mlp
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.repeat_interleave(T, dim=0), scale_mlp.repeat_interleave(T, dim=0), gate_mlp.repeat_interleave(T, dim=0)
        
        # self-attn
        if self.attn_mode == "scene":
            # attn mask for legal object
            mask_legal = kwargs.get("mask_legal")
            attn_mask = expand_attn_mask(mask_legal.bool(), n=N).to(x) 
            # scene-level attn
            x = rearrange(x, 'b t n c -> b (t n) c')
            x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask).reshape(B, T*N, C))
        else:
            # object-level attn
            x = rearrange(x, 'b t n c -> (b t) n c')
            shift_msa, scale_msa, gate_msa = (shift_msa.repeat_interleave(T, dim=0), scale_msa.repeat_interleave(T, dim=0), gate_msa.repeat_interleave(T, dim=0))
            x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=torch.ones((B*T,N,N)).bool().to(x)).reshape(B*T, N, C))
        
        # reshape for cross-attn
        x = x.reshape(B*T, N, C)
        y = y.reshape(B*T, *y.shape[-2:])
        
        # rotation cross-attn
        x_rot = x[:, :1]
        y_rot = y[:,:self.num_img_latents+self.num_pcd_latents]
        if self.use_caption:
            y_rot = torch.cat([y_rot, y[:,-self.num_text_latents:]], dim=1)
        x_rot = x_rot + self.cross_attn(x_rot, y_rot)
        
        # mlp for size and translation
        mark = self.num_img_latents + self.num_pcd_latents
        shift_mlp_trans, scale_mlp_trans, gate_mlp_trans, shift_mlp_size, scale_mlp_size, gate_mlp_size = (self.scale_shift_table_trans[None] + t.reshape(B, 6, -1)).repeat_interleave(T, dim=0).chunk(6, dim=1)
        x_trans, x_size = x[:,1:2], x[:,2:3]
        y_trans, y_size = y[:,mark:mark+1], y[:,mark+1:mark+2]
        x_trans = x_trans + self.drop_path(gate_mlp_trans * self.mlp_trans(t2i_modulate(self.norm_trans(y_trans), shift_mlp_trans, scale_mlp_trans)))
        x_size = x_size + self.drop_path(gate_mlp_size * self.mlp_size(t2i_modulate(self.norm_size(y_size), shift_mlp_size, scale_mlp_size)))
        
        # translation cross-attn
        x_trans_size = torch.cat([x_trans, x_size], dim=1)
        if y.shape[1] > (mark + 2):
            y_trans_size = y[:,self.num_img_latents+self.num_pcd_latents:]
            x_trans_size = x_trans_size + self.cross_attn_trans(x_trans_size, y_trans_size)
        
        # concatenate the features
        x = torch.cat([x_rot, x_trans_size, x[:,3:]], dim=1)
        
        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        
        # reshape
        x = x.reshape(B, T, N, C)

        return x
        
    
class DiTBlock_pose_sep_both(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0, use_rope=False, attn_mode="scene", \
        num_img_latents=257, num_pcd_latents=512, num_pose_latents=3, num_text_latents=77, freeze_rot=False, \
        use_caption=False, use_scene_img=False, use_scene_pcd=False, use_scene_mask=False):
        
        super().__init__()
        self.norm1 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.norm2 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        
        # object-level
        self.scale_shift_table_obj = nn.Parameter(torch.randn(3, width) / width ** 0.5)
        self.norm_obj = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn_obj = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        
        # translation estimation      
        self.scale_shift_table_trans = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        self.norm_trans = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp_trans = MLP(width=width, init_scale=init_scale)
        self.norm_size = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp_size = MLP(width=width, init_scale=init_scale)
        
        self.cross_attn_trans = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        
        self.use_caption = use_caption
        self.attn_mode = attn_mode
        self.num_img_latents = num_img_latents
        self.num_pose_latents = num_pose_latents
        self.num_pcd_latents = num_pcd_latents
        self.num_text_latents = num_text_latents
        
        if freeze_rot:
            self.norm1.requires_grad_(False)
            self.attn.requires_grad_(False)
            self.cross_attn.requires_grad_(False)
            self.norm2.requires_grad_(False)
            self.mlp.requires_grad_(False)
            self.scale_shift_table.requires_grad_(False)
            

    def forward(self, x, y, t, **kwargs):
        B, T, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        # repeat for mlp
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.repeat_interleave(T, dim=0), scale_mlp.repeat_interleave(T, dim=0), gate_mlp.repeat_interleave(T, dim=0)
        
        # scene-level attn
        # attn mask for legal object
        mask_legal = kwargs.get("mask_legal")
        attn_mask = expand_attn_mask(mask_legal.bool(), n=N).to(x) 
        x = rearrange(x, 'b t n c -> b (t n) c')
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask).reshape(B, T*N, C))

        # object-level attn
        shift_msa_obj, scale_msa_obj, gate_msa_obj = (self.scale_shift_table_obj[None] + t.reshape(B, 6, -1)[:,:3]).chunk(3, dim=1)
        shift_msa_obj, scale_msa_obj, gate_msa_obj = (shift_msa_obj.repeat_interleave(T, dim=0), scale_msa_obj.repeat_interleave(T, dim=0), gate_msa_obj.repeat_interleave(T, dim=0))
        x = x.reshape(B*T, N, C)
        x = x + self.drop_path(gate_msa_obj * self.attn_obj(t2i_modulate(self.norm_obj(x), shift_msa_obj, scale_msa_obj), attn_mask=torch.ones((B*T,N,N)).bool().to(x)).reshape(B*T, N, C))
        
        # reshape for cross-attn
        x = x.reshape(B*T, N, C)
        y = y.reshape(B*T, *y.shape[-2:])
        
        # rotation cross-attn
        x_rot = x[:, :1]
        y_rot = y[:,:self.num_img_latents+self.num_pcd_latents]
        if self.use_caption:
            y_rot = torch.cat([y_rot, y[:,-self.num_text_latents:]], dim=1)
        x_rot = x_rot + self.cross_attn(x_rot, y_rot)
        
        # mlp for size and translation
        mark = self.num_img_latents + self.num_pcd_latents
        shift_mlp_trans, scale_mlp_trans, gate_mlp_trans, shift_mlp_size, scale_mlp_size, gate_mlp_size = (self.scale_shift_table_trans[None] + t.reshape(B, 6, -1)).repeat_interleave(T, dim=0).chunk(6, dim=1)
        x_trans, x_size = x[:,1:2], x[:,2:3]
        y_trans, y_size = y[:,mark:mark+1], y[:,mark+1:mark+2]
        x_trans = x_trans + self.drop_path(gate_mlp_trans * self.mlp_trans(t2i_modulate(self.norm_trans(y_trans), shift_mlp_trans, scale_mlp_trans)))
        x_size = x_size + self.drop_path(gate_mlp_size * self.mlp_size(t2i_modulate(self.norm_size(y_size), shift_mlp_size, scale_mlp_size)))
        
        # translation cross-attn
        x_trans_size = torch.cat([x_trans, x_size], dim=1)
        if y.shape[1] > (mark + 2):
            y_trans_size = y[:,self.num_img_latents+self.num_pcd_latents:]
            x_trans_size = x_trans_size + self.cross_attn_trans(x_trans_size, y_trans_size)
        
        # concatenate the features
        x = torch.cat([x_rot, x_trans_size, x[:,3:]], dim=1)
        
        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        
        # reshape
        x = x.reshape(B, T, N, C)

        return x
    
class DiTBlock_pose_sep_only_gsa(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0, use_rope=False, attn_mode="scene", \
        num_img_latents=257, num_pcd_latents=512, num_pose_latents=3, num_text_latents=77, freeze_rot=False, \
        use_caption=False, use_scene_img=False, use_scene_pcd=False, use_scene_mask=False):
        
        super().__init__()
        self.norm1 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.norm2 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        
        # translation estimation      
        self.scale_shift_table_trans = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        self.norm_trans = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp_trans = MLP(width=width, init_scale=init_scale)
        self.norm_size = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp_size = MLP(width=width, init_scale=init_scale)
        
        self.cross_attn_trans = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        
        self.use_caption = use_caption
        self.attn_mode = attn_mode
        self.num_img_latents = num_img_latents
        self.num_pose_latents = num_pose_latents
        self.num_pcd_latents = num_pcd_latents
        self.num_text_latents = num_text_latents
        
        if freeze_rot:
            self.norm1.requires_grad_(False)
            self.attn.requires_grad_(False)
            self.cross_attn.requires_grad_(False)
            self.norm2.requires_grad_(False)
            self.mlp.requires_grad_(False)
            self.scale_shift_table.requires_grad_(False)
            

    def forward(self, x, y, t, **kwargs):
        B, T, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        # repeat for mlp
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.repeat_interleave(T, dim=0), scale_mlp.repeat_interleave(T, dim=0), gate_mlp.repeat_interleave(T, dim=0)
        
        # scene-level attn
        # attn mask for legal object
        mask_legal = kwargs.get("mask_legal")
        attn_mask = expand_attn_mask(mask_legal.bool(), n=N).to(x) 
        x = rearrange(x, 'b t n c -> b (t n) c')
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask).reshape(B, T*N, C))
 
        # reshape for cross-attn
        x = x.reshape(B*T, N, C)
        y = y.reshape(B*T, *y.shape[-2:])
        
        # rotation cross-attn
        x_rot = x[:, :1]
        y_rot = y[:,:self.num_img_latents+self.num_pcd_latents]
        if self.use_caption:
            y_rot = torch.cat([y_rot, y[:,-self.num_text_latents:]], dim=1)
        x_rot = x_rot + self.cross_attn(x_rot, y_rot)
        
        # mlp for size and translation
        mark = self.num_img_latents + self.num_pcd_latents
        shift_mlp_trans, scale_mlp_trans, gate_mlp_trans, shift_mlp_size, scale_mlp_size, gate_mlp_size = (self.scale_shift_table_trans[None] + t.reshape(B, 6, -1)).repeat_interleave(T, dim=0).chunk(6, dim=1)
        x_trans, x_size = x[:,1:2], x[:,2:3]
        y_trans, y_size = y[:,mark:mark+1], y[:,mark+1:mark+2]
        x_trans = x_trans + self.drop_path(gate_mlp_trans * self.mlp_trans(t2i_modulate(self.norm_trans(y_trans), shift_mlp_trans, scale_mlp_trans)))
        x_size = x_size + self.drop_path(gate_mlp_size * self.mlp_size(t2i_modulate(self.norm_size(y_size), shift_mlp_size, scale_mlp_size)))
        
        # translation cross-attn
        x_trans_size = torch.cat([x_trans, x_size], dim=1)
        if y.shape[1] > (mark + 2):
            y_trans_size = y[:,self.num_img_latents+self.num_pcd_latents:]
            x_trans_size = x_trans_size + self.cross_attn_trans(x_trans_size, y_trans_size)
        
        # concatenate the features
        x = torch.cat([x_rot, x_trans_size, x[:,3:]], dim=1)
        
        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        
        # reshape
        x = x.reshape(B, T, N, C)

        return x
     

class DiTBlock_pose_sep_mlp_rot(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0, use_rope=False, use_cross_rope=False, attn_mode="scene", \
        num_img_latents=257, num_pcd_latents=512, num_pose_latents=3, num_text_latents=77, freeze_rot=False, \
        use_caption=False, use_scene_img=False, use_scene_pcd=False, use_scene_mask=False):
        
        super().__init__()
        self.norm1 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_cross_rope,
        )
        self.norm2 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        self.scale_shift_table_trans = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        
        # translation estimation
        self.norm_trans = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp_trans = MLP(width=width, init_scale=init_scale)
        self.norm_size = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp_size = MLP(width=width, init_scale=init_scale)
        
        self.cross_attn_trans = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_cross_rope,
        )
        
        self.use_caption = use_caption
        self.attn_mode = attn_mode
        self.num_img_latents = num_img_latents
        self.num_pose_latents = num_pose_latents
        self.num_pcd_latents = num_pcd_latents
        self.num_text_latents = num_text_latents
        
        if freeze_rot:
            self.norm1.requires_grad_(False)
            self.attn.requires_grad_(False)
            self.cross_attn.requires_grad_(False)
            self.norm2.requires_grad_(False)
            self.mlp.requires_grad_(False)
            self.scale_shift_table.requires_grad_(False)
            

    def forward(self, x, y, t, **kwargs):
        B, T, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        # repeat for mlp
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.repeat_interleave(T, dim=0), scale_mlp.repeat_interleave(T, dim=0), gate_mlp.repeat_interleave(T, dim=0)
        
        # self-attn
        if self.attn_mode == "scene":
            # attn mask for legal object
            mask_legal = kwargs.get("mask_legal")
            attn_mask = expand_attn_mask(mask_legal.bool(), n=N).to(x) 
            # scene-level attn
            x = rearrange(x, 'b t n c -> b (t n) c')
            x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask).reshape(B, T*N, C))
        else:
            # object-level attn
            x = rearrange(x, 'b t n c -> (b t) n c')
            shift_msa, scale_msa, gate_msa = (shift_msa.repeat_interleave(T, dim=0), scale_msa.repeat_interleave(T, dim=0), gate_msa.repeat_interleave(T, dim=0))
            x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=torch.ones((B*T,N,N)).bool().to(x)).reshape(B*T, N, C))
        
        # reshape for cross-attn
        x = x.reshape(B*T, N, C)
        y = y.reshape(B*T, *y.shape[-2:])
        
        # rotation cross-attn
        x_rot = x[:, :1]
        y_rot = y[:,:self.num_img_latents+self.num_pcd_latents]
        if self.use_caption:
            y_rot = torch.cat([y_rot, y[:,-self.num_text_latents:]], dim=1)
        x_rot = x_rot + self.cross_attn(x_rot, y_rot)
        
        # mlp for size and translation
        mark = self.num_img_latents + self.num_pcd_latents
        shift_mlp_trans, scale_mlp_trans, gate_mlp_trans, shift_mlp_size, scale_mlp_size, gate_mlp_size = (self.scale_shift_table_trans[None] + t.reshape(B, 6, -1)).repeat_interleave(T, dim=0).chunk(6, dim=1)
        x_trans, x_size = x[:,1:2], x[:,2:3]
        y_trans, y_size = y[:,mark:mark+1], y[:,mark+1:mark+2]
        x_trans = x_trans + self.drop_path(gate_mlp_trans * self.mlp_trans(t2i_modulate(self.norm_trans(y_trans), shift_mlp_trans, scale_mlp_trans)))
        x_size = x_size + self.drop_path(gate_mlp_size * self.mlp_size(t2i_modulate(self.norm_size(y_size), shift_mlp_size, scale_mlp_size)))
        
        # translation cross-attn
        x_rot_trans_size = torch.cat([x_rot, x_trans, x_size], dim=1)
        if y.shape[1] > (mark + 2):
            y_trans_size = y[:,self.num_img_latents+self.num_pcd_latents:]
            x_rot_trans_size = x_rot_trans_size + self.cross_attn_trans(x_rot_trans_size, y_trans_size)
        
        # concatenate the features
        x = torch.cat([x_rot_trans_size, x[:,3:]], dim=1)
        
        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        
        # reshape
        x = x.reshape(B, T, N, C)

        return x
    
    
class DiTBlock_pose_sep_both_rot(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0, use_rope=False, attn_mode="scene", \
        num_img_latents=257, num_pcd_latents=512, num_pose_latents=3, num_text_latents=77, freeze_rot=False, \
        use_caption=False, use_scene_img=False, use_scene_pcd=False, use_scene_mask=False):
        
        super().__init__()
        self.norm1 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.norm2 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        
        # object-level
        self.scale_shift_table_obj = nn.Parameter(torch.randn(3, width) / width ** 0.5)
        self.norm_obj = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn_obj = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        
        # translation estimation      
        self.scale_shift_table_trans = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        self.norm_trans = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp_trans = MLP(width=width, init_scale=init_scale)
        self.norm_size = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp_size = MLP(width=width, init_scale=init_scale)
        
        self.cross_attn_trans = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        
        self.use_caption = use_caption
        self.attn_mode = attn_mode
        self.num_img_latents = num_img_latents
        self.num_pose_latents = num_pose_latents
        self.num_pcd_latents = num_pcd_latents
        self.num_text_latents = num_text_latents
        
        if freeze_rot:
            self.norm1.requires_grad_(False)
            self.attn.requires_grad_(False)
            self.cross_attn.requires_grad_(False)
            self.norm2.requires_grad_(False)
            self.mlp.requires_grad_(False)
            self.scale_shift_table.requires_grad_(False)
            

    def forward(self, x, y, t, **kwargs):
        B, T, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        # repeat for mlp
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.repeat_interleave(T, dim=0), scale_mlp.repeat_interleave(T, dim=0), gate_mlp.repeat_interleave(T, dim=0)
        
        # scene-level attn
        # attn mask for legal object
        mask_legal = kwargs.get("mask_legal")
        attn_mask = expand_attn_mask(mask_legal.bool(), n=N).to(x) 
        x = rearrange(x, 'b t n c -> b (t n) c')
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask).reshape(B, T*N, C))

        # object-level attn
        shift_msa_obj, scale_msa_obj, gate_msa_obj = (self.scale_shift_table_obj[None] + t.reshape(B, 6, -1)[:,:3]).chunk(3, dim=1)
        shift_msa_obj, scale_msa_obj, gate_msa_obj = (shift_msa_obj.repeat_interleave(T, dim=0), scale_msa_obj.repeat_interleave(T, dim=0), gate_msa_obj.repeat_interleave(T, dim=0))
        x = x.reshape(B*T, N, C)
        x = x + self.drop_path(gate_msa_obj * self.attn_obj(t2i_modulate(self.norm_obj(x), shift_msa_obj, scale_msa_obj), attn_mask=torch.ones((B*T,N,N)).bool().to(x)).reshape(B*T, N, C))
        
        # reshape for cross-attn
        x = x.reshape(B*T, N, C)
        y = y.reshape(B*T, *y.shape[-2:])
        
        # rotation cross-attn
        x_rot = x[:, :1]
        y_rot = y[:,:self.num_img_latents+self.num_pcd_latents]
        if self.use_caption:
            y_rot = torch.cat([y_rot, y[:,-self.num_text_latents:]], dim=1)
        x_rot = x_rot + self.cross_attn(x_rot, y_rot)
        
        # mlp for size and translation
        mark = self.num_img_latents + self.num_pcd_latents
        shift_mlp_trans, scale_mlp_trans, gate_mlp_trans, shift_mlp_size, scale_mlp_size, gate_mlp_size = (self.scale_shift_table_trans[None] + t.reshape(B, 6, -1)).repeat_interleave(T, dim=0).chunk(6, dim=1)
        x_trans, x_size = x[:,1:2], x[:,2:3]
        y_trans, y_size = y[:,mark:mark+1], y[:,mark+1:mark+2]
        x_trans = x_trans + self.drop_path(gate_mlp_trans * self.mlp_trans(t2i_modulate(self.norm_trans(y_trans), shift_mlp_trans, scale_mlp_trans)))
        x_size = x_size + self.drop_path(gate_mlp_size * self.mlp_size(t2i_modulate(self.norm_size(y_size), shift_mlp_size, scale_mlp_size)))
        
        # translation cross-attn
        x_pose = torch.cat([x_rot, x_trans, x_size], dim=1)
        if y.shape[1] > (mark + 2):
            y_trans_size = y[:,self.num_img_latents+self.num_pcd_latents:]
            x_pose = x_pose + self.cross_attn_trans(x_pose, y_trans_size)
        
        # concatenate the features
        x = torch.cat([x_pose, x[:,3:]], dim=1)
        
        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        
        # reshape
        x = x.reshape(B, T, N, C)

        return x
    
class DiTBlock_pose_sep_both_only_gca(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0, use_rope=False, attn_mode="scene", \
        num_img_latents=257, num_pcd_latents=512, num_pose_latents=3, num_text_latents=77, freeze_rot=False, \
        use_caption=False, use_scene_img=False, use_scene_pcd=False, use_scene_mask=False):
        
        super().__init__()
        self.norm1 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.norm2 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        
        # object-level
        self.scale_shift_table_obj = nn.Parameter(torch.randn(3, width) / width ** 0.5)
        self.norm_obj = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn_obj = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        
        # translation estimation      
        self.scale_shift_table_trans = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        self.norm_trans = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp_trans = MLP(width=width, init_scale=init_scale)
        self.norm_size = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp_size = MLP(width=width, init_scale=init_scale)
        
        self.cross_attn_trans = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        
        self.use_caption = use_caption
        self.attn_mode = attn_mode
        self.num_img_latents = num_img_latents
        self.num_pose_latents = num_pose_latents
        self.num_pcd_latents = num_pcd_latents
        self.num_text_latents = num_text_latents
        
        if freeze_rot:
            self.norm1.requires_grad_(False)
            self.attn.requires_grad_(False)
            self.norm2.requires_grad_(False)
            self.mlp.requires_grad_(False)
            self.scale_shift_table.requires_grad_(False)
            

    def forward(self, x, y, t, **kwargs):
        B, T, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        # repeat for mlp
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.repeat_interleave(T, dim=0), scale_mlp.repeat_interleave(T, dim=0), gate_mlp.repeat_interleave(T, dim=0)
        
        # scene-level attn
        # attn mask for legal object
        mask_legal = kwargs.get("mask_legal")
        attn_mask = expand_attn_mask(mask_legal.bool(), n=N).to(x) 
        x = rearrange(x, 'b t n c -> b (t n) c')
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask).reshape(B, T*N, C))

        # object-level attn
        shift_msa_obj, scale_msa_obj, gate_msa_obj = (self.scale_shift_table_obj[None] + t.reshape(B, 6, -1)[:,:3]).chunk(3, dim=1)
        shift_msa_obj, scale_msa_obj, gate_msa_obj = (shift_msa_obj.repeat_interleave(T, dim=0), scale_msa_obj.repeat_interleave(T, dim=0), gate_msa_obj.repeat_interleave(T, dim=0))
        x = x.reshape(B*T, N, C)
        x = x + self.drop_path(gate_msa_obj * self.attn_obj(t2i_modulate(self.norm_obj(x), shift_msa_obj, scale_msa_obj), attn_mask=torch.ones((B*T,N,N)).bool().to(x)).reshape(B*T, N, C))
        
        # reshape for cross-attn
        x = x.reshape(B*T, N, C)
        y = y.reshape(B*T, *y.shape[-2:])        
        
        # mlp for size and translation
        mark = self.num_img_latents + self.num_pcd_latents
        shift_mlp_trans, scale_mlp_trans, gate_mlp_trans, shift_mlp_size, scale_mlp_size, gate_mlp_size = (self.scale_shift_table_trans[None] + t.reshape(B, 6, -1)).repeat_interleave(T, dim=0).chunk(6, dim=1)
        x_trans, x_size = x[:,1:2], x[:,2:3]
        y_trans, y_size = y[:,mark:mark+1], y[:,mark+1:mark+2]
        x_trans = x_trans + self.drop_path(gate_mlp_trans * self.mlp_trans(t2i_modulate(self.norm_trans(y_trans), shift_mlp_trans, scale_mlp_trans)))
        x_size = x_size + self.drop_path(gate_mlp_size * self.mlp_size(t2i_modulate(self.norm_size(y_size), shift_mlp_size, scale_mlp_size)))
        
        # translation cross-attn
        x_rot = x[:, :1]
        x_pose = torch.cat([x_rot, x_trans, x_size], dim=1)
        if y.shape[1] > (mark + 2):
            y_trans_size = y[:,self.num_img_latents+self.num_pcd_latents:]
            x_pose = x_pose + self.cross_attn_trans(x_pose, y_trans_size)
        
        # concatenate the features
        x = torch.cat([x_pose, x[:,3:]], dim=1)
        
        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        
        # reshape
        x = x.reshape(B, T, N, C)

        return x
 
    
class DiTBlock_pose_sep_both_scene(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0, use_rope=False, use_cross_rope=False, attn_mode="scene", \
        num_img_latents=257, num_pcd_latents=512, num_pose_latents=3, num_text_latents=77, num_shape_latetns=512, freeze_rot=False, \
        use_caption=False, use_scene_img=False, use_scene_pcd=False, use_scene_mask=False):
        
        super().__init__()
        self.norm1 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_cross_rope,
        )
        self.norm2 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        
        # rotation cross-attn
        self.cross_attn_rot = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        
        # scene-level attn
        self.scale_shift_table_scene = nn.Parameter(torch.randn(3, width) / width ** 0.5)
        self.norm_scene = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn_scene = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        
        # translation estimation      
        self.scale_shift_table_trans = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        self.norm_trans = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp_trans = MLP(width=width, init_scale=init_scale)
        self.norm_size = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp_size = MLP(width=width, init_scale=init_scale)
        
        
        self.use_caption = use_caption
        self.attn_mode = attn_mode
        self.num_img_latents = num_img_latents
        self.num_pose_latents = num_pose_latents
        self.num_pcd_latents = num_pcd_latents
        self.num_text_latents = num_text_latents
        self.num_shape_latetns = num_shape_latetns
        
        if freeze_rot:
            self.norm1.requires_grad_(False)
            self.attn.requires_grad_(False)
            self.cross_attn.requires_grad_(False)
            self.norm2.requires_grad_(False)
            self.mlp.requires_grad_(False)
            self.scale_shift_table.requires_grad_(False)
            

    def forward(self, x, y, t, **kwargs):
        B, T, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        # repeat for mlp
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.repeat_interleave(T, dim=0), scale_mlp.repeat_interleave(T, dim=0), gate_mlp.repeat_interleave(T, dim=0)
        
        # scene-level attn
        mask_legal = kwargs.get("mask_legal")
        attn_mask = expand_attn_mask(mask_legal.bool(), n=N).to(x) 
        x = rearrange(x, 'b t n c -> b (t n) c')
        shift_msa_scene, scale_msa_scene, gate_msa_scene = (self.scale_shift_table_scene[None] + t.reshape(B, 6, -1)[:,:3]).chunk(3, dim=1)
        x = x + self.drop_path(gate_msa_scene * self.attn_scene(t2i_modulate(self.norm_scene(x), shift_msa_scene, scale_msa_scene), attn_mask=attn_mask).reshape(B, T*N, C))

        # object-level attn
        x = x.reshape(B*T, N, C)
        shift_msa, scale_msa, gate_msa = (shift_msa.repeat_interleave(T, dim=0), scale_msa.repeat_interleave(T, dim=0), gate_msa.repeat_interleave(T, dim=0))
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=torch.ones((B*T,N,N)).bool().to(x)).reshape(B*T, N, C))

        # reshape for cross-attn
        x = x.reshape(B*T, N, C)
        y = y.reshape(B*T, *y.shape[-2:])
        
        # rotation cross-attn
        x_rot = x[:, :1]
        y_rot = y[:,:self.num_img_latents+self.num_pcd_latents]
        if self.use_caption:
            y_rot = torch.cat([y_rot, y[:,-self.num_text_latents:]], dim=1)
        x_rot = x_rot + self.cross_attn_rot(x_rot, y_rot)
        
        # mlp for size and translation
        mark = self.num_img_latents + self.num_pcd_latents
        shift_mlp_trans, scale_mlp_trans, gate_mlp_trans, shift_mlp_size, scale_mlp_size, gate_mlp_size = (self.scale_shift_table_trans[None] + t.reshape(B, 6, -1)).repeat_interleave(T, dim=0).chunk(6, dim=1)
        x_trans, x_size = x[:,1:2], x[:,2:3]
        y_trans, y_size = y[:,mark:mark+1], y[:,mark+1:mark+2]
        x_trans = x_trans + self.drop_path(gate_mlp_trans * self.mlp_trans(t2i_modulate(self.norm_trans(y_trans), shift_mlp_trans, scale_mlp_trans)))
        x_size = x_size + self.drop_path(gate_mlp_size * self.mlp_size(t2i_modulate(self.norm_size(y_size), shift_mlp_size, scale_mlp_size)))
        
        # cross-attn for scene
        x_scene = torch.cat([x_rot, x_trans, x_size, x[:,3:3+self.num_shape_latetns]], dim=1)
        if y.shape[1] > (mark + 2):
            y_scene = y[:,mark+2:]
            x_scene = x_scene + self.cross_attn(x_scene, y_scene)
        
        # concatenate the features
        x = torch.cat([x_scene, x[:,3+self.num_shape_latetns:]], dim=1)
        
        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        
        # reshape
        x = x.reshape(B, T, N, C)

        return x
   
    
class DiTBlock_pose_trans(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0, use_rope=False, use_cross_rope=False, attn_mode="scene", \
        num_img_latents=257, num_pcd_latents=512, num_pose_latents=3, num_text_latents=77, num_shape_latetns=512, freeze_rot=False, \
        use_caption=False, use_scene_img=False, use_scene_pcd=False, use_scene_mask=False):
        
        super().__init__()
        self.norm1 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_cross_rope,
        )
        self.norm2 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        
        # # object-level
        # self.scale_shift_table_obj = nn.Parameter(torch.randn(3, width) / width ** 0.5)
        # self.norm_obj = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        # self.attn_obj = MultiheadAttention_pose(
        #     n_ctx=None,
        #     width=width,
        #     heads=heads,
        #     init_scale=init_scale,
        #     qkv_bias=qkv_bias,
        #     use_flash=use_flash,
        #     use_rope=use_rope,
        # )
        
        # # translation estimation      
        # self.scale_shift_table_trans = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        # self.norm_trans = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        # self.mlp_trans = MLP(width=width, init_scale=init_scale)
        # self.norm_size = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        # self.mlp_size = MLP(width=width, init_scale=init_scale)
        
        # self.cross_attn_trans = MultiheadCrossAttention(
        #     n_data=None,
        #     width=width,
        #     heads=heads,
        #     data_width=None,
        #     init_scale=init_scale,
        #     qkv_bias=qkv_bias,
        #     use_flash=use_flash,
        #     use_rope=use_rope,
        # )
        
        self.use_caption = use_caption
        self.attn_mode = attn_mode
        self.num_img_latents = num_img_latents
        self.num_pose_latents = num_pose_latents
        self.num_pcd_latents = num_pcd_latents
        self.num_text_latents = num_text_latents
        self.num_shape_latetns = num_shape_latetns
        
        if freeze_rot:
            self.norm1.requires_grad_(False)
            self.attn.requires_grad_(False)
            self.cross_attn.requires_grad_(False)
            self.norm2.requires_grad_(False)
            self.mlp.requires_grad_(False)
            self.scale_shift_table.requires_grad_(False)
            

    def forward(self, x, y, t, **kwargs):
        B, T, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        # repeat for mlp
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.repeat_interleave(T, dim=0), scale_mlp.repeat_interleave(T, dim=0), gate_mlp.repeat_interleave(T, dim=0)
        
        # # scene-level attn
        # mask_legal = kwargs.get("mask_legal")
        # attn_mask = expand_attn_mask(mask_legal.bool(), n=N).to(x) 
        
        # object-level attn
        x = x.reshape(B*T, N, C)
        shift_msa, scale_msa, gate_msa = (shift_msa.repeat_interleave(T, dim=0), scale_msa.repeat_interleave(T, dim=0), gate_msa.repeat_interleave(T, dim=0))
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=torch.ones((B*T,N,N)).bool().to(x)).reshape(B*T, N, C))

        # reshape for cross-attn
        x = x.reshape(B*T, N, C)
        y = y.reshape(B*T, *y.shape[-2:])
        
        # # rotation cross-attn
        # x_rot = x[:, :1]
        # y_rot = y[:,:self.num_img_latents+self.num_pcd_latents]
        # if self.use_caption:
        #     y_rot = torch.cat([y_rot, y[:,-self.num_text_latents:]], dim=1)
        # x_rot = x_rot + self.cross_attn(x_rot, y_rot)
        
        # # mlp for size and translation
        mark = self.num_img_latents + self.num_pcd_latents
        x_scene = x[:,3:3+self.num_shape_latetns]
        if y.shape[1] > (mark + 2):
            y_scene = y[:,mark+2:]
            x_scene = x_scene + self.cross_attn(x_scene, y_scene)
        
        # concatenate the features
        # x = torch.cat([x_scene, x[:,3+self.num_shape_latetns:]], dim=1)
        x = torch.cat([x[:, :3], x_scene, x[:,3+self.num_shape_latetns:]], dim=1)
        
        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        
        # reshape
        x = x.reshape(B, T, N, C)

        return x
   
    
class DiTBlock_pose_trans2(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, width, heads, init_scale=1.0, qkv_bias=True, use_flash=True, drop_path=0.0, use_rope=False, use_cross_rope=False, attn_mode="scene", \
        num_img_latents=257, num_pcd_latents=512, num_pose_latents=3, num_text_latents=77, num_shape_latetns=512, freeze_rot=False, \
        use_caption=False, use_scene_img=False, use_scene_pcd=False, use_scene_mask=False):
        
        super().__init__()
        self.norm1 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        self.cross_attn = MultiheadCrossAttention(
            n_data=None,
            width=width,
            heads=heads,
            data_width=None,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_cross_rope,
        )
        self.norm2 = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)

        self.mlp = MLP(width=width, init_scale=init_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        
        # scene-level attn
        self.scale_shift_table_scene = nn.Parameter(torch.randn(3, width) / width ** 0.5)
        self.norm_scene = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        self.attn_scene = MultiheadAttention_pose(
            n_ctx=None,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_rope=use_rope,
        )
        
        # # translation estimation      
        # self.scale_shift_table_trans = nn.Parameter(torch.randn(6, width) / width ** 0.5)
        # self.norm_trans = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        # self.mlp_trans = MLP(width=width, init_scale=init_scale)
        # self.norm_size = nn.RMSNorm(width, elementwise_affine=True, eps=1e-6)
        # self.mlp_size = MLP(width=width, init_scale=init_scale)
        
        # self.cross_attn_trans = MultiheadCrossAttention(
        #     n_data=None,
        #     width=width,
        #     heads=heads,
        #     data_width=None,
        #     init_scale=init_scale,
        #     qkv_bias=qkv_bias,
        #     use_flash=use_flash,
        #     use_rope=use_rope,
        # )
        
        self.use_caption = use_caption
        self.attn_mode = attn_mode
        self.num_img_latents = num_img_latents
        self.num_pose_latents = num_pose_latents
        self.num_pcd_latents = num_pcd_latents
        self.num_text_latents = num_text_latents
        self.num_shape_latetns = num_shape_latetns
        
        if freeze_rot:
            self.norm1.requires_grad_(False)
            self.attn.requires_grad_(False)
            self.cross_attn.requires_grad_(False)
            self.norm2.requires_grad_(False)
            self.mlp.requires_grad_(False)
            self.scale_shift_table.requires_grad_(False)
            

    def forward(self, x, y, t, **kwargs):
        B, T, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        # repeat for mlp
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.repeat_interleave(T, dim=0), scale_mlp.repeat_interleave(T, dim=0), gate_mlp.repeat_interleave(T, dim=0)
        
        # scene-level attn
        mask_legal = kwargs.get("mask_legal")
        attn_mask = expand_attn_mask(mask_legal.bool(), n=N).to(x) 
        x = rearrange(x, 'b t n c -> b (t n) c')
        shift_msa_scene, scale_msa_scene, gate_msa_scene = (self.scale_shift_table_scene[None] + t.reshape(B, 6, -1)[:,:3]).chunk(3, dim=1)
        x = x + self.drop_path(gate_msa_scene * self.attn_scene(t2i_modulate(self.norm_scene(x), shift_msa_scene, scale_msa_scene), attn_mask=attn_mask).reshape(B, T*N, C))

        # object-level attn
        x = x.reshape(B*T, N, C)
        shift_msa, scale_msa, gate_msa = (shift_msa.repeat_interleave(T, dim=0), scale_msa.repeat_interleave(T, dim=0), gate_msa.repeat_interleave(T, dim=0))
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=torch.ones((B*T,N,N)).bool().to(x)).reshape(B*T, N, C))

        # reshape for cross-attn
        x = x.reshape(B*T, N, C)
        y = y.reshape(B*T, *y.shape[-2:])
        
        # # rotation cross-attn
        # x_rot = x[:, :1]
        # y_rot = y[:,:self.num_img_latents+self.num_pcd_latents]
        # if self.use_caption:
        #     y_rot = torch.cat([y_rot, y[:,-self.num_text_latents:]], dim=1)
        # x_rot = x_rot + self.cross_attn(x_rot, y_rot)
        
        # # mlp for size and translation
        mark = self.num_img_latents + self.num_pcd_latents
        x_scene = x[:,3:3+self.num_shape_latetns]
        if y.shape[1] > (mark + 2):
            y_scene = y[:,mark+2:]
            x_scene = x_scene + self.cross_attn(x_scene, y_scene)
        
        # concatenate the features
        # x = torch.cat([x_scene, x[:,3+self.num_shape_latetns:]], dim=1)
        x = torch.cat([x[:, :3], x_scene, x[:,3+self.num_shape_latetns:]], dim=1)
        
        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        
        # reshape
        x = x.reshape(B, T, N, C)

        return x
    
    
def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift

def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, 'grad_checkpointing', False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, **kwargs)
    return module(*args, **kwargs)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype
    

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x