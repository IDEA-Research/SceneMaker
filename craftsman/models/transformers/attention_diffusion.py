import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from craftsman.utils.typing import *
from craftsman.utils.checkpoint import checkpoint

from .utils import init_linear, MLP
from timm.models.vision_transformer import Attention

import pdb

# def apply_rope_qk(q, k, seq_len, dim_head):
#     """Apply Rotary Position Embedding (RoPE) to q and k."""
#     # Generate sinusoidal embeddings
#     half_dim = dim_head // 2
#     dtype = q.dtype
#     theta = torch.arange(half_dim, device=q.device)
#     theta = 10000 ** (-2 * (theta / half_dim))
#     seq_idx = torch.arange(seq_len, device=q.device)
#     sinusoid = torch.einsum("i,j->ij", seq_idx, theta)
#     sin, cos = sinusoid.sin(), sinusoid.cos()

#     # Reshape for broadcasting
#     sin, cos = sin.unsqueeze(0).unsqueeze(-2), cos.unsqueeze(0).unsqueeze(-2)

#     # Apply RoPE to q and k
#     q1, q2 = q[..., ::2], q[..., 1::2]
#     k1, k2 = k[..., ::2], k[..., 1::2]
#     q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).to(dtype)
#     k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).to(dtype)

#     return q, k

# def apply_rope_single(q, seq_len, dim_head):
#     """Apply Rotary Position Embedding (RoPE) to q and k."""
#     # Generate sinusoidal embeddings
#     half_dim = dim_head // 2
#     dtype = q.dtype
#     theta = torch.arange(half_dim, device=q.device)
#     theta = 10000 ** (-2 * (theta / half_dim))
#     seq_idx = torch.arange(seq_len, device=q.device)
#     sinusoid = torch.einsum("i,j->ij", seq_idx, theta)
#     sin, cos = sinusoid.sin(), sinusoid.cos()

#     # Reshape for broadcasting
#     sin, cos = sin.unsqueeze(0).unsqueeze(-2), cos.unsqueeze(0).unsqueeze(-2)

#     # Apply RoPE to q and k
#     q1, q2 = q[..., ::2], q[..., 1::2]
#     q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).to(dtype)

#     return q

def apply_rope_qk(q, k, seq_len, dim_head):
    half_dim = dim_head // 2
    dtype = q.dtype
    theta = torch.arange(half_dim, device=q.device)
    theta = 10000 ** (-2 * (theta / half_dim))
    seq_idx = torch.arange(seq_len, device=q.device)
    sinusoid = torch.einsum("i,j->ij", seq_idx, theta)
    sin, cos = sinusoid.sin(), sinusoid.cos()
    sin = sin[None, :, None, :]  # [1, seq_len, 1, half_dim]
    cos = cos[None, :, None, :]

    def rope(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot.flatten(-2)

    q = rope(q)
    k = rope(k)
    return q.to(dtype), k.to(dtype)

def apply_rope_single(q, seq_len, dim_head):
    """Apply Rotary Position Embedding (RoPE) to q."""
    half_dim = dim_head // 2
    dtype = q.dtype
    theta = torch.arange(half_dim, device=q.device)
    theta = 10000 ** (-2 * (theta / half_dim))
    seq_idx = torch.arange(seq_len, device=q.device)
    sinusoid = torch.einsum("i,j->ij", seq_idx, theta)
    sin, cos = sinusoid.sin(), sinusoid.cos()
    sin = sin[None, :, None, :]  # [1, seq_len, 1, half_dim]
    cos = cos[None, :, None, :]

    q1, q2 = q[..., ::2], q[..., 1::2]
    q_rot = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    q = q_rot.flatten(-2)

    return q.to(dtype)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool,
        use_flash: bool = False,
        use_rope: bool = False
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_k = nn.Linear(width, width, bias=qkv_bias)
        self.c_v = nn.Linear(width, width, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads=heads, n_ctx=n_ctx, use_flash=use_flash, use_rope=use_rope)
        init_linear(self.c_q, init_scale)
        init_linear(self.c_k, init_scale)
        init_linear(self.c_v, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        x = checkpoint(self.attention, (q,k,v), (), True)
        x = self.c_proj(x)
        return x


class MultiheadAttention_pose(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool,
        use_flash: bool = False,
        use_rope: bool = False
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_k = nn.Linear(width, width, bias=qkv_bias)
        self.c_v = nn.Linear(width, width, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads=heads, n_ctx=n_ctx, use_flash=use_flash, use_rope=use_rope)
        init_linear(self.c_q, init_scale)
        init_linear(self.c_k, init_scale)
        init_linear(self.c_v, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x, attn_mask=None):
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        x = checkpoint(self.attention, (q,k,v,attn_mask), (), True)
        x = self.c_proj(x)
        return x

class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, heads: int, n_ctx: int, use_flash: bool = False, use_rope: bool = False):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.use_flash = use_flash
        self.use_rope = use_rope  # Add a flag for RoPE

    def forward(self, q, k, v, attn_mask=None):
        bs, n_ctx, width = q.shape
        attn_ch = width // self.heads
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1)
        k = k.view(bs, n_ctx, self.heads, -1)
        v = v.view(bs, n_ctx, self.heads, -1)

        # Apply RoPE if enabled
        if self.use_rope:
            q, k = apply_rope_qk(q, k, n_ctx, attn_ch)

        if self.use_flash:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(1).repeat(1,self.heads,1,1)
                attn_mask = attn_mask.bool()
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            if attn_mask is not None:
                attn_mask = attn_mask.bool()
                attn_mask = attn_mask.unsqueeze(1).repeat(1,self.heads,1,1)
                attn_bias = torch.zeros(n_ctx, n_ctx, dtype=q.dtype, device=q.device)
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                weight = weight + attn_bias
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out

class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
        qkv_bias: bool = True,
        use_flash: bool = False,
        use_checkpoint: bool = False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash
        )
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width)

    def _forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool = True,
        use_flash: bool = False,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
        use_rope: bool = False,
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_k = nn.Linear(width, width, bias=qkv_bias)
        self.c_v = nn.Linear(width, width, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(
            heads=heads, n_data=n_data, use_flash=use_flash, use_rope=use_rope
        )
        init_linear(self.c_q, init_scale)
        init_linear(self.c_k, init_scale)
        init_linear(self.c_v, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x, data):
        q = self.c_q(x)
        k = self.c_k(data)
        v = self.c_v(data)
        x = checkpoint(self.attention, (q, k, v,), (), True)
        x = self.c_proj(x)
        return x
    

class MultiheadCrossAttention_split(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool = True,
        use_flash: bool = False,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
        use_rope: bool = False,
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_k = nn.Linear(width, width, bias=qkv_bias)
        self.c_v = nn.Linear(width, width, bias=qkv_bias)
        self.c_k_pcd = nn.Linear(width, width, bias=qkv_bias)
        self.c_v_pcd = nn.Linear(width, width, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(
            heads=heads, n_data=n_data, use_flash=use_flash, use_rope=use_rope
        )
        init_linear(self.c_q, init_scale)
        init_linear(self.c_k, init_scale)
        init_linear(self.c_v, init_scale)
        init_linear(self.c_k_pcd, init_scale)
        init_linear(self.c_v_pcd, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x, data):
        q = self.c_q(x)
        k = torch.cat([self.c_k_pcd(data[:,2:]), self.c_k(data[:,:2])], dim=1)
        v = torch.cat([self.c_v_pcd(data[:,2:]), self.c_v(data[:,:2])], dim=1)
        x = checkpoint(self.attention, (q, k, v,), (), True)
        x = self.c_proj(x)
        return x


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, *, heads: int, use_flash: bool = False, use_rope: bool = False, n_data: Optional[int] = None):
        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.use_flash = use_flash
        self.use_rope = use_rope  # Add a flag for RoPE

    def forward(self, q, k, v):
        _, n_ctx, _ = q.shape
        bs, n_data, width = k.shape
        attn_ch = width // self.heads
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1)
        k = k.view(bs, n_data, self.heads, -1)
        v = v.view(bs, n_data, self.heads, -1)

        # Apply RoPE if enabled
        if self.use_rope:
            q = apply_rope_single(q, n_ctx, attn_ch).to(v)
            k = apply_rope_single(k, n_data, attn_ch).to(v)

        if self.use_flash:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)
            # return out, None  # Flash attention does not return weights
        
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)
            # return out, weight
        
        return out

class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        data_width: Optional[int] = None,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        use_flash: bool = False
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
        )
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(data_width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_3 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x