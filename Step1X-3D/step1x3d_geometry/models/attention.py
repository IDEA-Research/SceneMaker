# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional, Tuple, Union
import collections.abc
from itertools import repeat

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
    FP32LayerNorm,
    LayerNorm,
)

from .attention_processor import FluxAttnProcessor2_0, AttnProcessor2_0


@maybe_allow_in_graph
class MultiCondBasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        use_self_attention: bool = True,
        use_cross_attention: bool = False,
        self_attention_norm_type: Optional[
            str
        ] = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        cross_attention_dim: Optional[int] = None,
        cross_attention_norm_type: Optional[str] = None,
        # parallel second cross attention
        use_cross_attention_2: bool = False,
        cross_attention_2_dim: Optional[int] = None,
        cross_attention_2_norm_type: Optional[str] = None,
        # parallel third cross attention
        use_cross_attention_3: bool = False,
        cross_attention_3_dim: Optional[int] = None,
        cross_attention_3_norm_type: Optional[str] = None,
        dropout=0.0,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self.self_attention_norm_type = self_attention_norm_type
        self.cross_attention_dim = cross_attention_dim
        self.cross_attention_norm_type = cross_attention_norm_type
        self.use_cross_attention_2 = use_cross_attention_2
        self.cross_attention_2_dim = cross_attention_2_dim
        self.cross_attention_2_norm_type = cross_attention_2_norm_type
        self.use_cross_attention_3 = use_cross_attention_3
        self.cross_attention_3_dim = cross_attention_3_dim
        self.cross_attention_3_norm_type = cross_attention_3_norm_type
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.double_self_attention = double_self_attention
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.only_cross_attention = only_cross_attention

        # We keep these boolean flags for backward-compatibility.
        self.use_ada_layer_norm_zero = (
            num_embeds_ada_norm is not None
        ) and self_attention_norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (
            num_embeds_ada_norm is not None
        ) and self_attention_norm_type == "ada_norm"
        self.use_ada_layer_norm_single = self_attention_norm_type == "ada_norm_single"
        self.use_layer_norm = self_attention_norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = (
            self_attention_norm_type == "ada_norm_continuous"
        )

        if (
            self_attention_norm_type in ("ada_norm", "ada_norm_zero")
            and num_embeds_ada_norm is None
        ):
            raise ValueError(
                f"`self_attention_norm_type` is set to {self_attention_norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `self_attention_norm_type` to {self_attention_norm_type}."
            )

        self.self_attention_norm_type = self_attention_norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        if use_self_attention:
            # 1. Self-Attn
            if self_attention_norm_type == "ada_norm":
                self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif self_attention_norm_type == "ada_norm_zero":
                self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
            elif self_attention_norm_type == "ada_norm_continuous":
                self.norm1 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            elif (
                self_attention_norm_type == "fp32_layer_norm"
                or self_attention_norm_type is None
            ):
                self.norm1 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)
            else:
                self.norm1 = nn.RMSNorm(
                    dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
                )

            self.attn1 = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=dim // num_attention_heads,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=(
                    cross_attention_dim if only_cross_attention else None
                ),
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                processor=AttnProcessor2_0(),
            )

        # 2. Cross-Attn
        if use_cross_attention or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if cross_attention_norm_type == "ada_norm":
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif cross_attention_norm_type == "ada_norm_continuous":
                self.norm2 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            elif (
                cross_attention_norm_type == "fp32_layer_norm"
                or cross_attention_norm_type is None
            ):
                self.norm2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)
            else:
                self.norm2 = nn.RMSNorm(
                    dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
                )

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=(
                    cross_attention_dim if not double_self_attention else None
                ),
                heads=num_attention_heads,
                dim_head=dim // num_attention_heads,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                processor=AttnProcessor2_0(),
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 2'. Parallel Second Cross-Attn
        if use_cross_attention_2:
            assert cross_attention_2_dim is not None
            if cross_attention_2_norm_type == "ada_norm":
                self.norm2_2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif cross_attention_2_norm_type == "ada_norm_continuous":
                self.norm2_2 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            elif (
                cross_attention_2_norm_type == "fp32_layer_norm"
                or cross_attention_2_norm_type is None
            ):
                self.norm2_2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)
            else:
                self.norm2_2 = nn.RMSNorm(
                    dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
                )

            self.attn2_2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_2_dim,
                heads=num_attention_heads,
                dim_head=dim // num_attention_heads,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                processor=AttnProcessor2_0(),
            )

            # self.attn2_2 = Attention(
            #     query_dim=dim,
            #     cross_attention_dim=cross_attention_2_dim,
            #     dim_head=dim // num_attention_heads,
            #     heads=num_attention_heads,
            #     qk_norm="rms_norm" if qk_norm else None,
            #     cross_attention_norm=cross_attention_2_norm_type,
            #     eps=1e-6,
            #     bias=qkv_bias,
            #     processor=AttnProcessor2_0(),
            # )
        else:
            self.norm2_2 = None
            self.attn2_2 = None

        # 2'. Parallel Third Cross-Attn
        if use_cross_attention_3:
            assert cross_attention_3_dim is not None
            if cross_attention_3_norm_type == "ada_norm":
                self.norm2_3 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif cross_attention_3_norm_type == "ada_norm_continuous":
                self.norm2_3 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            elif (
                cross_attention_3_norm_type == "fp32_layer_norm"
                or cross_attention_3_norm_type is None
            ):
                self.norm2_3 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)
            else:
                self.norm2_3 = nn.RMSNorm(
                    dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
                )

            self.attn2_3 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_3_dim,
                heads=num_attention_heads,
                dim_head=dim // num_attention_heads,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                processor=AttnProcessor2_0(),
            )
        else:
            self.norm2_3 = None
            self.attn2_3 = None

        # 3. Feed-forward
        if self_attention_norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )

        elif self_attention_norm_type in ["ada_norm_zero", "ada_norm", "layer_norm"]:
            self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        elif self_attention_norm_type == "layer_norm_i2vgen":
            self.norm3 = None

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(
                dim, cross_attention_dim, num_attention_heads, attention_head_dim
            )

        # 5. Scale-shift for PixArt-Alpha.
        if self_attention_norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        encoder_hidden_states_3: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask_2: Optional[torch.Tensor] = None,
        encoder_attention_mask_3: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored."
                )

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.self_attention_norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.self_attention_norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.self_attention_norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.self_attention_norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif self.self_attention_norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        if self.self_attention_norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.self_attention_norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.cross_attention_norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.cross_attention_norm_type in [
                "ada_norm_zero",
                "layer_norm",
                "layer_norm_i2vgen",
            ]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.cross_attention_norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.cross_attention_norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(
                    hidden_states, added_cond_kwargs["pooled_text_emb"]
                )
            else:
                raise ValueError("Incorrect norm")

            if (
                self.pos_embed is not None
                and self.cross_attention_norm_type != "ada_norm_single"
            ):
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3.1 Parallel Second Cross-Attention
        if self.attn2_2 is not None:
            if self.cross_attention_2_norm_type == "ada_norm":
                norm_hidden_states = self.norm2_2(hidden_states, timestep)
            elif self.cross_attention_2_norm_type in [
                "ada_norm_zero",
                "layer_norm",
                "layer_norm_i2vgen",
            ]:
                norm_hidden_states = self.norm2_2(hidden_states)
            elif self.cross_attention_2_norm_type == "ada_norm_single":
                # For PixArt norm2_2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.cross_attention_2_norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2_2(
                    hidden_states, added_cond_kwargs["pooled_text_emb"]
                )
            else:
                raise ValueError("Incorrect norm")

            if (
                self.pos_embed is not None
                and self.cross_attention_2_norm_type != "ada_norm_single"
            ):
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output_2 = self.attn2_2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states_2,
                attention_mask=encoder_attention_mask_2,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output_2 + hidden_states

        # 3.2 Parallel Third Cross-Attention
        if self.attn2_3 is not None:
            if self.cross_attention_3_norm_type == "ada_norm":
                norm_hidden_states = self.norm2_3(hidden_states, timestep)
            elif self.cross_attention_3_norm_type in [
                "ada_norm_zero",
                "layer_norm",
                "layer_norm_i2vgen",
            ]:
                norm_hidden_states = self.norm2_3(hidden_states)
            elif self.cross_attention_3_norm_type == "ada_norm_single":
                # For PixArt norm2_3 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.cross_attention_3_norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2_3(
                    hidden_states, added_cond_kwargs["pooled_text_emb"]
                )
            else:
                raise ValueError("Incorrect norm")

            if (
                self.pos_embed is not None
                and self.cross_attention_3_norm_type != "ada_norm_single"
            ):
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output_3 = self.attn2_3(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states_3,
                attention_mask=encoder_attention_mask_3,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output_3 + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.self_attention_norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif not self.self_attention_norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.self_attention_norm_type == "ada_norm_zero":
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

        if self.self_attention_norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.self_attention_norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.self_attention_norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        if is_torch_npu_available():
            deprecation_message = (
                "Defaulting to FluxAttnProcessor2_0_NPU for NPU devices will be removed. Attention processors "
                "should be set explicitly using the `set_attn_processor` method."
            )
            deprecate("npu_processor", "0.34.0", deprecation_message)
            processor = FluxAttnProcessor2_0_NPU()
        else:
            processor = FluxAttnProcessor2_0()

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)

        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )

        mlp_ratio = 4.0
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )
        joint_attention_kwargs = joint_attention_kwargs or {}
        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
            + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        )
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states
