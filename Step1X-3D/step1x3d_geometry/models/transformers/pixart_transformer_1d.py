# Some parts of this file are adapted from Hugging Face Diffusers library.
from dataclasses import dataclass

import re
import math
import torch
from torch import nn
from typing import Callable, List, Optional, Union, Dict, Any
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import logging
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle

from ..attention_processor import FusedAttnProcessor2_0, AttnProcessor2_0
from ..attention import MultiCondBasicTransformerBlock

import step1x3d_geometry
from step1x3d_geometry.utils.base import BaseModule

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class Transformer1DModelOutput:
    sample: torch.FloatTensor


class PixArtTransformer1DModel(ModelMixin, ConfigMixin):
    r"""
    A 1D Transformer model as introduced in PixArt family of models (https://arxiv.org/abs/2310.00426,
    https://arxiv.org/abs/2403.04692).

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16):
            The number of heads to use for multi-head attention.
        width (`int`, *optional*, defaults to 2048):
            Maximum sequence length in latent space (equivalent to max_seq_length in Transformers).
            Determines the first dimension size of positional embedding matrices[1](@ref).
        in_channels (`int`, *optional*, defaults to 64):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1):
            The number of layers of Transformer blocks to use.
        cross_attention_dim (`int`, *optional*):
            Dimensionality of conditional embeddings for cross-attention mechanisms
        use_cross_attention_2 (`bool`, *optional*):
            Flag to enable secondary cross-attention mechanism. Used for multi-modal conditioning
            when processing hybrid inputs (e.g., text + image prompts)[1](@ref).
        cross_attention_2_dim (`int`, *optional*, defaults to 1024):
            Dimensionality of secondary cross-attention embeddings. Specifies encoding dimensions
            for additional conditional modalities when use_cross_attention_2 is enabled[1](@ref).
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MultiCondBasicTransformerBlock", "PatchEmbed"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm", "adaln_single"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        width: int = 2048,
        in_channels: int = 4,
        num_layers: int = 28,
        cross_attention_dim: int = 768,
        use_cross_attention_2: bool = True,
        cross_attention_2_dim: int = 1024,
        use_cross_attention_3: bool = True,
        cross_attention_3_dim: int = 1024,
    ):
        super().__init__()
        # Set some common variables used across the board.
        self.out_channels = in_channels
        self.num_heads = num_attention_heads
        self.inner_dim = width

        self.proj_in = nn.Linear(self.config.in_channels, self.inner_dim, bias=True)

        # 2. Initialize the transformer blocks.
        self.transformer_blocks = nn.ModuleList(
            [
                MultiCondBasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    use_self_attention=True,
                    use_cross_attention=True,
                    self_attention_norm_type="ada_norm_single",
                    cross_attention_dim=self.config.cross_attention_dim,
                    cross_attention_norm_type="ada_norm_single",
                    use_cross_attention_2=self.config.use_cross_attention_2,
                    cross_attention_2_dim=self.config.cross_attention_2_dim,
                    cross_attention_2_norm_type="ada_norm_single",
                    use_cross_attention_3=self.config.use_cross_attention_3,
                    cross_attention_3_dim=self.config.cross_attention_3_dim,
                    cross_attention_3_norm_type="ada_norm_single",
                    dropout=0.0,
                    attention_bias=False,
                    activation_fn="gelu-approximate",
                    num_embeds_ada_norm=1000,
                    norm_elementwise_affine=True,
                    upcast_attention=False,
                    norm_eps=1e-6,
                    attention_type="default",
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = nn.RMSNorm(self.inner_dim, elementwise_affine=True, eps=1e-6)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, self.inner_dim) / self.inner_dim**0.5
        )
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels)

        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, use_additional_conditions=None
        )
        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(AttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models having added KV projections."
                )

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        encoder_hidden_states_3: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask_2: Optional[torch.Tensor] = None,
        encoder_attention_mask_3: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        The [`PixArtTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, n_tokens)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            encoder_hidden_states_2 (`torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            encoder_hidden_states_3 (`torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep (`torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            encoder_attention_mask_2 ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states_2`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            encoder_attention_mask_3 ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states_3`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~Transformer1DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # convert encoder_attention_mask_2 to a bias the same way we do for attention_mask
        if encoder_attention_mask_2 is not None and encoder_attention_mask_2.ndim == 2:
            encoder_attention_mask_2 = (
                1 - encoder_attention_mask_2.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask_2 = encoder_attention_mask_2.unsqueeze(1)

        # convert encoder_attention_mask_2 to a bias the same way we do for attention_mask
        if encoder_attention_mask_3 is not None and encoder_attention_mask_3.ndim == 2:
            encoder_attention_mask_3 = (
                1 - encoder_attention_mask_3.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask_3 = encoder_attention_mask_3.unsqueeze(1)

        # 1. Input
        batch_size = hidden_states.shape[0]
        timestep, embedded_timestep = self.adaln_single(
            timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_hidden_states_2,
                    encoder_hidden_states_3,
                    encoder_attention_mask,
                    encoder_attention_mask_2,
                    encoder_attention_mask_3,
                    timestep,
                    cross_attention_kwargs,
                    None,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_2=encoder_hidden_states_2,
                    encoder_hidden_states_3=encoder_hidden_states_3,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_attention_mask_2=encoder_attention_mask_2,
                    encoder_attention_mask_3=encoder_attention_mask_3,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=None,
                )

        # 3. Output
        shift, scale = (
            self.scale_shift_table[None]
            + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(
            hidden_states.device
        )
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        if not return_dict:
            return (hidden_states,)

        return Transformer1DModelOutput(sample=hidden_states)


@step1x3d_geometry.register("pixart-denoiser")
class PixArtDenoiser(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: Optional[str] = None
        input_channels: int = 32
        width: int = 768
        layers: int = 28
        num_heads: int = 16
        condition_dim: int = 1024
        multi_condition_type: str = "cross_attention"
        use_visual_condition: bool = False
        visual_condition_dim: int = 1024
        n_views: int = 1  # for multi-view condition
        use_caption_condition: bool = False
        caption_condition_dim: int = 1024
        use_label_condition: bool = False
        label_condition_dim: int = 1024

        identity_init: bool = False

    cfg: Config

    def configure(self) -> None:
        self.dit_model = PixArtTransformer1DModel(
            num_attention_heads=self.cfg.num_heads,
            width=self.cfg.width,
            in_channels=self.cfg.input_channels,
            num_layers=self.cfg.layers,
            cross_attention_dim=self.cfg.condition_dim,
            use_cross_attention_2=self.cfg.use_caption_condition
            and self.cfg.multi_condition_type == "cross_attention",
            cross_attention_2_dim=self.cfg.condition_dim,
            use_cross_attention_3=self.cfg.use_label_condition
            and self.cfg.multi_condition_type == "cross_attention",
            cross_attention_3_dim=self.cfg.condition_dim,
        )
        if (
            self.cfg.use_visual_condition
            and self.cfg.visual_condition_dim != self.cfg.condition_dim
        ):
            self.proj_visual_condtion = nn.Sequential(
                nn.RMSNorm(self.cfg.visual_condition_dim),
                nn.Linear(self.cfg.visual_condition_dim, self.cfg.condition_dim),
            )
        if (
            self.cfg.use_caption_condition
            and self.cfg.caption_condition_dim != self.cfg.condition_dim
        ):
            self.proj_caption_condtion = nn.Sequential(
                nn.RMSNorm(self.cfg.caption_condition_dim),
                nn.Linear(self.cfg.caption_condition_dim, self.cfg.condition_dim),
            )
        if (
            self.cfg.use_label_condition
            and self.cfg.label_condition_dim != self.cfg.condition_dim
        ):
            self.proj_label_condtion = nn.Sequential(
                nn.RMSNorm(self.cfg.label_condition_dim),
                nn.Linear(self.cfg.label_condition_dim, self.cfg.condition_dim),
            )

        if self.cfg.identity_init:
            self.identity_initialize()

        if self.cfg.pretrained_model_name_or_path:
            print(
                f"Loading pretrained DiT model from {self.cfg.pretrained_model_name_or_path}"
            )
            ckpt = torch.load(
                self.cfg.pretrained_model_name_or_path,
                map_location="cpu",
                weights_only=False,
            )
            if "state_dict" in ckpt.keys():
                ckpt = ckpt["state_dict"]
            self.load_state_dict(ckpt, strict=True)

    def identity_initialize(self):
        for block in self.dit_model.blocks:
            nn.init.constant_(block.attn.c_proj.weight, 0)
            nn.init.constant_(block.attn.c_proj.bias, 0)
            nn.init.constant_(block.cross_attn.c_proj.weight, 0)
            nn.init.constant_(block.cross_attn.c_proj.bias, 0)
            nn.init.constant_(block.mlp.c_proj.weight, 0)
            nn.init.constant_(block.mlp.c_proj.bias, 0)

    def forward(
        self,
        model_input: torch.FloatTensor,
        timestep: torch.LongTensor,
        visual_condition: Optional[torch.FloatTensor] = None,
        caption_condition: Optional[torch.FloatTensor] = None,
        label_condition: Optional[torch.FloatTensor] = None,
        attention_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict: bool = True,
    ):
        r"""
        Args:
            model_input (torch.FloatTensor): [bs, n_data, c]
            timestep (torch.LongTensor): [bs,]
            visual_condition (torch.FloatTensor): [bs, visual_context_tokens, c]
            text_condition (torch.FloatTensor): [bs, text_context_tokens, c]

        Returns:
            sample (torch.FloatTensor): [bs, n_data, c]

        """

        B, n_data, _ = model_input.shape

        # 0. conditions projector
        condition = []
        if self.cfg.use_visual_condition:
            assert visual_condition.shape[-1] == self.cfg.visual_condition_dim
            if self.cfg.visual_condition_dim != self.cfg.condition_dim:
                visual_condition = self.proj_visual_condtion(visual_condition)
            condition.append(visual_condition)
        else:
            visual_condition = None
        if self.cfg.use_caption_condition:
            assert caption_condition.shape[-1] == self.cfg.caption_condition_dim
            if self.cfg.caption_condition_dim != self.cfg.condition_dim:
                caption_condition = self.proj_caption_condtion(caption_condition)
            condition.append(caption_condition)
        else:
            caption_condition = None
        if self.cfg.use_label_condition:
            assert label_condition.shape[-1] == self.cfg.label_condition_dim
            if self.cfg.label_condition_dim != self.cfg.condition_dim:
                label_condition = self.proj_label_condtion(label_condition)
            condition.append(label_condition)
        else:
            label_condition = None
        assert not (
            visual_condition is None
            and caption_condition is None
            and label_condition is None
        )

        # 1. denoise
        if self.cfg.multi_condition_type == "cross_attention":
            output = self.dit_model(
                model_input,
                timestep,
                visual_condition,
                caption_condition,
                label_condition,
                cross_attention_kwargs,
                return_dict=return_dict,
            )
        elif self.cfg.multi_condition_type == "in_context":
            output = self.dit_model(
                model_input,
                timestep,
                torch.cat(condition, dim=1),
                None,
                None,
                cross_attention_kwargs,
                return_dict=return_dict,
            )
        else:
            raise ValueError

        return output
