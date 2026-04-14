# Some parts of this file are adapted from Hugging Face Diffusers library.
from typing import Any, Dict, Optional, Union, Tuple
from dataclasses import dataclass

import re
import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    AttnProcessor,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.normalization import (
    AdaLayerNormSingle,
    AdaLayerNormContinuous,
    FP32LayerNorm,
    LayerNorm,
)

from ..attention_processor import FusedFluxAttnProcessor2_0, FluxAttnProcessor2_0
from ..attention import FluxTransformerBlock, FluxSingleTransformerBlock

import step1x3d_geometry
from step1x3d_geometry.utils.base import BaseModule

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class Transformer1DModelOutput:
    sample: torch.FloatTensor


class FluxTransformer1DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-la

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
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        width: int = 2048,
        in_channels: int = 4,
        num_layers: int = 19,
        num_single_layers: int = 38,
        cross_attention_dim: int = 768,
    ):
        super().__init__()
        # Set some common variables used across the board.
        self.out_channels = in_channels
        self.num_heads = num_attention_heads
        self.inner_dim = width

        # self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        # self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim)
        time_embed_dim, timestep_input_dim = self._set_time_proj(
            "positional",
            inner_dim=self.inner_dim,
            flip_sin_to_cos=False,
            freq_shift=0,
            time_embedding_dim=None,
        )
        self.time_proj = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, act_fn="gelu", out_dim=self.inner_dim
        )
        self.proj_in = nn.Linear(self.config.in_channels, self.inner_dim, bias=True)
        self.proj_cross_attention = nn.Linear(
            self.config.cross_attention_dim, self.inner_dim, bias=True
        )

        # 2. Initialize the transformer blocks.
        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=width // num_attention_heads,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=width // num_attention_heads,
                )
                for _ in range(self.config.num_single_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def _set_time_proj(
        self,
        time_embedding_type: str,
        inner_dim: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or inner_dim * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(
                    f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}."
                )
            self.time_embed = GaussianFourierProjection(
                time_embed_dim // 2,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos,
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or inner_dim * 4

            self.time_embed = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
            timestep_input_dim = inner_dim
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        return time_embed_dim, timestep_input_dim

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

        self.set_attn_processor(FusedFluxAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

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
        self.set_attn_processor(FluxAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(
        self, chunk_size: Optional[int] = None, dim: int = 0
    ) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        timestep: Union[int, float, torch.LongTensor],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        """
        The [`HunyuanDiT2DModel`] forward method.

        Args:
        hidden_states (`torch.Tensor` of shape `(batch size, dim, latents_size)`):
            The input tensor.
        timestep ( `torch.LongTensor`, *optional*):
            Used to indicate denoising step.
        encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer.
        encoder_hidden_states_2 ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer.
        return_dict: bool
            Whether to return a dictionary.
        """

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        _, N, _ = hidden_states.shape

        # import pdb; pdb.set_trace()
        # timesteps_proj = self.time_proj(timestep) # N x 256
        # temb = self.time_embed(timesteps_proj).to(hidden_states.dtype)
        temb = self.time_embed(timestep).to(hidden_states.dtype)  # N x 1280
        temb = self.time_proj(temb)  # N x 1280

        hidden_states = self.proj_in(hidden_states)
        encoder_hidden_states = self.proj_cross_attention(encoder_hidden_states)

        for layer, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                encoder_hidden_states, hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        None,  # image_rotary_emb
                        attention_kwargs,
                    )
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=None,
                    joint_attention_kwargs=attention_kwargs,
                )  # (N, L, D)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for layer, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    None,  # image_rotary_emb
                    attention_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                    image_rotary_emb=None,
                    joint_attention_kwargs=attention_kwargs,
                )  # (N, L, D)

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        # final layer
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer1DModelOutput(sample=hidden_states)


@step1x3d_geometry.register("flux-denoiser")
class FluxDenoiser(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: Optional[str] = None
        input_channels: int = 32
        width: int = 768
        layers: int = 12
        num_single_layers: int = 12
        num_heads: int = 16
        condition_dim: int = 1024
        multi_condition_type: str = "in_context"
        use_visual_condition: bool = False
        visual_condition_dim: int = 1024
        n_views: int = 1
        use_caption_condition: bool = False
        caption_condition_dim: int = 1024
        use_label_condition: bool = False
        label_condition_dim: int = 1024

        identity_init: bool = False

    cfg: Config

    def configure(self) -> None:
        assert (
            self.cfg.multi_condition_type == "in_context"
        ), "Flux Denoiser only support in_context learning of multiple conditions"
        self.dit_model = FluxTransformer1DModel(
            num_attention_heads=self.cfg.num_heads,
            width=self.cfg.width,
            in_channels=self.cfg.input_channels,
            num_layers=self.cfg.layers,
            num_single_layers=self.cfg.num_single_layers,
            cross_attention_dim=self.cfg.condition_dim,
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
                weights_only=True,
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
        return_dict: bool = True,
    ):
        r"""
        Args:
            model_input (torch.FloatTensor): [bs, n_data, c]
            timestep (torch.LongTensor): [bs,]
            visual_condition (torch.FloatTensor): [bs, visual_context_tokens, c]
            caption_condition (torch.FloatTensor): [bs, text_context_tokens, c]
            label_condition (torch.FloatTensor): [bs, c]

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
        if self.cfg.use_caption_condition:
            assert caption_condition.shape[-1] == self.cfg.caption_condition_dim
            if self.cfg.caption_condition_dim != self.cfg.condition_dim:
                caption_condition = self.proj_caption_condtion(caption_condition)
            condition.append(caption_condition)
        if self.cfg.use_label_condition:
            assert label_condition.shape[-1] == self.cfg.label_condition_dim
            if self.cfg.label_condition_dim != self.cfg.condition_dim:
                label_condition = self.proj_label_condtion(label_condition)
            condition.append(label_condition)

        # 1. denoise
        output = self.dit_model(
            model_input,
            timestep,
            torch.cat(condition, dim=1),
            attention_kwargs,
            return_dict=return_dict,
        )

        return output
