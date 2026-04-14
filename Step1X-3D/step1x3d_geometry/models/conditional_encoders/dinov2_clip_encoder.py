import random
import torch
from torch import nn
import numpy as np
import re
from einops import rearrange
from dataclasses import dataclass
from torchvision import transforms

from diffusers.models.modeling_utils import ModelMixin
from transformers import CLIPTokenizer, CLIPImageProcessor
from transformers import AutoImageProcessor, AutoModel
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer
from transformers.utils import ModelOutput
from typing import Iterable, Optional, Union, List

import step1x3d_geometry
from step1x3d_geometry.utils.typing import *
from .clip.modeling_clip import CLIPModel
from .clip.modeling_conditional_clip import ConditionalCLIPModel
from .base import BaseVisualEncoder, ImageType
from .dinov2.modeling_dinov2 import Dinov2Model
from .dinov2.modeling_conditional_dinov2 import ConditionalDinov2Model
from .dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersModel,
)

CLIP_IMAGE_SIZE = 224


@dataclass
class CLIPEmbedOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    embeds: torch.FloatTensor = None


class DINOEmbedOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None


@step1x3d_geometry.register("dinov2-clip-encoder")
class Dinov2CLIPEncoder(BaseVisualEncoder, ModelMixin):

    @dataclass
    class Config(BaseVisualEncoder.Config):
        pretrained_model_name_or_path: Optional[str] = (
            None  # the pretrained model name or path for condition model
        )
        pretrained_clip_name_or_path: Optional[str] = (
            None  # the pretrained model name or path for clip
        )
        pretrained_dino_name_or_path: Optional[str] = (
            None  # the pretrained model name or path for dino
        )
        pretrained_linear_proj: Optional[str] = None
        freeze_modulation_clip: bool = False
        freeze_modulation_dino: bool = False
        enable_gradient_checkpointing: bool = False
        image_size: int = CLIP_IMAGE_SIZE
        fuse_type: str = "concat"

        dino_type: Optional[str] = None
        clip_type: Optional[str] = None
        kwargs: Optional[dict] = None

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # Load the CLIP model and processor
        if not self.cfg.encode_camera:
            if self.cfg.pretrained_clip_name_or_path is not None:
                self.cfg.clip_type = f"openai/{self.cfg.pretrained_clip_name_or_path.split('openai--')[-1].split('/')[0]}"
                self.clip_model: CLIPModel = CLIPModel.from_pretrained(
                    self.cfg.pretrained_clip_name_or_path
                )
            else:
                print("Loading CLIP model from openai/clip-vit-large-patch14")
                self.dino_type = "openai/clip-vit-large-patch14"
                self.clip_model: CLIPModel = CLIPModel(
                    config=ConditionalCLIPModel.config_class.from_pretrained(
                        "openai/clip-vit-large-patch14",
                    )
                )
            if self.cfg.pretrained_dino_name_or_path is not None:
                self.cfg.dino_type = f"facebook/{self.cfg.pretrained_dino_name_or_path.split('facebook--')[-1].split('/')[0]}"
                self.dino_model: Dinov2Model = AutoModel.from_pretrained(
                    self.cfg.pretrained_dino_name_or_path
                )
            else:
                if (
                    self.cfg.pretrained_model_name_or_path is None
                ):  # default to load Dinov2-base model
                    assert (
                        self.cfg.dino_type is not None
                    ), "The dino_type should be provided"
                    print(f"Loading Dinov2 model from {self.cfg.dino_type}")
                    if "reg" in self.cfg.dino_type:
                        self.dino_model: Dinov2WithRegistersModel = (
                            Dinov2WithRegistersModel(
                                config=Dinov2WithRegistersModel.config_class.from_pretrained(
                                    self.cfg.dino_type,
                                )
                            )
                        )
                    else:
                        self.dino_model: Dinov2Model = Dinov2Model(
                            config=Dinov2Model.config_class.from_pretrained(
                                self.dino_type,
                            )
                        )
                elif "dinov2base" in self.cfg.pretrained_model_name_or_path:
                    print("Loading Dinov2 model from facebook/dinov2-base")
                    self.cfg.dino_type = "facebook/dinov2-base"
                    self.dino_model: Dinov2Model = Dinov2Model(
                        config=Dinov2Model.config_class.from_pretrained(
                            "facebook/dinov2-base",
                        )
                    )
                elif "dinov2regbase" in self.cfg.pretrained_model_name_or_path:
                    print(
                        "Loading Dinov2 model from facebook/dinov2-with-registers-base"
                    )
                    self.cfg.dino_type = "facebook/dinov2-with-registers-base"
                    self.dino_model: Dinov2WithRegistersModel = (
                        Dinov2WithRegistersModel(
                            config=Dinov2WithRegistersModel.config_class.from_pretrained(
                                "facebook/dinov2-with-registers-base",
                            )
                        )
                    )
                elif "dinov2reglarge" in self.cfg.pretrained_model_name_or_path:
                    print(
                        "Loading Dinov2 model from facebook/dinov2-with-registers-large"
                    )
                    self.cfg.dino_type = "facebook/dinov2-with-registers-large"
                    self.dino_model: Dinov2WithRegistersModel = (
                        Dinov2WithRegistersModel(
                            config=Dinov2WithRegistersModel.config_class.from_pretrained(
                                "facebook/dinov2-with-registers-large",
                            )
                        )
                    )
                else:
                    raise ValueError(
                        f"Unknown Dinov2 model: {self.cfg.pretrained_model_name_or_path}"
                    )
        else:
            # clip
            conditional_clip_config = ConditionalCLIPModel.config_class.from_pretrained(
                self.cfg.pretrained_clip_name_or_path,
            )
            conditional_clip_config.vision_config.modulation_dim = (
                self.cfg.camera_embeds_dim
            )
            self.clip_model: CLIPModel = ConditionalCLIPModel.from_pretrained(
                self.cfg.pretrained_clip_name_or_path,
                vision_config=conditional_clip_config.vision_config,
            )

            # dino
            conditional_vit_config = (
                ConditionalDinov2Model.config_class.from_pretrained(
                    self.cfg.pretrained_dino_name_or_path,
                )
            )
            conditional_vit_config.modulation_dim = self.cfg.camera_embeds_dim
            self.dino_model: ConditionalDinov2Model = (
                ConditionalDinov2Model.from_pretrained(
                    self.cfg.pretrained_dino_name_or_path, config=conditional_vit_config
                )
            )

        self.image_preprocess_clip = CLIPImageProcessor()
        self.image_preprocess_dino = AutoImageProcessor.from_pretrained(
            self.cfg.dino_type
            if self.cfg.pretrained_dino_name_or_path is None
            else self.cfg.pretrained_dino_name_or_path
        )
        self.transform_clip = transforms.Compose(
            [
                transforms.Resize(
                    CLIP_IMAGE_SIZE,
                    transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),  # clip is CLIP_IMAGE_SIZE
                transforms.CenterCrop(CLIP_IMAGE_SIZE),  # crop a square.
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.transform_dino = transforms.Compose(
            [
                transforms.Resize(
                    self.cfg.image_size,
                    transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(self.cfg.image_size),  # crop a square
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        if self.cfg.enable_gradient_checkpointing:
            self.dino_model.encoder.gradient_checkpointing = True

        if self.cfg.zero_uncond_embeds:
            image_size = max(self.cfg.image_size, self.cfg.image_size)
            self.empty_image_embeds_dino = torch.zeros(
                (self.cfg.n_views, (image_size // 14) ** 2 + 1, 1024)
            ).detach()
            self.empty_image_embeds_clip = torch.zeros(
                (self.cfg.n_views, (CLIP_IMAGE_SIZE // 14) ** 2 + 1, 1024)
            ).detach()
            if self.cfg.fuse_type == "concat":
                self.empty_image_embeds = torch.cat(
                    [self.empty_image_embeds_dino, self.empty_image_embeds_clip], dim=1
                )
            else:
                raise ValueError
        else:
            if self.cfg.encode_camera:
                self.empty_image_embeds_dino = self.encode_image_dino(
                    torch.zeros(
                        self.cfg.n_views, self.cfg.image_size, self.cfg.image_size, 3
                    ),
                    self.cameras[: self.cfg.n_views],
                ).detach()
                self.empty_image_embeds_clip = self.encode_image_clip(
                    torch.zeros(
                        self.cfg.n_views, self.cfg.image_size, self.cfg.image_size, 3
                    ),
                    self.cameras[: self.cfg.n_views],
                ).detach()
            else:
                self.empty_image_embeds_dino = self.encode_image_dino(
                    torch.zeros(
                        self.cfg.n_views, self.cfg.image_size, self.cfg.image_size, 3
                    )
                ).detach()
                self.empty_image_embeds_clip = self.encode_image_clip(
                    torch.zeros(
                        self.cfg.n_views, self.cfg.image_size, self.cfg.image_size, 3
                    )
                ).detach()
            self.empty_image_embeds_clip, self.empty_image_embeds_dino = (
                self.align_clip_dino(
                    self.empty_image_embeds_clip, self.empty_image_embeds_dino
                )
            )
            self.empty_image_embeds = torch.cat(
                [self.empty_image_embeds_dino, self.empty_image_embeds_clip], dim=1
            )

        # Freeze the clip model parameters
        self.clip_model.eval()
        for k, p in self.clip_model.named_parameters():
            ks = k.split(".")
            if (
                "mod_norm1" in ks
                or "mod_norm2" in ks
                and not self.cfg.freeze_modulation_clip
            ):
                p.requires_grad_(not self.cfg.freeze_modulation_clip)
            else:
                p.requires_grad_(False)

        # freeze the dino model parameters
        self.dino_model.eval()
        for k, p in self.dino_model.named_parameters():
            ks = k.split(".")
            if (
                "mod_norm1" in ks
                or "mod_norm2" in ks
                and not self.cfg.freeze_modulation_dino
            ):
                p.requires_grad_(not self.cfg.freeze_modulation_dino)
            else:
                p.requires_grad_(False)

        # add a linear projection layer to project the dino embeddings to the same dimension as clip embeddings
        if (
            self.clip_model.config.vision_config.hidden_size
            != self.dino_model.config.hidden_size
        ):
            self.linear_proj = nn.Linear(
                self.clip_model.config.vision_config.hidden_size,
                self.dino_model.config.vision_config.hidden_size,
                bias=False,
            )
        else:
            self.linear_proj = nn.Identity()

        if self.cfg.pretrained_model_name_or_path is not None:
            print(f"Loading ckpt from {self.cfg.pretrained_model_name_or_path}")
            ckpt = torch.load(
                self.cfg.pretrained_model_name_or_path, map_location="cpu"
            )["state_dict"]
            pretrained_model_ckpt = {}
            for k, v in ckpt.items():
                if k.startswith("condition."):
                    pretrained_model_ckpt[k.replace("condition.", "")] = v
            self.load_state_dict(pretrained_model_ckpt, strict=True)

    def encode_image_clip(
        self,
        images: Iterable[Optional[ImageType]],
        cameras: Optional[torch.Tensor] = None,
        force_none_camera_embeds: bool = False,
        return_dict: bool = False,
        **kwargs,
    ) -> torch.FloatTensor:
        camera_embeds = None
        if isinstance(images, (np.ndarray, torch.Tensor)):  # for training process
            assert (
                images.min() >= 0.0 and images.max() <= 1.0
            ), "The pixel values should be in the range of [0, 1]"
            if self.cfg.encode_camera:
                assert cameras is not None, "The cameras should be provided"
                camera_embeds = self.encode_camera(cameras)
            pixel_values = self.transform_clip(images.permute(0, 3, 1, 2))
        else:  # for inference process
            if self.cfg.encode_camera:
                if cameras is None:
                    bs = len(images) // self.cfg.n_views
                    cameras = (
                        self.cameras[: self.cfg.n_views]
                        .repeat(bs, 1, 1)
                        .to(self.clip_model.device)
                    )
                camera_embeds = self.encode_camera(cameras)
            pixel_values = self.image_preprocess_clip.preprocess(
                images,
                return_tensors="pt",
                do_rescale=True,
                do_resize=True,
                size=CLIP_IMAGE_SIZE,
                crop_size=CLIP_IMAGE_SIZE,
            ).pixel_values

        if force_none_camera_embeds:
            camera_embeds = None

        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
            if camera_embeds is not None:
                camera_embeds = camera_embeds.unsqueeze(1)

        if self.cfg.encode_camera and camera_embeds is not None:
            vision_outputs = self.clip_model.vision_model(
                pixel_values=rearrange(
                    pixel_values.to(self.clip_model.device), "B N C H W -> (B N) C H W"
                ),
                condition=rearrange(camera_embeds, "B N C -> (B N) C"),
            )

        else:
            vision_outputs = self.clip_model.vision_model(
                pixel_values=rearrange(
                    pixel_values.to(self.clip_model.device), "B N C H W -> (B N) C H W"
                ),
            )

        if return_dict:
            # clip
            pooler_output = vision_outputs[1]  # pooled_output
            image_features = self.clip_model.visual_projection(pooler_output)
            clip_embeds = vision_outputs.last_hidden_state

            clip_embeds_dict = CLIPEmbedOutput(
                last_hidden_state=clip_embeds,
                pooler_output=pooler_output,
                embeds=image_features,
            )

            return clip_embeds_dict
        else:
            return vision_outputs.last_hidden_state

    def encode_image_dino(
        self,
        images: Iterable[Optional[ImageType]],
        cameras: Optional[torch.Tensor] = None,
        force_none_camera_embeds: bool = False,
        return_dict: bool = False,
        **kwargs,
    ) -> torch.FloatTensor:
        camera_embeds = None
        if isinstance(images, (np.ndarray, torch.Tensor)):  # for training process
            assert (
                images.min() >= 0.0 and images.max() <= 1.0
            ), "The pixel values should be in the range of [0, 1]"
            if self.cfg.encode_camera:
                assert cameras is not None, "The cameras should be provided"
                camera_embeds = self.encode_camera(cameras)
            pixel_values = self.transform_dino(images.permute(0, 3, 1, 2))
        else:  # for inference process
            if self.cfg.encode_camera:
                if cameras is None:
                    bs = len(images) // self.cfg.n_views
                    cameras = (
                        self.cameras[: self.cfg.n_views]
                        .repeat(bs, 1, 1)
                        .to(self.dino_model.device)
                    )
                camera_embeds = self.encode_camera(cameras)
            pixel_values = self.image_preprocess_dino.preprocess(
                images,
                return_tensors="pt",
                do_rescale=True,
                do_resize=True,
                size=self.cfg.image_size,
                crop_size=self.cfg.image_size,
            ).pixel_values

        if force_none_camera_embeds:
            camera_embeds = None

        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
            if camera_embeds is not None:
                camera_embeds = camera_embeds.unsqueeze(1)

        if self.cfg.encode_camera and camera_embeds is not None:
            vision_outputs = self.dino_model(
                rearrange(
                    pixel_values.to(self.dino_model.device), "B N C H W -> (B N) C H W"
                ),
                condition=rearrange(camera_embeds, "B N C -> (B N) C"),
            )
        else:
            vision_outputs = self.dino_model(
                rearrange(
                    pixel_values.to(self.dino_model.device), "B N C H W -> (B N) C H W"
                ),
            )

        if return_dict:
            # dino
            dino_embeds_dict = DINOEmbedOutput(
                last_hidden_state=vision_outputs.last_hidden_state,
                pooler_output=vision_outputs.pooler_output,
            )
            return dino_embeds_dict
        else:
            return vision_outputs.last_hidden_state

    def align_clip_dino(self, clip_embeds, dino_embeds):
        if (
            clip_embeds.shape[-2] != dino_embeds.shape[-2]
        ):  # different shape, interpolate the clip embeddings to the same shape as dino embeddings
            assert (
                clip_embeds.shape[-2] == (self.cfg.image_size // 14) ** 2 + 1
            ), "The clip embeddings should have the shape of (n_views, (image_size // 14) ** 2 + 1, 1024)"
            clip_embeds_patch_tokens = clip_embeds[:, 1:].view(
                clip_embeds.shape[0],
                self.cfg.image_size // 14,
                self.cfg.image_size // 14,
                1024,
            )
            clip_embeds_patch_tokens = (
                torch.nn.functional.interpolate(
                    clip_embeds_patch_tokens.permute(0, 3, 1, 2),
                    size=(self.cfg.image_size // 14, self.cfg.image_size // 14),
                    mode="bilinear",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .view(clip_embeds.shape[0], -1, 1024)
            )
            clip_embeds = torch.cat(
                [clip_embeds[:, :1], clip_embeds_patch_tokens], dim=1
            )
        return clip_embeds, dino_embeds

    def encode_image(
        self,
        images: Iterable[Optional[ImageType]],
        cameras: Optional[torch.Tensor] = None,
        force_none_camera_embeds: bool = False,
        return_dict: bool = False,
        **kwargs,
    ) -> torch.FloatTensor:
        clip_embeds = self.encode_image_clip(images, cameras)
        dino_embeds = self.encode_image_dino(images, cameras)
        if (
            self.dino_model.__class__.__name__ == "Dinov2WithRegistersModel"
        ):  # x_norm_clstoken, x_norm_regtokens, x_norm_patchtokens
            dino_embeds = torch.cat(
                [
                    dino_embeds[:, :1],
                    dino_embeds[:, self.dino_model.config.num_register_tokens + 1 :],
                ],
                dim=1,
            )

        clip_embeds = self.linear_proj(clip_embeds)  # bs, 257, 1024

        if self.cfg.fuse_type == "concat":
            visual_embeds = torch.cat([dino_embeds, clip_embeds], dim=1)
        # elif self.cfg.fuse_type == 'add':
        #     clip_embeds, dino_embeds = self.align_clip_dino(clip_embeds, dino_embeds)
        else:
            raise ValueError

        return visual_embeds
