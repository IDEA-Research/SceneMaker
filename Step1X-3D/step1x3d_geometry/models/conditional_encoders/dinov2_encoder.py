import random
import torch
from torch import nn
import numpy as np
import re
from einops import rearrange
from dataclasses import dataclass
from torchvision import transforms

from diffusers.models.modeling_utils import ModelMixin
from transformers import AutoImageProcessor, AutoModel
from transformers.utils import ModelOutput
from typing import Iterable, Optional, Union, List

import step1x3d_geometry
from step1x3d_geometry.utils.typing import *
from .base import BaseVisualEncoder, ImageType
from .dinov2.modeling_dinov2 import Dinov2Model
from .dinov2.modeling_conditional_dinov2 import ConditionalDinov2Model
from .dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersModel,
)


class DINOEmbedOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None


@step1x3d_geometry.register("dinov2-encoder")
class Dinov2Encoder(BaseVisualEncoder, ModelMixin):

    @dataclass
    class Config(BaseVisualEncoder.Config):
        pretrained_model_name_or_path: Optional[str] = (
            None  # the pretrained model name or path for condition model
        )
        pretrained_dino_name_or_path: Optional[str] = (
            None  # the pretrained model name or path for dino
        )
        freeze_modulation_dino: bool = False
        enable_gradient_checkpointing: bool = False
        image_size: int = 224
        dino_type: Optional[str] = None
        kwargs: Optional[dict] = None

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # Load the DINOV2 model and processor
        if not self.cfg.encode_camera:
            if self.cfg.pretrained_dino_name_or_path is not None:
                self.cfg.dino_type = f"facebook/{self.cfg.pretrained_dino_name_or_path.split('facebook--')[-1].split('/')[0]}"
                if self.cfg.kwargs is not None:
                    self.dino_model: Dinov2Model = AutoModel.from_pretrained(
                        self.cfg.pretrained_dino_name_or_path, **self.cfg.kwargs
                    )
                else:
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

        self.image_preprocess_dino = AutoImageProcessor.from_pretrained(
            self.cfg.dino_type
            if self.cfg.pretrained_dino_name_or_path is None
            else self.cfg.pretrained_dino_name_or_path
        )
        self.transform_dino = transforms.Compose(
            [
                transforms.Resize(
                    self.cfg.image_size,
                    transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(
                    self.cfg.image_size
                ),  # crop a (image_size, image_size) square
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        if self.cfg.enable_gradient_checkpointing:
            self.dino_model.encoder.gradient_checkpointing = True

        if self.cfg.zero_uncond_embeds:
            self.empty_image_embeds = torch.zeros(
                (
                    self.cfg.n_views,
                    (self.cfg.image_size // 14) ** 2 + 1,
                    self.dino_model.config.hidden_size,
                )
            ).detach()
        else:
            if self.cfg.encode_camera:
                self.empty_image_embeds = self.encode_image_dino(
                    torch.zeros(
                        self.cfg.n_views, self.cfg.image_size, self.cfg.image_size, 3
                    ),
                    self.cameras[: self.cfg.n_views],
                ).detach()
            else:
                self.empty_image_embeds = self.encode_image_dino(
                    torch.zeros(
                        self.cfg.n_views, self.cfg.image_size, self.cfg.image_size, 3
                    )
                ).detach()

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

        # load pretrained_model_name_or_path
        if self.cfg.pretrained_model_name_or_path is not None:
            print(f"Loading ckpt from {self.cfg.pretrained_model_name_or_path}")
            ckpt = torch.load(
                self.cfg.pretrained_model_name_or_path, map_location="cpu"
            )["state_dict"]
            pretrained_model_ckpt = {}
            for k, v in ckpt.items():
                if k.startswith("visual_condition."):
                    pretrained_model_ckpt[k.replace("visual_condition.", "")] = v
            self.load_state_dict(pretrained_model_ckpt, strict=True)

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

    def encode_image(
        self,
        images: Iterable[Optional[ImageType]],
        cameras: Optional[torch.Tensor] = None,
        force_none_camera_embeds: bool = False,
        return_dict: bool = False,
        **kwargs,
    ) -> torch.FloatTensor:
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
        return dino_embeds
