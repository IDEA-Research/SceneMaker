import random
import torch
from torch import nn
import numpy as np
import re
from einops import rearrange
from dataclasses import dataclass
from torchvision import transforms
from diffusers.models.modeling_utils import ModelMixin

from transformers.utils import ModelOutput
from typing import Iterable, Optional, Union, List

import step1x3d_geometry
from step1x3d_geometry.utils.typing import *
from step1x3d_geometry.utils.misc import get_device

from .base import BaseLabelEncoder

DEFAULT_POSE = 0  # "unknown", "t-pose", "a-pose", uncond
NUM_POSE_CLASSES = 3
POSE_MAPPING = {"unknown": 0, "t-pose": 1, "a-pose": 2, "uncond": 3}

DEFAULT_SYMMETRY_TYPE = 0  # "asymmetry", "x", uncond
NUM_SYMMETRY_TYPE_CLASSES = 2
SYMMETRY_TYPE_MAPPING = {"asymmetry": 0, "x": 1, "y": 0, "z": 0, "uncond": 2}

DEFAULT_GEOMETRY_QUALITY = 0  # "normal", "smooth", "sharp", uncond,
NUM_GEOMETRY_QUALITY_CLASSES = 3
GEOMETRY_QUALITY_MAPPING = {"normal": 0, "smooth": 1, "sharp": 2, "uncod": 3}


@step1x3d_geometry.register("label-encoder")
class LabelEncoder(BaseLabelEncoder, ModelMixin):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.

    Args:
        num_classes (`int`): The number of classes.
        hidden_size (`int`): The size of the vector embeddings.
    """

    def configure(self) -> None:
        super().configure()

        if self.cfg.zero_uncond_embeds:
            self.embedding_table_tpose = nn.Embedding(
                NUM_POSE_CLASSES, self.cfg.hidden_size
            )
            self.embedding_table_symmetry_type = nn.Embedding(
                NUM_SYMMETRY_TYPE_CLASSES, self.cfg.hidden_size
            )
            self.embedding_table_geometry_quality = nn.Embedding(
                NUM_GEOMETRY_QUALITY_CLASSES, self.cfg.hidden_size
            )
        else:
            self.embedding_table_tpose = nn.Embedding(
                NUM_POSE_CLASSES + 1, self.cfg.hidden_size
            )
            self.embedding_table_symmetry_type = nn.Embedding(
                NUM_SYMMETRY_TYPE_CLASSES + 1, self.cfg.hidden_size
            )
            self.embedding_table_geometry_quality = nn.Embedding(
                NUM_GEOMETRY_QUALITY_CLASSES + 1, self.cfg.hidden_size
            )

        if self.cfg.zero_uncond_embeds:
            self.empty_label_embeds = torch.zeros((1, 3, self.cfg.hidden_size)).detach()
        else:
            self.empty_label_embeds = (
                self.encode_label(  # the last class label is for the uncond
                    [{"pose": "", "symetry": "", "geometry_type": ""}]
                ).detach()
            )

        # load pretrained_model_name_or_path
        if self.cfg.pretrained_model_name_or_path is not None:
            print(f"Loading ckpt from {self.cfg.pretrained_model_name_or_path}")
            ckpt = torch.load(
                self.cfg.pretrained_model_name_or_path, map_location="cpu"
            )["state_dict"]
            pretrained_model_ckpt = {}
            for k, v in ckpt.items():
                if k.startswith("label_condition."):
                    pretrained_model_ckpt[k.replace("label_condition.", "")] = v
            self.load_state_dict(pretrained_model_ckpt, strict=True)

    def encode_label(self, labels: List[dict]) -> torch.FloatTensor:
        tpose_label_embeds = []
        symmetry_type_label_embeds = []
        geometry_quality_label_embeds = []

        for label in labels:
            if "pose" in label.keys():
                if label["pose"] is None or label["pose"] == "":
                    tpose_label_embeds.append(
                        torch.zeros(self.cfg.hidden_size).detach().to(get_device())
                    )
                else:
                    tpose_label_embeds.append(
                        self.embedding_table_symmetry_type(
                            torch.tensor(POSE_MAPPING[label["pose"][0]]).to(
                                get_device()
                            )
                        )
                    )
            else:
                tpose_label_embeds.append(
                    self.embedding_table_tpose(
                        torch.tensor(DEFAULT_POSE).to(get_device())
                    )
                )

            if "symmetry" in label.keys():
                if label["symmetry"] is None or label["symmetry"] == "":
                    symmetry_type_label_embeds.append(
                        torch.zeros(self.cfg.hidden_size).detach().to(get_device())
                    )
                else:
                    symmetry_type_label_embeds.append(
                        self.embedding_table_symmetry_type(
                            torch.tensor(
                                SYMMETRY_TYPE_MAPPING[label["symmetry"]]
                            ).to(get_device())
                        )
                    )
            else:
                symmetry_type_label_embeds.append(
                    self.embedding_table_symmetry_type(
                        torch.tensor(DEFAULT_SYMMETRY_TYPE).to(get_device())
                    )
                )

            if "geometry_type" in label.keys():
                if label["geometry_type"] is None or label["geometry_type"] == "":
                    geometry_quality_label_embeds.append(
                        torch.zeros(self.cfg.hidden_size).detach().to(get_device())
                    )
                else:
                    geometry_quality_label_embeds.append(
                        self.embedding_table_geometry_quality(
                            torch.tensor(
                                GEOMETRY_QUALITY_MAPPING[label["geometry_type"][0]]
                            ).to(get_device())
                        )
                    )
            else:
                geometry_quality_label_embeds.append(
                    self.embedding_table_geometry_quality(
                        torch.tensor(DEFAULT_GEOMETRY_QUALITY).to(get_device())
                    )
                )

        tpose_label_embeds = torch.stack(tpose_label_embeds)
        symmetry_type_label_embeds = torch.stack(symmetry_type_label_embeds)
        geometry_quality_label_embeds = torch.stack(geometry_quality_label_embeds)

        label_embeds = torch.stack(
            [
                tpose_label_embeds,
                symmetry_type_label_embeds,
                geometry_quality_label_embeds,
            ],
            dim=1,
        ).to(self.dtype)

        return label_embeds
