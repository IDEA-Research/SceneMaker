from dataclasses import dataclass

import os
import copy
import json
from omegaconf import OmegaConf
import torch
import torch.nn as nn

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import (
    extract_commit_hash,
)

from step1x3d_geometry.utils.config import parse_structured
from step1x3d_geometry.utils.misc import get_device, load_module_weights
from step1x3d_geometry.utils.typing import *


class Configurable:
    @dataclass
    class Config:
        pass

    def __init__(self, cfg: Optional[dict] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)


class Updateable:
    def do_update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except:
                continue  # ignore attributes like property, which can't be retrived using getattr?
            if isinstance(module, Updateable):
                module.do_update_step(
                    epoch, global_step, on_load_weights=on_load_weights
                )
        self.update_step(epoch, global_step, on_load_weights=on_load_weights)

    def do_update_step_end(self, epoch: int, global_step: int):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except:
                continue  # ignore attributes like property, which can't be retrived using getattr?
            if isinstance(module, Updateable):
                module.do_update_step_end(epoch, global_step)
        self.update_step_end(epoch, global_step)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # override this method to implement custom update logic
        # if on_load_weights is True, you should be careful doing things related to model evaluations,
        # as the models and tensors are not guarenteed to be on the same device
        pass

    def update_step_end(self, epoch: int, global_step: int):
        pass


def update_if_possible(module: Any, epoch: int, global_step: int) -> None:
    if isinstance(module, Updateable):
        module.do_update_step(epoch, global_step)


def update_end_if_possible(module: Any, epoch: int, global_step: int) -> None:
    if isinstance(module, Updateable):
        module.do_update_step_end(epoch, global_step)


class BaseObject(Updateable):
    @dataclass
    class Config:
        pass

    cfg: Config  # add this to every subclass of BaseObject to enable static type checking

    def __init__(
        self, cfg: Optional[Union[dict, DictConfig]] = None, *args, **kwargs
    ) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()
        self.configure(*args, **kwargs)

    def configure(self, *args, **kwargs) -> None:
        pass


class BaseModule(ModelMixin, Updateable, nn.Module):
    @dataclass
    class Config:
        weights: Optional[str] = None

    cfg: Config  # add this to every subclass of BaseModule to enable static type checking
    config_name = "config.json"

    def __init__(
        self, cfg: Optional[Union[dict, DictConfig]] = None, *args, **kwargs
    ) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        # self.device = get_device()
        self.configure(*args, **kwargs)
        if self.cfg.weights is not None:
            # format: path/to/weights:module_name
            weights_path, module_name = self.cfg.weights.split(":")
            state_dict, epoch, global_step = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.load_state_dict(state_dict)
            self.do_update_step(
                epoch, global_step, on_load_weights=True
            )  # restore states
        # dummy tensor to indicate model state
        self._dummy: Float[Tensor, "..."]
        self.register_buffer("_dummy", torch.zeros(0).float(), persistent=False)

    def configure(self, *args, **kwargs) -> None:
        pass

    @classmethod
    def load_config(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        return_unused_kwargs=False,
        return_commit_hash=False,
        **kwargs,
    ):
        subfolder = kwargs.pop("subfolder", None)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            if subfolder is not None and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, cls.config_name)
            ):
                config_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, cls.config_name
                )
            elif os.path.isfile(
                os.path.join(pretrained_model_name_or_path, cls.config_name)
            ):
                # Load from a PyTorch checkpoint
                config_file = os.path.join(
                    pretrained_model_name_or_path, cls.config_name
                )
            else:
                raise EnvironmentError(
                    f"Error no file named {cls.config_name} found in directory {pretrained_model_name_or_path}."
                )
        else:
            raise ValueError

        config_dict = json.load(open(config_file, "r"))
        commit_hash = extract_commit_hash(config_file)

        outputs = (config_dict,)

        if return_unused_kwargs:
            outputs += (kwargs,)

        if return_commit_hash:
            outputs += (commit_hash,)

        return outputs

    @classmethod
    def from_config(cls, config: Dict[str, Any] = None, **kwargs):
        model = cls(config)
        return model

    def register_to_config(self, **kwargs):
        pass

    def save_config(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save a configuration object to the directory specified in `save_directory` so that it can be reloaded using the
        [`~ConfigMixin.from_config`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file is saved (will be created if it does not exist).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )

        os.makedirs(save_directory, exist_ok=True)

        # If we save using the predefined names, we can load using `from_config`
        output_config_file = os.path.join(save_directory, self.config_name)

        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        for k in copy.deepcopy(config_dict).keys():
            if k.startswith("pretrained"):
                config_dict.pop(k)
        config_dict.pop("weights")
        with open(output_config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)

        print(f"Configuration saved in {output_config_file}")
