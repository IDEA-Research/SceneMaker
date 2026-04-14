import math
import os
import json
import re
import cv2
from dataclasses import dataclass, field

import random
import imageio
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import streaming
from streaming.base import StreamingDataset

from craftsman.utils.typing import *
from craftsman.utils.misc import get_rank

from .base import BaseDataModuleConfig, BaseDataset

@dataclass
class BaseS3DataModuleConfig(BaseDataModuleConfig):
    remote_dir: Optional[List[str]] = None     # remote directory of the data
    local_dir: Optional[List[str]] = None      # root directory of the data, used as cache
    download_retry: int = 2              # number of retries for downloading data
    download_timeout: float = 60         # timeout for downloading data
    cache_limit: Optional[int] = None    # maximum number of files to cache, (e.g., 100b, 64kb, 77mb, and so on)
    shuffle: bool = False                # whether to shuffle the data

    ################################# General argumentation #################################
    random_flip: bool = False            # whether to randomly flip the input point cloud and the input images
    random_color_jitter: bool = False    # whether to randomly color jitter the input images

    ################################# Geometry part #################################
    load_geometry: bool = True           # whether to load geometry data
    with_sharp_data: bool = False
    geo_data_type: str = "occupancy"     # occupancy, sdf
    # for occupancy and sdf data
    n_samples: int = 4096                # number of points in input point cloud
    upsample_ratio: int = 1              # upsample ratio for input point cloud
    sampling_strategy: Optional[str] = "random"    # sampling strategy for input point cloud
    scale: float = 1.0                   # scale of the input point cloud and target supervision
    noise_sigma: float = 0.0             # noise level of the input point cloud
    rotate_points: bool = False          # whether to rotate the input point cloud and the supervision
    load_supervision: bool = False        # whether to load supervision
    supervision_type: str = "occupancy"  # occupancy, sdf, tsdf, tsdf_w_surface
    n_supervision: int = 10000           # number of points in supervision
    tsdf_threshold: float = 0.01         # threshold for truncating sdf values, used when input is sdf

    ################################# Image part #################################
    load_image: bool = False             # whether to load images 
    image_type: str = "rgb"              # rgb, normal, rgb_or_normal
    image_type_ratio: float = 1.0        # ratio of rgb for each dataset when image_type is "rgb_or_normal"
    images_per_sample: int = -1          # the loaded number of images
    background_color: Tuple[int, int, int] = field(
            default_factory=lambda: (255, 255, 255)
        )
    idx: Optional[List[int]] = None      # index of the image to load
    n_views: int = 1                     # number of views
    foreground_ratio: Optional[float] = 0.95 

    ################################# Caption part #################################
    load_caption: bool = False           # whether to load captions
    
    batch_size: int = 32
    num_workers: int = 0
    text_max_length: int = 77
    

class BaseS3Dataset(StreamingDataset, BaseDataset):
    def __init__(self, cfg: Any, split: str) -> None:
        self.cfg: BaseDataModuleConfig = cfg
        self.split = split
        
        os.environ["RANK"] = str(get_rank())

        if self.cfg.remote_dir is None:
            streams = []
            for local_dir in self.cfg.local_dir:
                streams.append(streaming.Stream(local=local_dir, split=split))
        else:
            print(f"Load data from {self.cfg.remote_dir} and cache to {self.cfg.local_dir}")
            assert len(self.cfg.remote_dir) == len(self.cfg.local_dir), "length of remote_dir and local_dir should be same"
            streams = []
            for remote_dir, local_dir in zip(self.cfg.remote_dir, self.cfg.local_dir):
                streams.append(streaming.Stream(remote=remote_dir, local=local_dir, split=split))

        streaming.base.util.clean_stale_shared_memory()
        StreamingDataset.__init__(
            self,
            streams=streams,
            remote=None,
            local=None,
            split=None,
            batch_size=self.cfg.batch_size,
            download_retry=self.cfg.download_retry,
            download_timeout=self.cfg.download_timeout,
            cache_limit=self.cfg.cache_limit,
            shuffle=self.cfg.shuffle,
            allow_unsafe_types=True,
        )
        print(f"Found {len(self)} data")

        # add ColorJitter transforms for input images
        if self.cfg.random_color_jitter:
            self.transforms = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)

        # default camera embedding
        if self.cfg.n_views != 1:
            self.camera_embedding = self._get_default_camera()
            assert self.cfg.n_views == self.camera_embedding.shape[0]

    def _load_shape_from_occupancy_or_sdf(self, data) -> Dict[str, Any]:
        sharp_surface = None
        if self.cfg.geo_data_type == "occupancy":
            # for input point cloud, using Objaverse-MIX data
            surface = data['points'] * 2 # range from -1 to 1
            normal = data['normals']
            surface = np.concatenate([surface, normal], axis=1)
        elif self.cfg.geo_data_type == "sdf":
            # for input point cloud
            # surface = data["surface"].copy()
            surface = data["fps_coarse_surface"].copy()
            if self.cfg.with_sharp_data:
                sharp_surface = data["fps_sharp_surface"].copy()
        else:
            raise NotImplementedError(f"Data type {self.cfg.geo_data_type} not implemented")
        
        # random sampling
        if self.cfg.sampling_strategy == "random":
            rng = np.random.default_rng()
            ind = rng.choice(surface.shape[0], self.cfg.upsample_ratio * self.cfg.n_samples, replace=False)
            surface = surface[ind]
            if self.cfg.with_sharp_data:
                sharp_surface = sharp_surface[ind]
        elif self.cfg.sampling_strategy == "fps":
            import fpsample
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(surface[:, :3], self.cfg.n_samples, h=5)
            surface = surface[kdline_fps_samples_idx]
            if self.cfg.with_sharp_data:
                kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(sharp_surface[:, :3], self.cfg.n_samples, h=5)
                sharp_surface = sharp_surface[kdline_fps_samples_idx]
        elif self.cfg.sampling_strategy is None:
            pass
        else:
            raise NotImplementedError(f"sampling strategy {self.cfg.sampling_strategy} not implemented")
        # rescale data
        surface[:, :3] = surface[:, :3] * self.cfg.scale # target scale
        # add noise to input point cloud
        surface[:, :3] += (np.random.rand(surface.shape[0], 3) * 2 - 1) * self.cfg.noise_sigma
        surface = surface.astype(np.float32)
        if self.cfg.with_sharp_data:
            sharp_surface[:, :3] = sharp_surface[:, :3] * self.cfg.scale # target scale
            # add noise to input point cloud
            sharp_surface[:, :3] += (np.random.rand(sharp_surface.shape[0], 3) * 2 - 1) * self.cfg.noise_sigma
            sharp_surface = sharp_surface.astype(np.float32)
            if sharp_surface.shape[0] < 16384:
                # raise exception
                raise Exception(f"the point of sharp_surface is smaller than 16384:{sharp_surface.shape[0]}")

        if sharp_surface is None:
            ret = {
                "surface": surface
            }
        else:
            ret = {
                "surface": surface,
                "sharp_surface": sharp_surface
            }

        return ret

    def _load_shape_supervision_occupancy_or_sdf(self, data) -> Dict[str, Any]:
        # for supervision
        ret = {}
        if self.cfg.geo_data_type == "occupancy":
            rand_points = np.asarray(data['points']) * 2 # range from -1.1 to 1.1
            occupancies = np.asarray(data['occupancies'])
            occupancies = np.unpackbits(occupancies)
        elif self.cfg.geo_data_type == "sdf":
            rand_points = data['rand_points']
            sdfs = data['sdfs']
        else:
            raise NotImplementedError(f"Data type {self.cfg.geo_data_type} not implemented")

        # random sampling
        rng = np.random.default_rng()
        ind = rng.choice(rand_points.shape[0], self.cfg.n_supervision, replace=False)
        rand_points = rand_points[ind]
        rand_points = rand_points * self.cfg.scale
        ret["rand_points"] = rand_points.astype(np.float32)

        if self.cfg.geo_data_type == "occupancy":
            assert self.cfg.supervision_type == "occupancy", "Only occupancy supervision is supported for occupancy data"
            occupancies = occupancies[ind]
            ret["occupancies"] = occupancies.astype(np.float32)
        elif self.cfg.geo_data_type == "sdf":
            if self.cfg.supervision_type == "sdf":
                ret["sdf"] = sdfs[ind].flatten().astype(np.float32)
            elif self.cfg.supervision_type == "occupancy":
                ret["occupancies"] = np.where(sdfs[ind].flatten() < 1e-3, 0, 1).astype(np.float32)
            elif self.cfg.supervision_type == "tsdf":
                ret["sdf"] = sdfs[ind].flatten().astype(np.float32).clip(-self.cfg.tsdf_threshold, self.cfg.tsdf_threshold) / self.cfg.tsdf_threshold
            else:
                raise NotImplementedError(f"Supervision type {self.cfg.supervision_type} not implemented")

        return ret

    def _load_caption(self, data) -> Dict[str, Any]:
        symmetry_label = {
            "asymmetry": "asymmetry",
            "x": "symmetry-x",
            "y": "symmetry-y",
            "z": "symmetry-z"
        }
        caption = f"{data['caption']}; {symmetry_label[data['label']['symmetry']]}; {data['label']['edge_type']}"
        return {"caption": caption}

    def _load_image(self, data) -> Dict[str, Any]:
        def process_img(image, background_color=(255, 255, 255), foreground_ratio=0.9):
            alpha = image.getchannel('A') 
            background = Image.new("RGBA", image.size, (*background_color, 255))
            image = Image.alpha_composite(background, image)
            image = image.crop(alpha.getbbox())

            new_size = tuple(int(dim * foreground_ratio) for dim in image.size)
            resized_image = image.resize(new_size)
            padded_image = Image.new("RGBA", image.size, (*background_color, 255))
            paste_position = ((image.width - resized_image.width) // 2, 
                            (image.height - resized_image.height) // 2)
            padded_image.paste(resized_image, paste_position)

            # Expand image to 1:1
            max_dim = max(padded_image.size)
            image = Image.new("RGBA", (max_dim, max_dim), (*background_color, 255))
            paste_position = ((max_dim - padded_image.width) // 2, 
                            (max_dim - padded_image.height) // 2)
            image.paste(padded_image, paste_position)
            image = image.resize((512, 512))
            return image.convert("RGB"), alpha

        ret = {}
        if self.cfg.image_type == "rgb" or self.cfg.image_type == "normal" or self.cfg.image_type == "rgb_or_normal":
            image_type_options = ['rgb', 'normal']
            assert self.cfg.n_views == 1, "Only single view is supported for single image"

            sel_idxs = (
                self.cfg.idx if self.cfg.images_per_sample == -1 
                else np.random.default_rng().choice(self.cfg.idx, self.cfg.images_per_sample, replace=False)
            )
            ret["sel_image_idxs"] = np.array(sel_idxs)

            imgs, alphas = [], []
            for idx in sel_idxs:
                if self.cfg.image_type == "rgb_or_normal":
                    assert 0.0 <= self.cfg.image_type_ratio <= 1.0, "image_type_ratio should in [0.0, 1.0]"
                    sel_image_type = np.random.choice(image_type_options, p=[self.cfg.image_type_ratio, 1.0 - self.cfg.image_type_ratio])
                    if sel_image_type == "rgb":
                        image = Image.open(BytesIO(data[f"{str(idx).zfill(4)}_rgb"])).copy()
                    else:
                        image = Image.open(BytesIO(data[f"{str(idx).zfill(4)}_normal"])).copy()
                        alpha = Image.open(BytesIO(data[f"{str(idx).zfill(4)}_rgb"])).copy().split()[-1]
                        image = Image.merge("RGBA", (image.split()[0], image.split()[1], image.split()[2], alpha))
                else:
                    image = Image.open(BytesIO(data[f"{str(idx).zfill(4)}_{self.cfg.image_type}"])).copy()
                    
                # add random color jitter
                if self.cfg.random_color_jitter:
                    rgb = self.transforms(image.convert("RGB"))
                    image = Image.merge("RGBA", (*rgb.split(), image.getchannel('A')))

                image, alpha = process_img(image, self.cfg.background_color, self.cfg.foreground_ratio)
                imgs.append(torch.from_numpy(np.array(image) / 255))
                alphas.append(torch.from_numpy(np.array(alpha) / 255))

                ret["image"] = torch.stack(imgs)
                ret["mask"] = torch.stack(alphas)

        elif self.cfg.image_type == "mvrgb" or self.cfg.image_type == "mvnormal":
            raise NotImplementedError(f"Image type {self.cfg.image_type} not implemented")
        else:
            raise NotImplementedError(f"Image type {self.cfg.image_type} not implemented")
        
        return ret

    def get_item(self, index: int) -> Any:
        data = super().get_item(index)
        try:
            ret = {"uid": data["uid"].split("/")[-1].replace(".glb", "")}

            if self.cfg.random_flip:
                flip = np.random.rand() < 0.5
            else:
                flip = False

            # load geometry
            if self.cfg.load_geometry:
                if self.cfg.geo_data_type == "occupancy" or self.cfg.geo_data_type == "sdf":
                    ret.update(self._load_shape_from_occupancy_or_sdf(data))
                    if self.cfg.load_supervision:
                        ret.update(self._load_shape_supervision_occupancy_or_sdf(data))

                    if flip: # random flip the input point cloud and the supervision
                        for key in ret.keys():
                            if key in ["surface", "sharp_surface"]: # N x (xyz + normal)
                                ret[key][:, 0] = -ret[key][:, 0]
                                ret[key][:, 3] = -ret[key][:, 3]
                            elif key in ["rand_points"]:
                                ret[key][:, 0] = -ret[key][:, 0]
                else:
                    raise NotImplementedError(f"Geo data type {self.cfg.geo_data_type} not implemented")

            # load image
            if self.cfg.load_image:
                ret.update(self._load_image(data))

                if flip: # random flip the input image
                    for key in ret.keys():
                        if key in ["image"]: # random flip the input image
                            ret[key] = torch.flip(ret[key], [2])
                        if key in ["mask"]: # random flip the input image
                            ret[key] = torch.flip(ret[key], [2])
                
            # load caption
            if self.cfg.load_caption:
                ret.update(self._load_caption(data))

            return ret
        except Exception as e:
            print(f"Error in {data['uid']}: {e}")
            return self.get_item(np.random.randint(len(self)))
    
    def collate(self, batch):
        from torch.utils.data._utils.collate import default_collate_fn_map
        ret = {}
        for key, value in batch[0].items():
            if isinstance(value, str):
                ret[key] = [b[key] for b in batch]
            elif isinstance(value, torch.Tensor):
                ret[key] = torch.stack([b[key] for b in batch])
            elif isinstance(value, np.ndarray):
                ret[key] = torch.stack([torch.from_numpy(b[key]) for b in batch])
            else:
                ret[key] = default_collate_fn_map[type(batch[0][key])](batch)
        return ret
