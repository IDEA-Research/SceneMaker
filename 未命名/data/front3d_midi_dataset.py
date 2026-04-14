# Dataloader of InstPIFu.
# author: LiuHaolin
# date: Aug, 2022
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch.utils.data
from torchvision import transforms
import pickle
from PIL import Image
import numpy as np
import json
import random
from craftsman.data.net_utils.bins import *
import glob
from tqdm import tqdm
from utils.space_transform import get_rot_from_pitch, get_pitch_from_R, q2rot, get_yaw_from_R, get_rot_from_yaw
from utils.transforms import *
import trimesh
import fpsample
import rembg
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from craftsman import register
from craftsman.utils.typing import *
from craftsman.utils.config import parse_structured

import math
import re
import cv2
from dataclasses import dataclass, field

import random
import imageio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from craftsman.utils.typing import *
from .base_s3 import BaseS3DataModuleConfig

import pdb

category_label_mapping = {"table": 0,
                          "sofa": 1,
                          "cabinet": 2,
                          "night_stand": 3,
                          "chair": 4,
                          "bookshelf": 5,
                          "bed": 6,
                          "desk": 7,
                          "dresser": 8,
                          "default": 9
                          }
inv_category_mapping = {v: k for k, v in category_label_mapping.items()}


data_transforms_patch = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms_mask = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
])


def _parse_object_list_single(object_list_path: str):
    all_objects = []
    if object_list_path.endswith(".json"):
        with open(object_list_path) as f:
            all_objects = json.loads(f.read())
    else:
        raise NotImplementedError

    return all_objects


def _parse_object_list(object_list_path: Union[str, List[str]]):
    all_objects = []
    if isinstance(object_list_path, str):
        object_list_path = [object_list_path]
    for object_list_path_ in object_list_path:
        all_objects += _parse_object_list_single(object_list_path_)
    return all_objects


def _parse_scene_list_single(scene_list_path: str, root_data_dir: str):
    all_scenes = []
    if scene_list_path.endswith(".json"):
        with open(scene_list_path) as f:
            for p in json.loads(f.read()):
                all_scenes.append(os.path.join(root_data_dir, p))
    elif scene_list_path.endswith(".txt"):
        with open(scene_list_path) as f:
            for p in f.readlines():
                p = p.strip()
                all_scenes.append(os.path.join(root_data_dir, p))
    else:
        raise NotImplementedError

    return all_scenes


def _parse_scene_list(
    scene_list_path: Union[str, List[str]], root_data_dir: Union[str, List[str]]
):
    all_scenes = []
    if isinstance(scene_list_path, str):
        scene_list_path = [scene_list_path]
    if isinstance(root_data_dir, str):
        root_data_dir = [root_data_dir]
    for scene_list_path_, root_data_dir_ in zip(scene_list_path, root_data_dir):
        all_scenes += _parse_scene_list_single(scene_list_path_, root_data_dir_)
    return all_scenes



@dataclass
class Front3DMIDIDataModuleConfig(BaseS3DataModuleConfig):    
    ################################# Scene #################################
    scene_list: str = "/comp_robot/shiyukai/datasets/midi/3D-Front/midi_room_ids.json"
    object_list: str = "/comp_robot/shiyukai/datasets/midi/3D-Front/midi_furniture_ids.json"
    surface_root_dir: str = "/comp_robot/shiyukai/datasets/midi/3D-Front/3D-FRONT-SURFACE/"
    image_data_path: str = "/comp_robot/shiyukai/datasets/midi/3D-Front/3D-FRONT-RENDER/"
    mask_path: str = "/comp_robot/shiyukai/datasets/midi/3D-Front/3D-FRONT-RENDER-mask/"
    geo_data_path: str = "/comp_robot/shiyukai/datasets/instPiFU/datasets/normalized_watertight/sampling_objects"
    # deocc_image_path: str = '/comp_robot/shiyukai/datasets/instPiFU/datasets/render_obj_scene_img/'
    
    train_indices: Optional[Tuple[Any, Any]] = None
    val_indices: Optional[Tuple[Any, Any]] = None
    test_indices: Optional[Tuple[Any, Any]] = None
    
    ################################# Scene argumentation #################################
    max_objs: int = 5
    min_pcd: int = 1024
    translation_mode: str = "pcd2"
    refine_mask: bool = False
    num_scene: int = -1
    add_test: bool = False  # whether to add val set to train set
    use_scene_geometry: bool = False  # whether to use scene geometry data, if False, use Objaverse-MIX data
    use_deocc_image: bool = False  # whether to use deocc_image, if False, use original image
    use_wrd_pose: bool = False # whether to use world transformation matrix, if False, use camera transformation matrix
    num_views: int = 11
    only_use_pitch: bool = False # whether to only use pitch rotation
    use_mix_coord: bool = False # whether to use mix data coord, -z + opengl as canonical scene space
    render_mode: str = "render" # render, rerender, both

    ################################# General argumentation #################################
    random_flip: bool = False            # whether to randomly flip the input point cloud and the input images
    random_color_jitter: bool = False    # whether to randomly color jitter the input images
    image_width: int = 768
    image_height: int = 768
    image_batchfy_mode: str = "resize"

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

class Front3DMIDIDataset(Dataset):
    def __init__(self, config, mode):
        super(Front3DMIDIDataset, self).__init__()
        self.mode=mode
        self.cfg: Front3DMIDIDataModuleConfig = config
        assert mode in ["train", "val", "test"]

        self.all_scenes = _parse_scene_list(self.cfg.scene_list, self.cfg.mask_path)
        # self.all_objects = _parse_object_list(self.cfg.object_list)
        # if len(self.all_scenes) != len(self.all_objects):
        #     raise ValueError(
        #         f"Number of scenes and objects must be the same, got {len(self.all_scenes)} scenes and {len(self.all_objects)} object lists."
        #     )
        # self.all_images = _parse_scene_list(self.cfg.scene_list, self.cfg.image_root_dir)

        self.splits = []
        if self.mode == "train" and self.cfg.train_indices is not None:
            self.splits = (self.cfg.train_indices[0], self.cfg.train_indices[1])
        elif self.mode == "val" and self.cfg.val_indices is not None:
            self.splits = (self.cfg.val_indices[0], self.cfg.val_indices[1])
            # self.splits = (self.cfg.val_indices[0], self.cfg.val_indices[0]+10)
        elif self.mode == "test" and self.cfg.test_indices is not None:
            # self.splits = (self.cfg.test_indices[0], self.cfg.test_indices[1])
            self.splits = (self.cfg.test_indices[0], self.cfg.test_indices[0]+10)
        else:
            self.splits = (0, len(self.all_scenes))

        self.all_scenes = self.all_scenes[self.splits[0] : self.splits[1]]
        # self.all_objects = self.all_objects[self.splits[0] : self.splits[1]]
        # self.all_images = self.all_images[self.splits[0] : self.splits[1]]

    def __len__(self):
        return len(self.all_scenes)
    
    def _load_shape_from_occupancy_or_sdf(self, surface_path, scale=1) -> Dict[str, Any]:
        # for supervision
        data = np.load(surface_path)
        ret = {}
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
        
        if len(surface.shape) == 3:
            surface = surface[:,0]
            if self.cfg.with_sharp_data:
                sharp_surface = sharp_surface[:,0]
            
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
        surface[:, :3] = surface[:, :3] * scale # target scale
        # add noise to input point cloud
        surface[:, :3] += (np.random.rand(surface.shape[0], 3) * 2 - 1) * self.cfg.noise_sigma
        surface = surface.astype(np.float32)
        if self.cfg.with_sharp_data:
            sharp_surface[:, :3] = sharp_surface[:, :3] * scale # target scale
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
    
    def resize_and_pad(self, image, masks, target_height, target_width):
        """
        Resize proportionally so one side reaches the target size and the other remains smaller, then pad to target size.
        image: torch.Tensor, [H, W, 3]
        masks: torch.Tensor, [N, H, W]
        target_height, target_width: int
        返回: image [target_height, target_width, 3], masks [N, target_height, target_width]
        """
        # 获取原始尺寸
        orig_height, orig_width = image.shape[0], image.shape[1]
        scale = min(target_height / orig_height, target_width / orig_width)
        new_height = int(orig_height * scale)
        new_width = int(orig_width * scale)

        # 缩放
        image_resized = F.interpolate(image.permute(2, 0, 1).unsqueeze(0), size=(new_height, new_width), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0)
        masks_resized = F.interpolate(masks.unsqueeze(1).float(), size=(new_height, new_width), mode="nearest").squeeze(1)

        # 计算padding
        pad_h = target_height - new_height
        pad_w = target_width - new_width
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # image用白色padding
        image_padded = F.pad(image_resized.permute(2, 0, 1), (pad_left, pad_right, pad_top, pad_bottom), value=1.0).permute(1, 2, 0)
        # mask用0 padding
        masks_padded = F.pad(masks_resized, (pad_left, pad_right, pad_top, pad_bottom), value=0)
        
        # viz image and mask
        # Image.fromarray((image_padded.numpy() * 255).astype(np.uint8)).save("outputs/viz/debug_image.png")
        # Image.fromarray((masks_padded[0,:,:,None].repeat(1,1,3).numpy() * 255).astype(np.uint8)).save("outputs/viz/debug_mask.png")

        return image_padded, masks_padded

    def reconstruct_pcd(self, depth, intrinsic, scale=1, normalize=False, valid_mask=None):
        # apply scale on intrinsic
        intrinsic = intrinsic.copy()
        intrinsic[0, 0] *= scale
        intrinsic[1, 1] *= scale
        intrinsic[0, 2] *= scale
        intrinsic[1, 2] *= scale
        
        if valid_mask is not None:
            valid_Y, valid_X = np.where((depth > 0) & (depth < 10) & valid_mask)
        else:
            valid_Y, valid_X = np.where((depth > 0) & (depth < 10))

        max_num_points = 100000
        if valid_Y.shape[0] > max_num_points:
            random_ind = np.random.choice(valid_Y.shape[0], max_num_points, replace=False)
            valid_Y = valid_Y[random_ind]
            valid_X = valid_X[random_ind]

        unprojected_X = valid_X
        unprojected_Y = valid_Y
        unprojected_Z = np.ones_like(unprojected_X)
        point_cloud_xyz = np.stack([unprojected_X, unprojected_Y, unprojected_Z], axis=1)

        intrinsic_inv = np.linalg.inv(intrinsic[0:3, 0:3])
        point_cloud_incam = (intrinsic_inv @ point_cloud_xyz.T).T * depth[valid_Y, valid_X][:,None]
        if self.cfg.use_mix_coord:
            point_cloud_incam[:, :2] = -point_cloud_incam[:, :2].copy()
        else:
            point_cloud_incam[:, 1:3] = -point_cloud_incam[:, 1:3].copy()

        if normalize:
            translation = (np.max(point_cloud_incam, axis=0) + np.min(point_cloud_incam, axis=0)) / 2
            point_cloud_incam = point_cloud_incam - translation
            scale = np.max(np.abs(point_cloud_incam[:, :2]))
            point_cloud_incam = point_cloud_incam / scale * 0.95
            return point_cloud_incam, translation, scale
        else:
            return point_cloud_incam
    
    def refine_instance_mask(self, image: np.ndarray, instance_mask: np.ndarray):
        # Step 1: Get bounding box of the instance mask
        ys, xs = np.where(instance_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return torch.from_numpy(instance_mask)  # skip empty mask

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # Step 2: Crop image and mask
        cropped_image = image[y_min:y_max+1, x_min:x_max+1]
        cropped_pil = Image.fromarray(cropped_image).convert("RGBA")

        # Step 3: Apply rembg to get a new alpha mask
        removed = rembg.remove(cropped_pil)
        alpha = np.array(removed.split()[-1])  # get alpha channel as mask
        new_mask = (alpha > 128).astype(np.uint8)  # binarize

        # Step 4: Paste the new mask back to original image mask size
        refined_mask = np.zeros_like(instance_mask, dtype=np.uint8)
        refined_mask[y_min:y_max+1, x_min:x_max+1] = new_mask

        return refined_mask*255
    
    def dilate_mask(self, mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
        """
        Dilate a binary mask.

        Args:
            mask (np.ndarray): Input mask, shape (H, W), values in [0, 255] or [0.0, 1.0].
            kernel_size (int): Size of the square kernel for dilation.
            iterations (int): Number of times dilation is applied.

        Returns:
            np.ndarray: Dilated mask (same dtype as input).
        """
        # Ensure mask is uint8 and values are 0 or 255
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        mask_bin = (mask > 0).astype(np.uint8) * 255

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(mask_bin, kernel, iterations=iterations)

        return dilated
    

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
    def crop_image(self, image, mask, bbox_ratio=0.9, pad_color=(255, 255, 255)):    
        '''
        transform image to 512x512, crop the image with mask
        Input:
            image: numpy array
            mask: numpy array
        Output:
            image: numpy array
            mask: numpy array
        '''
        from PIL import Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)

        coordinates = mask.getbbox()
        if coordinates is None:
            # 无前景，返回原图和全零mask
            image = image.resize((512, 512), Image.BICUBIC)
            mask = mask.resize((512, 512), Image.NEAREST)
            image = np.array(image)
            mask = np.array(mask)
            return image, mask

        left, upper, right, lower = coordinates
        width = right - left
        height = lower - upper
        side = max(width, height)
        crop_side = int(side / bbox_ratio)
        center_x = (left + right) // 2
        center_y = (upper + lower) // 2

        new_left = center_x - crop_side // 2
        new_upper = center_y - crop_side // 2
        new_right = new_left + crop_side
        new_lower = new_upper + crop_side

        pad_left = max(0, -new_left)
        pad_upper = max(0, -new_upper)
        pad_right = max(0, new_right - image.width)
        pad_lower = max(0, new_lower - image.height)

        crop_box = (
            max(new_left, 0),
            max(new_upper, 0),
            min(new_right, image.width),
            min(new_lower, image.height)
        )
        cropped = image.crop(crop_box)
        cropped_mask = mask.crop(crop_box)

        if any([pad_left, pad_upper, pad_right, pad_lower]):
            new_img = Image.new(image.mode, (crop_side, crop_side), pad_color)
            new_img.paste(cropped, (pad_left, pad_upper))
            new_mask = Image.new(mask.mode, (crop_side, crop_side), 0)
            new_mask.paste(cropped_mask, (pad_left, pad_upper))
            # resize
            new_img = new_img.resize((512, 512), Image.BICUBIC)
            new_mask = new_mask.resize((512, 512), Image.NEAREST)
            # transfer to numpy
            new_img = np.array(new_img)
            new_mask = np.array(new_mask)
            return new_img, new_mask
        else:
            cropped = cropped.resize((512, 512), Image.BICUBIC)
            cropped_mask = cropped_mask.resize((512, 512), Image.NEAREST)
            cropped = np.array(cropped)
            cropped_mask = np.array(cropped_mask)
            return cropped, cropped_mask

    
    def recover_transform_matrix(self, rotation_6d, translation):
        """
        Recover 4x4 transformation matrix from 6D rotation representation and 3D translation.
        
        Args:
            rotation_6d: Tensor of shape (..., 6), representing 6D rotation.
            translation: Tensor of shape (..., 3), representing 3D translation.
        
        Returns:
            transformation: Tensor of shape (..., 4, 4)
        """
        # r1 = self.normalize_vector(rotation_6d[..., :3])
        # r2 = self.normalize_vector(rotation_6d[..., 3:6] - torch.sum(r1 * rotation_6d[..., 3:6], dim=-1, keepdim=True) * r1)
        # r3 = torch.cross(r1, r2, dim=-1)
        
        # R = torch.stack([r1, r2, r3], dim=-1)  # (..., 3, 3)
        
        R = repr6d2mat(rotation_6d)  # (..., 3, 3)
        T = torch.eye(4, device=rotation_6d.device).expand(*rotation_6d.shape[:-1], 4, 4).clone()
        T[..., :3, :3] = R
        T[..., :3, 3] = translation
        
        return T

    def augment_surface_with_random_y_rotation(self, ret, angle_list=None):
        """
        对 ret["surface"] 的点云和 normal 绕 y 轴旋转
        如果 angle_list 不为空，则从中随机选取一个角度（单位：弧度），否则随机采样 [0, 2π)
        """
        surface = ret["surface"].copy()
        # 选择旋转角度
        if angle_list is not None and len(angle_list) > 0:
            theta = np.random.choice(angle_list)
        else:
            theta = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        # 绕y轴旋转矩阵
        R = np.array([
            [cos_theta, 0, sin_theta],
            [0,        1, 0],
            [-sin_theta, 0, cos_theta]
        ], dtype=np.float32)
        # 点云坐标旋转
        surface[:, :3] = surface[:, :3] @ R.T
        # normal 旋转
        surface[:, 3:6] = surface[:, 3:6] @ R.T
        ret["surface"] = surface

        # sharp surface rotation
        if ret.get("sharp_surface") is not None:
            sharp_surface = ret["sharp_surface"].copy()
            sharp_surface[:, :3] = sharp_surface[:, :3] @ R.T
            sharp_surface[:, 3:6] = sharp_surface[:, 3:6] @ R.T
            ret["sharp_surface"] = sharp_surface
        return ret, R

    def rotate_transformation_y_180(self, trans_matrix):
        """
        输入4x4 transformation matrix，返回绕y轴旋转180度后的transformation matrix
        """
        # 构造绕y轴旋转180度的旋转矩阵
        R_y = np.array([
            [-1, 0,  0],
            [ 0, 1,  0],
            [ 0, 0, -1]
        ], dtype=trans_matrix.dtype)
        # 新的旋转部分
        new_rot = R_y @ trans_matrix[:3, :3]
        # 新的transformation matrix
        new_trans_matrix = trans_matrix.copy()
        new_trans_matrix[:3, :3] = new_rot
        new_trans_matrix[:3, 3] = R_y @ trans_matrix[:3, 3]
        return new_trans_matrix

    def normalize_vector(self, v):
        return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)
    
    
    def alpha_to_white(self, image_rgba):
        """
        将 RGBA 图像中 alpha=0 的部分改为白色
        Args:
            image_rgba: numpy array, shape (H, W, 4), dtype uint8 or float
        Returns:
            image_rgb: numpy array, shape (H, W, 3)
        """
        img = image_rgba.copy()
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        alpha = img[..., 3]
        mask = (alpha == 0)
        img[..., :3][mask] = 255  # 白色
        return img[..., :3]

    def __getitem_scene(self, index):
        # load scene info
        scene = self.all_scenes[index]
        scene_id = os.path.basename(os.path.dirname(scene))
        room_id = os.path.basename(scene)
        scene_data_path = os.path.join(self.cfg.mask_path, scene_id, room_id)
        scene_info = np.load(os.path.join(scene_data_path, "scene_info.npy"), allow_pickle=True).item()
        
        # camera pose
        cam_id = random.randint(0, self.cfg.num_views-1) if self.mode == "train" else index%3
        cam_info = np.load(os.path.join(scene_data_path, f"camera_{cam_id:04d}/camera_info.npy"), allow_pickle=True).item()
        camera_K, cam2wrd = cam_info["camera_intrinsics"], cam_info["camera_extrinsics"]
        if self.cfg.only_use_pitch:
            cam2wrd[:3,:3] = get_rot_from_yaw(get_yaw_from_R(cam2wrd[:3, :3]))
        
        # load image
        if self.cfg.render_mode == "rerender":
            image_path = os.path.join(self.cfg.image_data_path, scene_id, room_id, f"rerender_{cam_id:04d}.webp") # augmentation data
            image = np.array(Image.open(image_path))
        elif self.cfg.render_mode == "render":
            image_path = os.path.join(self.cfg.image_data_path, scene_id, room_id, f"render_{cam_id:04d}.webp")
            image = np.array(Image.open(image_path))
            image = self.alpha_to_white(image)
        elif self.cfg.render_mode == "both":
            if random.random() < 0.5:
                image_path = os.path.join(self.cfg.image_data_path, scene_id, room_id, f"rerender_{cam_id:04d}.webp") # augmentation data
                image = np.array(Image.open(image_path))
            else:
                image_path = os.path.join(self.cfg.image_data_path, scene_id, room_id, f"render_{cam_id:04d}.webp")
                image = np.array(Image.open(image_path))
                image = self.alpha_to_white(image)
        height, width = image.shape[0], image.shape[1]
        # Image.fromarray(image).save("outputs/viz/scene_img.png")

        # depth
        depth_path = os.path.join(self.cfg.image_data_path, scene_id, room_id, f"depth_{cam_id:04d}.exr")
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)[:,:,0]
        # rescale depth
        depth_scale = (width / depth.shape[0] + height / depth.shape[1]) / 2
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST) # resize depth
        # reconstruct scene
        point_cloud_incam, pcd_scene_trans, pcd_scene_scale = self.reconstruct_pcd(depth=depth, intrinsic=camera_K, scale=depth_scale, normalize=True)
        # trimesh.points.PointCloud(point_cloud_incam, colors=[255, 0, 0]).export(f"outputs/viz/pcd/scene_pcd.ply")

        # load object information
        n_objects = len(scene_info['furniture'])
        geo_shape, geo_shape_sharp = [], []
        crop_imgs, masks = [], []
        obj_pcds, scene_pcds, scene_surface = [], [], []
        bbox_sizes = []
        pcd_sizes, pcd_trans = [], []
        transform_cam_list = []
        class_list = []
        mask_legal = torch.zeros(self.cfg.max_objs) # mask for legal object
        miss_obj = False # whether there is missing object
        for object_ind in range(n_objects):
            if object_ind >= self.cfg.max_objs:
                break
            
            # load furniture info
            furniture_info = scene_info['furniture'][object_ind]
            jid = furniture_info['model_id'] # object index
            class_list.append(furniture_info['category']) # category
            
            # load transformation
            tran_matrix = furniture_info['transformation']
            tran_matrix_cam = tran_matrix if self.cfg.use_wrd_pose else np.linalg.inv(cam2wrd) @ tran_matrix
            if self.cfg.use_mix_coord:
                tran_matrix_cam = self.rotate_transformation_y_180(tran_matrix_cam)
            transform_cam_list.append(tran_matrix_cam)
            bbox_size = furniture_info['scale'] * 2
            bbox_sizes.append(bbox_size.max()*np.ones_like(bbox_size))

            # load object shape
            surface_path = f'{self.cfg.geo_data_path}/{jid}/normalized_watertight.npz'
            size = bbox_size / bbox_size.max()
            ret = self._load_shape_from_occupancy_or_sdf(surface_path=surface_path, scale=size)
            if "pcd2_aug" in self.cfg.translation_mode and self.mode == "train":
                # trimesh.PointCloud(ret["surface"][:,:3]).export(f"outputs/viz/pts/pointcloud_{object_ind}_before.ply") # viz pts before augmentation
                if "pcd2_aug_pi" in self.cfg.translation_mode:
                    ret, aug_rot = self.augment_surface_with_random_y_rotation(ret, angle_list=[0, np.pi/2, np.pi, 3*np.pi/2]) # apply rotation augmentation
                else:
                    ret, aug_rot = self.augment_surface_with_random_y_rotation(ret) # apply random rotation
                # trimesh.PointCloud(ret["surface"][:,:3]).export(f"outputs/viz/pts/pointcloud_{object_ind}_after.ply") # viz pts after augmentation
                tran_matrix_cam[:3, :3] = tran_matrix_cam[:3, :3] @ aug_rot.T # apply augmentation to tran_matrix_cam
            geo_shape.append(torch.from_numpy(ret["surface"]))
            geo_shape_sharp.append(torch.from_numpy(ret["sharp_surface"]))
            # gt surface in scene space
            scene_pts_gt = ((ret["surface"][:,:3] * bbox_size.max()/2) @ tran_matrix_cam[:3,:3].T) + tran_matrix_cam[:3,3]
            scene_surface.append(scene_pts_gt)
            
            # mask
            instance_mask_path = os.path.join(scene_data_path, f"camera_{cam_id:04d}", f"mask_{object_ind:04d}.png")
            if os.path.exists(instance_mask_path):
                instance_mask = np.array(Image.open(instance_mask_path))
                instance_mask = cv2.resize(instance_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            else:
                instance_mask = np.zeros((height, width), dtype=np.uint8)
            masks.append(torch.from_numpy(instance_mask)/255.0)
        
            # crop from scene image
            rgb, cropped_mask = self.crop_image(image=image, mask=instance_mask, bbox_ratio=0.8)
            white_bg = np.ones_like(rgb) * 255 # apply mask with white background
            rgb = rgb * (cropped_mask[..., None].astype(np.float32)/255) + white_bg * (1 - cropped_mask[..., None].astype(np.float32)/255) 
            crop_imgs.append(torch.from_numpy(rgb/255.0))
            
            # save rgb, cropped_mask and label under folder named with taskid
            # self.save_crop_and_mask(rgb, cropped_mask, furniture_info["category"], room_id, object_ind, save_dir="outputs/viz/deocc_data")
            
            # object pcd
            try:
                # full pcd
                if "full_pcd2" in self.cfg.translation_mode:
                    # downsample geo shape to min_pcd
                    if ret["surface"].shape[0] > self.cfg.min_pcd:
                        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(ret["surface"][:, :3], self.cfg.min_pcd, h=5)
                        full_obj_pcd = ret["surface"][kdline_fps_samples_idx,:3]
                    else:
                        full_obj_pcd = ret["surface"][:,:3]
                    # apply transformation and scale
                    scene_obj_pcd = full_obj_pcd * bbox_size.max() / 2.0  # scale to [0, 1]
                    scene_obj_pcd = (scene_obj_pcd @ tran_matrix_cam[:3, :3].T) + tran_matrix_cam[:3, 3]
                    # normalize to [-1, 1]
                    pcd_tran = (np.max(scene_obj_pcd, axis=0) + np.min(scene_obj_pcd, axis=0)) / 2
                    obj_pcd_cam = scene_obj_pcd - pcd_tran
                    pcd_size = np.max(np.abs(obj_pcd_cam[:, :2]))
                    # set default values if pcd_size is zero
                    if pcd_size == 0:
                        obj_pcd_cam = torch.zeros(self.cfg.min_pcd, 3)
                        pcd_tran = np.zeros(3)
                        pcd_size = 1e-4
                    else:
                        obj_pcd_cam = obj_pcd_cam / pcd_size
                        mask_legal[object_ind] = 1  # full pcd always legal
                    # # viz
                    # trimesh.PointCloud(full_obj_pcd).export(f"outputs/viz/pts/full_obj_pcd.ply")
                    # trimesh.PointCloud(scene_obj_pcd).export(f"outputs/viz/pts/scene_obj_pcd.ply")
                    # trimesh.PointCloud(obj_pcd_cam).export(f"outputs/viz/pcd/obj_pcd_cam.ply")
                    
                # project from mask
                else:
                    # mask = ~self.dilate_mask(~instance_mask, kernel_size=3, iterations=1) # dialate mask
                    obj_pcd_cam, pcd_tran, pcd_size = self.reconstruct_pcd(depth=depth, intrinsic=camera_K, scale=depth_scale, normalize=True, valid_mask=instance_mask)
            except:
                obj_pcd_cam = torch.zeros(self.cfg.min_pcd, 3)
                pcd_tran = np.zeros(3)
                pcd_size = 1e-4
            
            if "full_pcd" in self.cfg.translation_mode:
                pass
            elif obj_pcd_cam.shape[0] <= self.cfg.min_pcd:
                obj_pcd_cam = torch.zeros(self.cfg.min_pcd, 3)
            elif obj_pcd_cam.shape[0] > self.cfg.min_pcd:
                kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(obj_pcd_cam[:, :3], self.cfg.min_pcd, h=5)
                obj_pcd_cam = obj_pcd_cam[kdline_fps_samples_idx]
                mask_legal[object_ind] = 1

            pcd_sizes.append(pcd_size)
            pcd_trans.append(pcd_tran)

            obj_pcds.append(obj_pcd_cam)
            scene_pcds.append(obj_pcd_cam*pcd_size+pcd_tran)

            # # viz
            # points = ret["surface"][:,:3]
            # points_scaled = points / 2 * bbox_size
            # points_transformed = (points_scaled @ tran_matrix_cam[:3, :3].T) + tran_matrix_cam[:3, 3]
            # trimesh.PointCloud(points_scaled).export(f"outputs/viz/pts/pointcloud_{object_ind}.ply")
            # trimesh.points.PointCloud(obj_pcd_cam*pcd_size+pcd_tran, colors=[255, 0, 0]).export(f"outputs/viz/depth_pcd/depth_pcd_{object_ind}.ply")
            
            # # viz crop image
            # Image.fromarray((crop_imgs[-1].numpy()*255).astype(np.uint8)).save(f"outputs/viz/crop/crop_{object_ind}.png")
            # Image.fromarray((instance_mask).astype(np.uint8)).save(f"outputs/viz/crop/mask_{object_ind}.png")            

        # gt scene params
        scene_surface = np.concatenate(scene_surface, axis=0)
        centroid = (np.max(scene_surface, axis=0) + np.min(scene_surface, axis=0)) / 2
        layout_size = np.max(np.abs(scene_surface-centroid))
        
        # scene level params
        tran_matrix_cam = np.stack(transform_cam_list) if len(transform_cam_list) > 0 else np.zeros((1,4,4))           
        bbox_sizes = np.stack(bbox_sizes) / layout_size * 0.95# object size [0,2]
        translation = (tran_matrix_cam[:bbox_sizes.shape[0],:3,3]-centroid) / layout_size * 0.95 # translation
        rotation = mat2repr6d(tran_matrix_cam[:bbox_sizes.shape[0],:3,:3]) # rotation

        # reference translation
        scene_pcds = np.stack(scene_pcds, axis=0)
        mask_index = torch.nonzero(mask_legal).numpy().squeeze(-1)
        # normalize scene pcd with pcd min-max
        if self.cfg.translation_mode == "pcd" or "pcd_aug" in self.cfg.translation_mode:
            # normalize in whole scene
            if mask_index.shape[0] > 0:
                # scene_pcds[mask_index] = (scene_pcds[mask_index] - pcd_scene_trans) / pcd_scene_scale * 0.95
                scene_pcds = (scene_pcds - pcd_scene_trans) / pcd_scene_scale * 0.95
            # calculate pcd size of objects
            pcd_trans = np.stack(pcd_trans, axis=0)
            pcd_trans = (pcd_trans - pcd_scene_trans) / pcd_scene_scale * 0.95
            pcd_sizes = np.stack(pcd_sizes, axis=0)
            pcd_sizes = pcd_sizes / pcd_scene_scale * 0.95
            pcd_sizes = np.repeat(pcd_sizes[:, np.newaxis], repeats=3, axis=-1)
            # # viz scene pcd
            # trimesh.points.PointCloud(scene_pcds.reshape(-1,3), colors=[255, 0, 0]).export(f"outputs/viz/pcd/depth_pts_scene_wrd.ply")
            
        elif self.cfg.translation_mode == "pcd2" or "full_pcd2" in self.cfg.translation_mode or "pcd2_aug" in self.cfg.translation_mode:
            # normalize in whole scene
            pcd_trans = np.stack(pcd_trans, axis=0)
            pcd_sizes = np.stack(pcd_sizes, axis=0)
            if mask_index.shape[0] > 0:
                # modify active object pcd to fit in the scene
                scene_pcds[mask_index] = (scene_pcds[mask_index] - centroid) / layout_size * 0.95
                pcd_trans[mask_index] = (pcd_trans[mask_index] - centroid) / layout_size * 0.95
                pcd_sizes[mask_index] = pcd_sizes[mask_index] / layout_size * 0.95

            pcd_sizes = np.repeat(pcd_sizes[:, np.newaxis], repeats=3, axis=-1)
            # viz scene pcd
            # trimesh.points.PointCloud(scene_pcds.reshape(-1,3), colors=[255, 0, 0]).export(f"outputs/viz/pcd/depth_pts_scene.ply")

        elif self.cfg.translation_mode == "aug":
            # augment gt translation
            pcd_trans = translation + np.random.uniform(-0.5, 0.5, size=translation.shape) * bbox_sizes * 1.0
            pcd_sizes = bbox_sizes + np.random.uniform(-0.5, 0.5, size=bbox_sizes.shape) * bbox_sizes * 1.0
        
        elif self.cfg.translation_mode == "gt":
            # gt translation
            pcd_trans = translation
            pcd_sizes = bbox_sizes
        
        elif self.cfg.translation_mode == "empty":
            pcd_trans = np.zeros_like(translation)
            pcd_sizes = np.ones_like(bbox_sizes)
            
        else:
            raise NotImplementedError(f"translation mode {self.cfg.translation_mode} not implemented")

        # transform to tensor
        crop_imgs = torch.stack(crop_imgs)
        masks = torch.stack(masks)
        image = torch.from_numpy(image/255.0).float()
        geo_shape = torch.stack(geo_shape)
        geo_shape_sharp = torch.stack(geo_shape_sharp)
        bbox_sizes = torch.from_numpy(bbox_sizes).float()
        obj_pcds = torch.from_numpy(np.stack(obj_pcds)).float()
        rotation = torch.from_numpy(rotation).reshape(-1, 6).float()
        translation = torch.from_numpy(translation).float()
        tran_matrix_cam = torch.from_numpy(tran_matrix_cam).float()
        scene_pcds = torch.from_numpy(scene_pcds).float()
        pcd_sizes = torch.from_numpy(pcd_sizes).float()
        pcd_trans = torch.from_numpy(pcd_trans).float()

        # resize and crop image
        if self.cfg.image_batchfy_mode == "resize":
            image = F.interpolate(image.permute(2, 0, 1).unsqueeze(0), size=(self.cfg.image_height, self.cfg.image_width), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0)
            masks = F.interpolate(masks.unsqueeze(1).float(), size=(self.cfg.image_height, self.cfg.image_width), mode="nearest").squeeze(1) 
        elif self.cfg.image_batchfy_mode == "resize_padding":  
            image, masks = self.resize_and_pad(image, masks, self.cfg.image_height, self.cfg.image_width)

        # padding to max_objs
        if n_objects < self.cfg.max_objs:
            pad_len = self.cfg.max_objs - n_objects
            geo_shape = F.pad(geo_shape, (0, 0, 0, 0, 0, pad_len))
            geo_shape_sharp = F.pad(geo_shape_sharp, (0, 0, 0, 0, 0, pad_len))
            crop_imgs = F.pad(crop_imgs, (0, 0, 0, 0, 0, 0, 0, pad_len))
            masks = F.pad(masks, (0, 0, 0, 0, 0, pad_len))
            bbox_sizes = F.pad(bbox_sizes, (0, 0, 0, pad_len))
            obj_pcds = F.pad(obj_pcds, (0, 0, 0, 0, 0, pad_len))
            rotation = F.pad(rotation, (0, 0, 0, pad_len))
            translation = F.pad(translation, (0, 0, 0, pad_len))     
            tran_matrix_cam = F.pad(tran_matrix_cam, (0, 0, 0, 0, 0, pad_len))
            scene_pcds = F.pad(scene_pcds, (0, 0, 0, 0, 0, pad_len))
            pcd_sizes = F.pad(pcd_sizes, (0, 0, 0, pad_len))
            pcd_trans = F.pad(pcd_trans, (0, 0, 0, pad_len))
            class_list = class_list + [""] * pad_len
        # remove extra objects         
        elif n_objects > self.cfg.max_objs:
            geo_shape = geo_shape[:self.cfg.max_objs]
            geo_shape_sharp = geo_shape_sharp[:self.cfg.max_objs]
            crop_imgs = crop_imgs[:self.cfg.max_objs]
            masks = masks[:self.cfg.max_objs]
            bbox_sizes = bbox_sizes[:self.cfg.max_objs]
            obj_pcds = obj_pcds[:self.cfg.max_objs]
            rotation = rotation[:self.cfg.max_objs]
            translation = translation[:self.cfg.max_objs]
            mask_legal = mask_legal[:self.cfg.max_objs]
            tran_matrix_cam = tran_matrix_cam[:self.cfg.max_objs]
            scene_pcds = scene_pcds[:self.cfg.max_objs]
            pcd_sizes = pcd_sizes[:self.cfg.max_objs]
            pcd_trans = pcd_trans[:self.cfg.max_objs]
            class_list = class_list[:self.cfg.max_objs]
        
        data_dict = {"whole_img": image, "pose": rotation, "translation": translation, "size": bbox_sizes[...,:1], "trans_matrix": tran_matrix_cam, "surface": geo_shape, "sharp_surface": geo_shape_sharp, \
                "image": crop_imgs, "obj_pcds": obj_pcds, "scene_pcds": scene_pcds, "pcd_sizes": pcd_sizes[...,:1], "pcd_trans": pcd_trans, "mask_legal": mask_legal, "taskid": room_id, "masks": masks, \
                "class_list": class_list, "caption": class_list, "miss_obj": miss_obj}
        
        if self.cfg.use_deocc_image:
            deocc_images = torch.stack(deocc_images)
            deocc_images = F.pad(deocc_images, (0, 0, 0, 0, 0, 0, 0, pad_len)) if n_objects < self.cfg.max_objs else deocc_images[:self.cfg.max_objs]
            data_dict["deocc_images"] = deocc_images
        
        # add geometry under scene space
        if self.cfg.use_scene_geometry:
            scene_surface, scene_sharp_surface = [], []
            bbox_size = bbox_sizes[:,None,:1] / 2
            transform_matrix = self.recover_transform_matrix(rotation, translation)
            for idx, legal in enumerate(mask_legal):
                if not legal:
                    scene_surface.append(torch.zeros_like(geo_shape[0]))
                    scene_sharp_surface.append(torch.zeros_like(geo_shape_sharp[0]))
                    continue
                # surface
                points_transformed = ((geo_shape[idx,:,:3] * bbox_size[idx]) @ transform_matrix[idx,:3,:3].T) + transform_matrix[idx,:3,3]
                normals_transformed = geo_shape[idx,:,3:] @ transform_matrix[idx,:3,:3].T
                scene_surface.append(torch.cat([points_transformed, normals_transformed], dim=-1))
                # sharp surface
                sharp_surface_transformed = ((geo_shape_sharp[idx,:,:3] * bbox_size[idx]) @ transform_matrix[idx,:3, :3].T) + transform_matrix[idx,:3,3]
                sharp_surface_normals_transformed = geo_shape_sharp[idx,:,3:] @ transform_matrix[idx,:3,:3].T
                scene_sharp_surface.append(torch.cat([sharp_surface_transformed, sharp_surface_normals_transformed], dim=-1))
            data_dict["scene_surface"] = torch.stack(scene_surface, dim=0)
            data_dict["scene_sharp_surface"] = torch.stack(scene_sharp_surface, dim=0)
            # # viz
            # trimesh.points.PointCloud(data_dict["scene_surface"][...,:3].reshape(-1,3)).export(f"outputs/viz/pts/scene_surface.ply")
            # trimesh.points.PointCloud(scene_pcds.reshape(-1,3)).export(f"outputs/viz/pts/scene_pcd.ply")
            # pdb.set_trace()

        return data_dict
    
    def __getitem__(self, index):
        """
        Robust version of __getitem__: if file/data missing or error, try index+1, up to max_attempts.
        # """
        max_attempts = 10
        attempts = 0
        orig_index = index
        while attempts < max_attempts:
            try:
                return self.__getitem_scene(index)
            except Exception as e:
                print(f"[Warning] Failed to load index {index} ({getattr(self, 'all_scenes', ['?'])[index]}): {e}")
                index += 1
                attempts += 1
                if index >= len(self.all_scenes):
                    index = index - len(self.all_scenes)  # wrap around if needed
        raise RuntimeError(f"Too many consecutive missing/corrupted files in dataset (start from {orig_index}).")
        
        return self.__getitem_scene(index)
    
    # def collate(self, batch):
    #     from torch.utils.data._utils.collate import default_collate_fn_map
    #     return torch.utils.data.default_collate(batch)
    
    def collate(self, batch):
        from torch.utils.data._utils.collate import default_collate_fn_map
        ret = {}
        for key, value in batch[0].items():
            if isinstance(value, str):
                ret[key] = [b[key] for b in batch]
            elif isinstance(value, list):
                ret[key] = [b[key] for b in batch]
            elif isinstance(value, dict):
                ret[key] = [b[key] for b in batch]
            elif isinstance(value, torch.Tensor):
                ret[key] = torch.stack([b[key] for b in batch])
            elif isinstance(value, bool):
                ret[key] = [b[key] for b in batch]
            elif isinstance(value, np.ndarray):
                ret[key] = torch.stack([torch.from_numpy(b[key]) for b in batch])
            else:
                ret[key] = default_collate_fn_map[type(batch[0][key])](batch)
        return ret
    
    def save_crop_and_mask(self, rgb, cropped_mask, label, taskid, object_ind, save_dir):
        """
        保存裁剪后的rgb、mask和label到以taskid命名的文件夹下
        Args:
            rgb: numpy array, shape (H, W, 3), 0~255或0~1
            cropped_mask: numpy array, shape (H, W)
            label: str
            taskid: str
            object_ind: int
            save_dir: str, 根目录
        """
        import os
        from PIL import Image

        # 确保保存目录存在
        out_dir = os.path.join(save_dir, str(taskid))
        os.makedirs(out_dir, exist_ok=True)

        # 保存rgb
        rgb_img = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
        Image.fromarray(rgb_img).save(os.path.join(out_dir, f"crop_{object_ind}.png"))

        # 保存mask
        mask_img = (cropped_mask * 255).astype(np.uint8) if cropped_mask.max() <= 1.0 else cropped_mask.astype(np.uint8)
        Image.fromarray(mask_img).save(os.path.join(out_dir, f"mask_{object_ind}.png"))

        # 保存label
        with open(os.path.join(out_dir, f"label_{object_ind}.txt"), "w") as f:
            f.write(str(label))
    

# data loader  
@register("Front3D-midi-datamodule")
class Front3DMIDIDataModule(pl.LightningDataModule):
    cfg: Front3DMIDIDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(Front3DMIDIDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = Front3DMIDIDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = Front3DMIDIDataset(self.cfg, "test")
            self.train_dataset = Front3DMIDIDataset(self.cfg, "train")
        if stage in [None, "test", "predict"]:
            self.test_dataset = Front3DMIDIDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None, num_workers=0) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
            num_workers=self.cfg.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, 
            batch_size=1,
            collate_fn=self.val_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, 
            batch_size=1,
            collate_fn=self.test_dataset.collate)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)
    
    
# for debug
if __name__ == "__main__":
    
    
    import argparse
    parser = argparse.ArgumentParser(description="Quick debug for Front3DMIDIDataset/DataModule")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'], help='Dataset mode')
    parser.add_argument('--cfg', type=str, default=None, help='Path to config file (optional)')
    args = parser.parse_args()

    # 加载配置
    if args.cfg is not None:
        import yaml
        with open(args.cfg, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        cfg = Front3DMIDIDataModuleConfig(**cfg_dict)
    else:
        cfg = Front3DMIDIDataModuleConfig()

    # 初始化数据集
    dataset = Front3DMIDIDataset(cfg, args.mode)
    print(f"Dataset length: {len(dataset)}")

    # 随机取一个样本调试
    sample_idx = random.randint(0, len(dataset)-1)
    print(f"Debug sample index: {sample_idx}")
    sample = dataset[sample_idx]