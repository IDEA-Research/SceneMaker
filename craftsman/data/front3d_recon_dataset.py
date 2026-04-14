# Dataloader of InstPIFu.
# author: LiuHaolin
# date: Aug, 2022
from omegaconf import OmegaConf
import os
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
from utils.space_transform import reconstruct_pcd, get_rot_from_pitch, get_pitch_from_R, q2rot, get_yaw_from_R, get_rot_from_yaw
from utils.transforms import *
import trimesh
import fpsample
import rembg

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

def read_obj_point(obj_path):
    with open(obj_path, 'r') as f:
        content_list = f.readlines()
        point_list = [line.rstrip("\n").lstrip("v ").split(" ") for line in content_list]
        for point in point_list:
            for i in range(3):
                point[i] = float(point[i])
        return np.array(point_list)

def R_from_yaw_pitch_roll(yaw, pitch, roll):
    '''
    get rotation matrix from predicted camera yaw, pitch, roll angles.
    :param yaw: batch_size x 1 tensor
    :param pitch: batch_size x 1 tensor
    :param roll: batch_size x 1 tensor
    :return: camera rotation matrix
    '''
    Rp = np.zeros((3, 3))
    Ry = np.zeros((3, 3))
    Rr = np.zeros((3, 3))
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cr = np.cos(roll)
    sr = np.sin(roll)
    Rp[0, 0] = 1
    Rp[1, 1] = cp
    Rp[1, 2] = -sp
    Rp[2, 1] = sp
    Rp[2, 2] = cp

    Ry[0, 0] = cy
    Ry[0, 2] = sy
    Ry[1, 1] = 1
    Ry[2, 0] = -sy
    Ry[2, 2] = cy

    Rr[0, 0] = cr
    Rr[0, 1] = -sr
    Rr[1, 0] = sr
    Rr[1, 1] = cr
    Rr[2, 2] = 1

    R = np.dot(np.dot(Rr, Rp), Ry)
    return R

def get_centroid_from_proj(centroid_depth, proj_centroid, K):
    x_temp = (proj_centroid[0] - K[0, 2]) / K[0, 0]
    y_temp = (proj_centroid[1] - K[1, 2]) / K[1, 1]
    z_temp = 1
    ratio = centroid_depth / np.sqrt(x_temp ** 2 + y_temp ** 2 + z_temp ** 2)
    x_cam = x_temp * ratio
    y_cam = y_temp * ratio
    z_cam = z_temp * ratio
    p = np.stack([x_cam, y_cam, z_cam])
    return p


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



@dataclass
class Front3DReconDataModuleConfig(BaseS3DataModuleConfig):    
    ################################# Scene #################################
    data_path: str = "/comp_robot/shiyukai/datasets/instPiFU/datasets/prepare_data/"
    split_dir: str = 'data/3dfront/split'
    occ_path: str = '/comp_robot/shiyukai/datasets/instPiFU/datasets/occ'
    deocc_mask_path: str = '/comp_robot/shiyukai/datasets/instPiFU/datasets/mask'
    mask_path: str = '/comp_robot/shiyukai/datasets/instPiFU/datasets/render_mask/'
    geo_data_path: str = "/comp_robot/shiyukai/datasets/instPiFU/datasets/normalized_watertight/sampling_objects"
    avg_layout_path: str = 'data/3dfront/avg_layout.pkl'
    raw_path: str = "/comp_robot/shiyukai/datasets/instPiFU/datasets/3d-front-data-small/"
    deocc_image_path: str = '/comp_robot/shiyukai/datasets/instPiFU/datasets/render_obj_scene_img/'
    
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
    only_use_pitch: bool = False # whether to only use pitch rotation
    use_mix_coord: bool = False
    use_ccm_pts: bool = False

    ################################# General argumentation #################################
    random_flip: bool = True            # whether to randomly flip the input point cloud and the input images
    random_color_jitter: bool = True    # whether to randomly color jitter the input images
    image_width: int = 648
    image_height: int = 484
    image_batchfy_mode: str = "resize"

    ################################# Geometry part #################################
    load_geometry: bool = True           # whether to load geometry data
    with_sharp_data: bool = True
    geo_data_type: str = "sdf"     # occupancy, sdf
    # for occupancy and sdf data
    n_samples: int = 16384                # number of points in input point cloud
    upsample_ratio: int = 1              # upsample ratio for input point cloud
    sampling_strategy: Optional[str] = None    # sampling strategy for input point cloud
    scale: float = 1.0                   # scale of the input point cloud and target supervision
    noise_sigma: float = 0.0             # noise level of the input point cloud
    rotate_points: bool = False          # whether to rotate the input point cloud and the supervision
    load_supervision: bool = False        # whether to load supervision
    supervision_type: str = "sdf"  # occupancy, sdf, tsdf, tsdf_w_surface
    n_supervision: int = 10000           # number of points in supervision
    tsdf_threshold: float = 0.01         # threshold for truncating sdf values, used when input is sdf

    ################################# Image part #################################
    load_image: bool = True             # whether to load images 
    image_type: str = "rgb_or_normal"              # rgb, normal, rgb_or_normal
    image_type_ratio: float = 0.95        # ratio of rgb for each dataset when image_type is "rgb_or_normal"
    images_per_sample: int = 1          # the loaded number of images
    background_color: Tuple[int, int, int] = field(
            default_factory=lambda: (255, 255, 255)
        )
    idx: Optional[List[int]] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])      # index of the image to load
    n_views: int = 1                     # number of views
    foreground_ratio: Optional[float] = 0.95 

    ################################# Caption part #################################
    load_caption: bool = False           # whether to load captions
    
    batch_size: int = 32
    num_workers: int = 0
    text_max_length: int = 77

class Front3DReconDataset(Dataset):
    def __init__(self, config, mode):
        super(Front3DReconDataset, self).__init__()
        self.mode=mode
        self.cfg: Front3DReconDataModuleConfig = config

        if mode=="train":
            self.data_path=os.path.join(config.data_path, 'train')
        elif mode=="test":
            self.data_path = os.path.join(config.data_path,'test') if self.cfg.num_scene < 0 else os.path.join(config.data_path,'train')
            # self.data_path=os.path.join(config.data_path,'train')
        elif mode=="val":
            self.data_path=os.path.join(config.data_path,'val')

        json_file = os.path.join(self.data_path, "pkl_filenames.json")
        if not os.path.exists(json_file):
            self.split=glob.glob(self.data_path+"/*.pkl")
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(self.split, f, ensure_ascii=False, indent=4)

        with open(json_file, "r", encoding="utf-8") as f:
            self.split = json.load(f)
            
        # load pickle
        with open(config.avg_layout_path, 'rb') as f:
            self.avg_layout = pickle.load(f)
        
        if mode == "test":
            self.split = self.split[:10]
            # self.split = self.split[20:50]
        if mode == "train" and self.cfg.num_scene >= 0:
            self.split = self.split[:self.cfg.num_scene]
            
        if self.cfg.add_test and mode == "train":
            self.test_len = 0
            test_split = os.path.join(config.data_path, 'test', "pkl_filenames.json")
            if os.path.exists(test_split):
                with open(test_split, "r", encoding="utf-8") as f:
                    test_split = json.load(f)
                self.test_len = len(test_split) - 100
                self.split += test_split[100:]

    def __len__(self):
        return len(self.split)
    
    def _load_shape_from_occupancy_or_sdf(self, taskid, index, scale=1) -> Dict[str, Any]:
        # for supervision
        data = np.load(f'{self.cfg.geo_data_path}/{index}/normalized_watertight.npz')
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
        等比例缩放，使得宽或高其中一边等于目标大小，另一边小于目标大小，然后用padding补齐。
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

    def normalize_vector(self, v):
        return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)
    
    def _load_image(self, path):
        image = Image.fromarray(path)
        image = torch.from_numpy(np.asarray(image)/ 255.0).float()
        return image        
    
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
        fname = self.split[index]
        prepare_data_path = os.path.join(self.data_path, fname)
        
        # if in train mode and add test set, return test set
        if self.mode == "train" and self.cfg.add_test and index >= len(self.split) - self.test_len:
            prepare_data_path = os.path.join(self.data_path.replace("train", "test"), fname)

        # load pickle
        with open(prepare_data_path, 'rb') as f:
            sequence = pickle.load(f)
        
        taskid = sequence["sequence_id"]
        layout = sequence["layout"]
        boxes = sequence['boxes']
        camera = sequence['camera']
        n_objects = boxes['bdb2D_pos'].shape[0]

        # encode class
        class_list = [inv_category_mapping[v] for v in boxes['size_cls']]
        
        # load image
        image = sequence['rgb_img']
        if image.shape[2] == 4:
            image = image[:,:,:3]
        width, height = image.shape[1], image.shape[0]
        
        # raw
        json_path = os.path.join(self.cfg.raw_path, taskid,"desc.json")
        with open(json_path,'rb') as f:
            content=json.load(f)
        bbox_infos = content["bbox_infos"]
        camera_K = np.asarray(bbox_infos["camera"]["K"]).reshape(3,3)
        depth_path=os.path.join(self.cfg.raw_path, taskid, "depth.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
        depth = (1 - depth / 255.0) * 10
        
        # rescale depth
        depth_scale = (width / depth.shape[0] + height / depth.shape[1]) / 2
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST) # resize depth
        pitch_rot = get_rot_from_pitch(layout["pitch"])
        point_cloud_incam, pcd_scene_trans, pcd_scene_scale = reconstruct_pcd(depth=depth, cam2wrd_rot=pitch_rot, intrinsic=camera_K, scale=depth_scale, normalize=True)
        # trimesh.points.PointCloud(point_cloud_incam, colors=[255, 0, 0]).export(f"outputs/viz/pcd2/depth_pts_{index}.ply")
        
        # # load depth map and pts
        # depth = sequence['depth_map']/255.0*10
        # point_cloud_incam = reconstruct_pcd(depth=depth, cam2wrd_rot=pitch_rot, intrinsic=camera["K"], scale=1)
        # trimesh.points.PointCloud(point_cloud_incam, colors=[255, 0, 0]).export(f"outputs/viz/pcd/depth_pts_{index}.ply")
        
        # viz
        # Image.fromarray(image).save("outputs/viz/rgb.png")
        # Image.fromarray(sequence['depth_map']).save("outputs/viz/depth.png")
        # Image.fromarray((depth/10*255).astype(np.uint8)).save("outputs/viz/depth2.png")
        
        # transformation under cam pose
        tran_matrix = boxes["tran_matrix"]
        wrd2cam_matrix = sequence['camera']['wrd2cam_matrix']
        if self.cfg.only_use_pitch:
            wrd2cam_matrix[:3,:3] = get_rot_from_yaw(get_yaw_from_R(wrd2cam_matrix[:3, :3]))
        # only use pose in world space when use_wrd_pose is True
        tran_matrix_cam = tran_matrix if self.cfg.use_wrd_pose else (wrd2cam_matrix[None,...] @ tran_matrix)
            
        # mask for legal object
        mask_legal = torch.zeros(self.cfg.max_objs)
        
        geo_shape, geo_shape_sharp = [], []
        crop_imgs, masks = [], []
        obj_pcds, scene_pcds, scene_surface = [], [], []
        bbox_sizes = []
        pcd_sizes, pcd_trans = [], []
        if self.cfg.use_deocc_image:
            deocc_images = []

        miss_obj = False    
        for object_ind in range(n_objects):
            if object_ind >= self.cfg.max_objs:
                break
            # size
            size_cls = boxes['size_cls'][object_ind]
            bbox_size = (boxes['size_reg'][object_ind] + 1) * bin['avg_size'][size_cls]
            # normalize in xyz
            size = bbox_size / bbox_size.max()
            bbox_size = bbox_size.max()*np.ones_like(bbox_size)
            bbox_sizes.append(bbox_size)
            
            # object index
            jid = boxes['jid'][object_ind]
            
            # load object shape
            ret = self._load_shape_from_occupancy_or_sdf(taskid, jid, scale=size)
            if "pcd2_aug" in self.cfg.translation_mode and self.mode == "train":
                # trimesh.PointCloud(ret["surface"][:,:3]).export(f"outputs/viz/pts/pointcloud_{object_ind}_before.ply") # viz pts before augmentation
                if "pcd2_aug_pi" in self.cfg.translation_mode:
                    ret, aug_rot = self.augment_surface_with_random_y_rotation(ret, angle_list=[0, np.pi/2, np.pi, 3*np.pi/2]) # apply rotation augmentation
                else:
                    ret, aug_rot = self.augment_surface_with_random_y_rotation(ret) # apply random rotation
                # trimesh.PointCloud(ret["surface"][:,:3]).export(f"outputs/viz/pts/pointcloud_{object_ind}_after.ply") # viz pts after augmentation
                tran_matrix_cam[object_ind, :3, :3] = tran_matrix_cam[object_ind, :3, :3] @ aug_rot.T # apply augmentation to tran_matrix_cam
            geo_shape.append(torch.from_numpy(ret["surface"]))
            geo_shape_sharp.append(torch.from_numpy(ret["sharp_surface"]))
            # gt surface in scene space
            scene_pts_gt = ((ret["surface"][:,:3] * bbox_size.max()/2) @ tran_matrix_cam[object_ind,:3,:3].T) + tran_matrix_cam[object_ind,:3,3]
            scene_surface.append(scene_pts_gt)
            
            # mask
            # instance_mask_path = os.path.join(self.cfg.mask_path_original, "%s_%s.png" % (taskid, object_ind))
            instance_mask_path = os.path.join(self.cfg.mask_path, taskid, f"mask_{object_ind}.png")
            if os.path.exists(instance_mask_path):
                instance_mask = np.array(Image.open(instance_mask_path))
                instance_mask = cv2.resize(instance_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            else:
                instance_mask = np.zeros((height, width), dtype=np.uint8)
            masks.append(torch.from_numpy(instance_mask)/255.0)
        
            # crop from scene image
            if self.cfg.use_deocc_image:
                # load deocc image
                deocc_image_path = os.path.join(self.cfg.deocc_image_path, taskid, f"{jid}.png")
                deocc_image = np.array(Image.open(deocc_image_path).convert("RGB"))
                deocc_image = cv2.resize(deocc_image, (width, height), interpolation=cv2.INTER_LINEAR)
                # load deocc mask
                deocc_mask_path = os.path.join(self.cfg.deocc_mask_path, "%s_%s.png" % (taskid, object_ind))
                deocc_mask = np.array(Image.open(deocc_mask_path))
                deocc_mask = cv2.resize(deocc_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                # apply mask on deocc image
                deocc_image = (deocc_image * (deocc_mask[..., None].astype(np.float32)/255) + np.ones_like(deocc_image) * 255 * (1 - deocc_mask[..., None].astype(np.float32)/255)).astype(np.uint8)
                # apply deocc mask
                rgb, cropped_mask = self.crop_image(image=deocc_image, mask=deocc_mask, bbox_ratio=0.8)
                deocc_images.append(torch.from_numpy(deocc_image/255.0))
            else:
                rgb, cropped_mask = self.crop_image(image=image, mask=instance_mask, bbox_ratio=0.8)
            white_bg = np.ones_like(rgb) * 255 # apply mask with white background
            rgb = rgb * (cropped_mask[..., None].astype(np.float32)/255) + white_bg * (1 - cropped_mask[..., None].astype(np.float32)/255) 
            crop_imgs.append(torch.from_numpy(rgb/255.0))
            
            # save rgb, cropped_mask and label under folder named with taskid
            # self.save_crop_and_mask(rgb, cropped_mask, class_list[object_ind], taskid, object_ind, save_dir="outputs/viz/deocc_data")
            
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
                    scene_obj_pcd = (scene_obj_pcd @ tran_matrix_cam[object_ind,:3, :3].T) + tran_matrix_cam[object_ind, :3, 3]
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
                    # dialate mask
                    # mask = ~self.dilate_mask(~instance_mask, kernel_size=3, iterations=1)
                    obj_pcd_cam, pcd_tran, pcd_size = reconstruct_pcd(depth=depth, cam2wrd_rot=pitch_rot, intrinsic=camera_K, scale=depth_scale, normalize=True, valid_mask=instance_mask)
            except:
                obj_pcd_cam = torch.zeros(self.cfg.min_pcd, 3)
                pcd_tran = np.zeros(3)
                pcd_size = 1e-4
            
            if "full_pcd" in self.cfg.translation_mode:
                pass
            elif (obj_pcd_cam.shape[0] <= self.cfg.min_pcd):
                obj_pcd_cam = torch.zeros(self.cfg.min_pcd, 3)
                pcd_tran = np.zeros(3)
                pcd_size = 1e-4
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
            # points_transformed = (points_scaled @ tran_matrix_cam[object_ind,:3, :3].T) + tran_matrix_cam[object_ind, :3, 3]
            # trimesh.PointCloud(points_transformed).export(f"outputs/viz/pts/pointcloud_{object_ind}.ply")
            # trimesh.points.PointCloud(obj_pcd_cam*pcd_size+pcd_tran, colors=[255, 0, 0]).export(f"outputs/viz/depth_pcd/depth_pcd_{object_ind}.ply")
            
            # viz crop image
            # Image.fromarray((crop_imgs[-1].numpy()*255).astype(np.uint8)).save(f"outputs/viz/crop/crop_{object_ind}.png")
            # Image.fromarray((instance_mask).astype(np.uint8)).save(f"outputs/viz/crop/mask_{object_ind}.png")            

        # scene level params
        if self.cfg.use_mix_coord:
            scene_surface = np.concatenate(scene_surface, axis=0)
            centroid = (np.max(scene_surface, axis=0) + np.min(scene_surface, axis=0)) / 2
            layout_size = np.max(np.abs(scene_surface-centroid)) * 2 * 0.95
        else:
            layout_size = np.max(layout['coeffs_reg'] + self.avg_layout['avg_size'])
            centroid = self.avg_layout['avg_centroid'] + layout['centroid_reg'] # camera space
       
        # object size
        bbox_sizes = np.stack(bbox_sizes)
        bbox_sizes = bbox_sizes / layout_size * 2.0
        # translation
        translation = tran_matrix_cam[:bbox_sizes.shape[0],:3,3]
        translation = (translation - centroid) / layout_size * 2.0
        # rotation
        rotation = mat2repr6d(tran_matrix_cam[:bbox_sizes.shape[0],:3,:3])
        
        # reference translation
        scene_pcds = np.stack(scene_pcds, axis=0)
        mask_index = torch.nonzero(mask_legal).numpy().squeeze(-1)
        # normalize scene pcd with pcd min-max
        if self.cfg.translation_mode == "pcd":
            if mask_index.shape[0] > 0:
                scene_pcds[mask_index] = (scene_pcds[mask_index] - pcd_scene_trans) / pcd_scene_scale
            # calculate pcd size of objects
            pcd_trans = np.stack(pcd_trans, axis=0)
            pcd_trans = (pcd_trans - centroid) / layout_size * 2.0
            pcd_sizes = np.stack(pcd_sizes, axis=0)
            pcd_sizes = pcd_sizes / layout_size * 2.0
            pcd_sizes = np.repeat(pcd_sizes[:, np.newaxis], repeats=3, axis=-1)
        
        # normalize pcd with gt scene centroid and size
        elif self.cfg.translation_mode == "pcd2" or "full_pcd2" in self.cfg.translation_mode or "pcd2_aug" in self.cfg.translation_mode:
            # normalize in whole scene
            if mask_index.shape[0] > 0:
                # scene_pcds[mask_index] = (scene_pcds[mask_index] - centroid) / layout_size * 2
                scene_pcds = (scene_pcds - centroid) / layout_size * 2.0
            # calculate pcd size of objects
            pcd_trans = np.stack(pcd_trans, axis=0)
            pcd_trans = (pcd_trans - centroid) / layout_size * 2.0
            pcd_sizes = np.stack(pcd_sizes, axis=0)
            pcd_sizes = pcd_sizes / layout_size * 2.0
            pcd_sizes = np.repeat(pcd_sizes[:, np.newaxis], repeats=3, axis=-1)
            # # viz scene pcd
            # trimesh.points.PointCloud(scene_pcds.reshape(-1,3), colors=[255, 0, 0]).export(f"outputs/viz/pcd/depth_pts_scene_wrd.ply")

        # normalize pcd with gt scene centroid and size
        elif self.cfg.translation_mode == "pcd2_whole":
            # normalize in whole scene
            if mask_index.shape[0] > 0:
                scene_pcds[mask_index] = (scene_pcds[mask_index] - pcd_scene_trans) / pcd_scene_scale
            # calculate pcd size of objects
            pcd_trans = np.stack(pcd_trans, axis=0)
            pcd_trans = (pcd_trans - centroid) / layout_size * 2.0
            pcd_sizes = np.stack(pcd_sizes, axis=0)
            pcd_sizes = pcd_sizes / layout_size * 2.0
            pcd_sizes = np.repeat(pcd_sizes[:, np.newaxis], repeats=3, axis=-1)
            # sample min pcd points from scene pcds
            scene_pcds = scene_pcds.reshape(-1, 3)
            if scene_pcds.shape[0] > self.cfg.min_pcd:
                rng = np.random.default_rng()
                ind = rng.choice(scene_pcds.shape[0], self.cfg.min_pcd, replace=False)
                scene_pcds = scene_pcds[ind]
            scene_pcds = scene_pcds.reshape(-1, self.cfg.min_pcd, 3).repeat(pcd_trans.shape[0], axis=0)
            # viz scene pcd
            # trimesh.points.PointCloud(scene_pcds.reshape(-1,3), colors=[255, 0, 0]).export(f"outputs/viz/pcd/depth_pts_{index}.ply")
            
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
                "image": crop_imgs, "obj_pcds": obj_pcds, "scene_pcds": scene_pcds, "pcd_sizes": pcd_sizes[...,:1], "pcd_trans": pcd_trans, "mask_legal": mask_legal, "taskid": taskid, "masks": masks, \
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
        """
        max_attempts = 10
        attempts = 0
        orig_index = index
        while attempts < max_attempts:
            try:
                return self.__getitem_scene(index)
            except Exception as e:
                print(f"[Warning] Failed to load index {index} ({getattr(self, 'split', ['?'])[index]}): {e}")
                index += 1
                attempts += 1
                if index >= len(self.split):
                    index = index - len(self.split)  # wrap around if needed
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
    
    
    