# Dataloader of InstPIFu.
# author: LiuHaolin
# date: Aug, 2022
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
from utils.space_transform import reconstruct_pcd, get_rot_from_pitch, get_pitch_from_R, q2rot
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
import numpy as np
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
class PoseDataModuleConfig(BaseS3DataModuleConfig):    
    ################################# Scene #################################
    data_path: str = '/comp_robot/shiyukai/datasets/instPiFU/datasets/prepare_data'
    split_dir: str = 'data/3dfront/split'
    occ_path: str = '/comp_robot/shiyukai/datasets/instPiFU/datasets/occ'
    mask_path_original: str = '/comp_robot/shiyukai/datasets/instPiFU/datasets/mask'
    mask_path: str = '/comp_robot/shiyukai/datasets/instPiFU/datasets/render_mask/'
    geo_data_path: str = 'data/samples/sample_sampling_objects'
    avg_layout_path: str = 'data/3dfront/avg_layout.pkl'
    raw_path: str = "/comp_robot/shiyukai/datasets/instPiFU/datasets/3d-front-data-small/"
    
    ################################# Scene argumentation #################################
    max_objs: int = 5
    min_pcd: int = 1000
    translation_mode: str = "pcd"
    refine_mask: bool = False
    augmentation: bool = False

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

class PoseDataset(Dataset):
    def __init__(self, config, mode):
        super(PoseDataset, self).__init__()
        self.mode=mode
        self.cfg: PoseDataModuleConfig = config

        if mode=="train":
            self.data_path=os.path.join(config.data_path, 'train')
        elif mode=="test":
            self.data_path=os.path.join(config.data_path,'test')
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
            self.split = self.split[:100]

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
    
    def crop_image(self, image, mask):    
        '''
        transform image to 512x512, crop the image with mask
        Input:
            image: numpy array
            mask: numpy array
        Output:
            image: numpy array
        '''
        # transfer to PIL image
        image = Image.fromarray((image).astype(np.uint8))
        mask = Image.fromarray((mask).astype(np.uint8))
        
        # Apply mask on image
        image = Image.composite(image, Image.new("RGB", image.size, (255, 255, 255)), mask)
        
        # crop  
        image = image.crop(mask.getbbox())
        
        # remove bg
        # image = rembg.remove(image, bgcolor=(255, 255, 255, 255))
    
        # pad image to 1:1
        width, height = image.size
        new_size = (width, width) if width > height else (height, height)
        if width > height: 
            new_size = (width, width)
            new_image = Image.new("RGBA", new_size, (255, 255, 255, 255))
            new_image.paste(image, (0, (width - height) // 2))
        else:
            new_size = (height, height)
            new_image = Image.new("RGBA", new_size, (255, 255, 255, 255))
            new_image.paste(image, ((height - width) // 2, 0))
        image = new_image    
        
        # convert to rgba
        image = image.convert("RGB")
        
        # resize
        image = np.array(image.resize((512,512)))
        
        return image

    
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

    def normalize_vector(self, v):
        return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)

    
    def _load_image(self, path):
        image = Image.fromarray(path)
        image = torch.from_numpy(np.asarray(image)/ 255.0).float()
        return image
    
    def random_sample_rot(self, batch_size=1):
        u1 = torch.rand(batch_size)
        u2 = torch.rand(batch_size)
        u3 = torch.rand(batch_size)

        q1 = torch.sqrt(1 - u1) * torch.sin(2 * np.pi * u2)
        q2 = torch.sqrt(1 - u1) * torch.cos(2 * np.pi * u2)
        q3 = torch.sqrt(u1) * torch.sin(2 * np.pi * u3)
        q4 = torch.sqrt(u1) * torch.cos(2 * np.pi * u3)

        # 四元数 (x, y, z, w)
        quaternions = torch.stack([q1, q2, q3, q4], dim=1)  # shape: (B, 4)

        # 转换为旋转矩阵
        rot = quat2repr6d(quaternions)  # 你需要实现或使用现有库
        return rot
        

    def __getitem__(self, index):
        fname = self.split[index]
        prepare_data_path = os.path.join(self.data_path, fname)

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
        pitch_rot = get_rot_from_pitch(layout["pitch"])
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
        tran_matrix_cam = (wrd2cam_matrix[None,...] @ tran_matrix)
        # mask for legal object
        mask_legal = torch.zeros(self.cfg.max_objs)
        
        geo_shape, geo_shape_sharp = [], []
        crop_imgs, masks = [], []
        obj_pcds, scene_pcds = [], []
        bbox_sizes = []
        pcd_sizes, pcd_trans = [], []
        for object_ind in range(n_objects):
            if object_ind >= self.cfg.max_objs:
                break
            
            # size
            size_cls = boxes['size_cls'][object_ind]
            bbox_size = (boxes['size_reg'][object_ind] + 1) * bin['avg_size'][size_cls]
            # normalize in xyz
            size = bbox_size / bbox_size.max()
            bbox_sizes.append(bbox_size.max()*np.ones_like(bbox_size))
            
            # object index
            jid = boxes['jid'][object_ind]
            
            # load object shape
            ret = self._load_shape_from_occupancy_or_sdf(taskid, jid, scale=size)
            geo_shape.append(torch.from_numpy(ret["surface"]))
            geo_shape_sharp.append(torch.from_numpy(ret["sharp_surface"]))
            
            # mask
            # instance_mask_path = os.path.join(self.cfg.mask_path_original, "%s_%s.png" % (taskid, object_ind))
            instance_mask_path = os.path.join(self.cfg.mask_path, taskid, f"mask_{object_ind}.png")
            if os.path.exists(instance_mask_path):
                instance_mask = np.array(Image.open(instance_mask_path))
                instance_mask = cv2.resize(instance_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            else:
                instance_mask = np.zeros((height, width), dtype=np.uint8)
            masks.append(torch.from_numpy(instance_mask)/255.0)
            
            # crop image
            crop_imgs.append(torch.from_numpy(self.crop_image(image, instance_mask)/255.0).float())
            
            # object pcd
            try:
                obj_pcd_cam, pcd_tran, pcd_size = reconstruct_pcd(depth=depth, cam2wrd_rot=pitch_rot, intrinsic=camera_K, scale=depth_scale, normalize=True, valid_mask=instance_mask)
            except:
                obj_pcd_cam = torch.zeros(self.cfg.min_pcd, 3)
                pcd_tran = np.zeros(3)
                pcd_size = 1e-4
            pcd_sizes.append(pcd_size)
            pcd_trans.append(pcd_tran)
            
            if obj_pcd_cam.shape[0] <= self.cfg.min_pcd:
                obj_pcd_cam = torch.zeros(self.cfg.min_pcd, 3)
            elif obj_pcd_cam.shape[0] > self.cfg.min_pcd:
                kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(obj_pcd_cam[:, :3], self.cfg.min_pcd, h=5)
                obj_pcd_cam = obj_pcd_cam[kdline_fps_samples_idx]
                mask_legal[object_ind] = 1
            
            obj_pcds.append(obj_pcd_cam)
            scene_pcds.append(obj_pcd_cam*pcd_size+pcd_tran)
            

            # viz
            # points = ret["surface"][:,:3]
            # points = points / 2 * bbox_size
            # points_scaled = points
            # ones = np.ones((points_scaled.shape[0], 1))
            # points_homogeneous = np.hstack([points_scaled, ones])
            # points_transformed = (points_homogeneous @ tran_matrix_cam[object_ind].T)[:, :3]
            # trimesh.PointCloud(points_transformed).export(f"outputs/viz/pts/pointcloud_{object_ind}.ply")
            # trimesh.points.PointCloud(obj_pcd_cam, colors=[255, 0, 0]).export(f"outputs/viz/depth_pcd/depth_pcd_{object_ind}.ply")
            
            # viz crop image
            # Image.fromarray((crop_imgs[-1].numpy()*255).astype(np.uint8)).save(f"outputs/viz/crop/crop_{object_ind}.png")
            # Image.fromarray((instance_mask).astype(np.uint8)).save(f"outputs/viz/crop/mask_{object_ind}.png")            

        # scene level params
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
        if self.cfg.translation_mode == "pcd":
            # normalize scene pcd with min-max
            if mask_index.shape[0] > 0:
                # # normalize in whole scene
                # scene_pcds[mask_index] = (scene_pcds[mask_index] - pcd_scene_trans) / pcd_scene_scale
                
                # # normalize in legal object
                scene_pcds_idx = scene_pcds[mask_index].reshape(-1, 3)
                scene_center = (np.max(scene_pcds_idx, axis=0, keepdims=True) + np.min(scene_pcds_idx, axis=0, keepdims=True)) / 2
                scene_pcds_idx = scene_pcds_idx - scene_center
                scene_pcds_idx = scene_pcds_idx / np.max(np.abs(scene_pcds[:,:2]))
                scene_pcds[mask_index] = scene_pcds_idx.reshape(-1, self.cfg.min_pcd, 3)

            # calculate pcd size of objects
            pcd_trans = (np.max(scene_pcds, axis=1) + np.min(scene_pcds, axis=1)) / 2
            pcd_sizes = np.max((np.max(scene_pcds, axis=1) - np.min(scene_pcds, axis=1)), axis=-1, keepdims=True)
            # viz
            # trimesh.points.PointCloud(scene_pcds.reshape(-1,3), colors=[255, 0, 0]).export(f"outputs/viz/pcd/depth_pts_{index}.ply")
        
        elif self.cfg.translation_mode == "pcd2":
            # normalize in whole scene
            if mask_index.shape[0] > 0:
                scene_pcds[mask_index] = (scene_pcds[mask_index] - pcd_scene_trans) / pcd_scene_scale
            # calculate pcd size of objects
            pcd_trans = np.stack(pcd_trans, axis=0)
            pcd_trans = (pcd_trans - centroid) / layout_size * 2.0
            pcd_sizes = np.stack(pcd_sizes, axis=0)
            pcd_sizes = pcd_sizes / layout_size * 2.0
            pcd_sizes = np.repeat(pcd_sizes[:, np.newaxis], repeats=3, axis=-1)
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
        
        # augmentation
        if self.cfg.augmentation and self.mode=="train":
            B_p = rotation.shape[0]
            trans_aug = torch.rand(B_p, 3) * 2 - 1
            size_aug = torch.rand(B_p, 3) * 2
            rot_aug = self.random_sample_rot(B_p)
            # ramdom replace pose
            mask = torch.rand(B_p) < 0.8
            rotation = torch.where(mask.unsqueeze(-1), rot_aug, rotation)
            translation = torch.where(mask.unsqueeze(-1), trans_aug, translation)
            bbox_sizes = torch.where(mask.unsqueeze(-1), size_aug, bbox_sizes)     
        
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
                "class_list": class_list, "caption": class_list}

        return data_dict
    
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
            elif isinstance(value, np.ndarray):
                ret[key] = torch.stack([torch.from_numpy(b[key]) for b in batch])
            else:
                ret[key] = default_collate_fn_map[type(batch[0][key])](batch)
        return ret
    
    


