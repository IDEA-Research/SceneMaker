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
from utils.space_transform import reconstruct_pcd, get_rot_from_pitch
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
                          "dresser": 8
                          }

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
    data_path: str = '/comp_robot/shiyukai/datasets/instPiFU/datasets/prepare_data'
    split_dir: str = 'data/3dfront/split'
    occ_path: str = '/comp_robot/shiyukai/datasets/instPiFU/datasets/occ'
    mask_path: str = '/comp_robot/shiyukai/datasets/instPiFU/datasets/mask'
    geo_data_path: str = 'data/samples/sample_sampling_objects'
    avg_layout_path: str = 'data/3dfront/avg_layout.pkl'
    
    ################################# Scene argumentation #################################
    max_objs: int = 5
    min_pcd: int = 1000
    translation_mode: str = "pcd"
    refine_mask: bool = False

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

class Front3DProcessDataset(Dataset):
    def __init__(self, config, mode):
        super(Front3DProcessDataset, self).__init__()
        self.mode=mode
        self.cfg: Front3DReconDataModuleConfig = config

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
            self.split = self.split[:10]

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
        
        # load image
        image = sequence['rgb_img']
        if image.shape[2] == 4:
            image = image[:,:,:3]
        width, height = image.shape[1], image.shape[0]
        
        # load depth map and pts
        depth = sequence['depth_map']/255.0*10
        pitch_rot = get_rot_from_pitch(layout["pitch"])
        point_cloud_incam = reconstruct_pcd(depth=depth, cam2wrd_rot=pitch_rot, intrinsic=camera["K"], scale=1)
        
        # viz
        # Image.fromarray(image).save("outputs/viz/rgb.png")
        # Image.fromarray(sequence['depth_map']).save("outputs/viz/depth.png")
        # trimesh.points.PointCloud(point_cloud_incam, colors=[255, 0, 0]).export("outputs/viz/depth_pts.ply")
        
        # transformation under cam pose
        tran_matrix = boxes["tran_matrix"]
        wrd2cam_matrix = sequence['camera']['wrd2cam_matrix']
        org_K = sequence['camera']['K'].copy()
        tran_matrix_cam = (wrd2cam_matrix[None,...] @ tran_matrix)
        # mask for legal object
        mask_legal = torch.zeros(self.cfg.max_objs)
        
        geo_shape, geo_shape_sharp = [], []
        crop_imgs, masks = [], []
        obj_pcds, scene_pcds = [], []
        bbox_sizes = []
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
            instance_mask_path = os.path.join(self.cfg.mask_path, "%s_%s.png" % (taskid, object_ind))
            instance_mask = np.array(Image.open(instance_mask_path))
            masks.append(torch.from_numpy(instance_mask))
            
            # crop image
            crop_imgs.append(torch.from_numpy(self.crop_image(image, instance_mask)/255.0).float())
            
            # refine mask with rembg
            if self.cfg.refine_mask:
                instance_mask = self.refine_instance_mask(image=image, instance_mask=instance_mask)
                # Image.fromarray((instance_mask).astype(np.uint8)).save(instance_mask_path.replace(".png", "_refine.png"))             

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
        

        data_dict = {}

        return data_dict
    
    def collate(self, batch):
        from torch.utils.data._utils.collate import default_collate_fn_map
        return torch.utils.data.default_collate(batch)
    
    


