import json
import os
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import fpsample
import trimesh

from craftsman import register
from craftsman.utils.typing import *
from craftsman.utils.config import parse_structured
from utils.transforms import *

import pdb

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

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


def random_morphological_transform(
    mask: torch.Tensor, max_kernel_size: int = 5, p_dilation: float = 0.5
) -> torch.Tensor:
    """
    Randomly dilate or erode the mask
    :param mask: [H, W] 0-1 mask
    :param max_kernel_size: Maximum kernel size; controls the scale of the morphological operation
    :param p_dilation: Probability of dilation; if random sample > p_dilation, erosion is performed
    :return: Perturbed mask
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)

    kernel_size = np.random.randint(1, max_kernel_size + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if np.random.rand() < p_dilation:
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    else:
        mask_np = cv2.erode(mask_np, kernel, iterations=1)

    return torch.from_numpy(mask_np).float().to(mask.device)


@dataclass
class MultiObjectDataModuleConfig:
    scene_list: Any = ""
    object_list: Any = ""

    # Surface
    surface_root_dir: Any = ""
    surface_suffix: str = "npy"
    n_samples: int = 20480
    return_scene: bool = False

    max_objs: Optional[int] = None
    min_pcd: int = 1000
    padding: bool = True
    num_instances_per_batch: Optional[int] = 10
    with_sharp_data: bool = True

    # Image input
    image_root_dir: Any = ""
    image_prefix: Any = "render"
    image_suffix: str = "webp"
    idmap_prefix: str = "semantic"
    idmap_suffix: str = "png"
    depth_prefix: str = "depth"
    depth_suffix: str = "exr"
    background_color: Union[str, float] = "white"
    image_names: List[str] = field(default_factory=lambda: [])
    height: int = 768
    width: int = 768

    use_scene_image: bool = True
    remove_scene_bg: bool = False

    # Data processing
    skip_small_object: bool = False
    small_image_proportion: float = 0.005  # (16/224)^2

    ## Mask perturbation
    morph_perturb: bool = False
    max_kernel_size: int = 5
    p_dilation: float = 0.5

    return_crop_padded: bool = True
    height_crop_padded: int = 224
    width_crop_padded: int = 224

    # Mix data
    do_mix: bool = False
    do_mix_prob: float = 0.5

    mix_length: int = 80000
    mix_scene_list: str = ""
    mix_image_root_dir: str = ""
    mix_surface_root_dir: str = ""
    mix_surface_suffix: str = "npy"
    mix_image_prefix: str = "render_opaque"
    mix_image_names: List[str] = field(default_factory=lambda: [])
    mix_image_suffix: str = "webp"

    train_indices: Optional[Tuple[Any, Any]] = None
    val_indices: Optional[Tuple[Any, Any]] = None
    test_indices: Optional[Tuple[Any, Any]] = None

    repeat: int = 1

    batch_size: int = 1
    eval_batch_size: int = 1

    num_workers: int = 16
    
    # scene
    translation_mode: str = "gt"


class MultiObjectDataset(Dataset):
    def __init__(self, cfg: Any, split: str = "train") -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.cfg: MultiObjectDataModuleConfig = cfg

        self.all_scenes = _parse_scene_list(
            self.cfg.scene_list, self.cfg.surface_root_dir
        )
        self.all_objects = _parse_object_list(self.cfg.object_list)
        if len(self.all_scenes) != len(self.all_objects):
            raise ValueError(
                f"Number of scenes and objects must be the same, got {len(self.all_scenes)} scenes and {len(self.all_objects)} object lists."
            )

        self.all_images = _parse_scene_list(
            self.cfg.scene_list, self.cfg.image_root_dir
        )

        self.split = split
        self.indices = []
        if self.split == "train" and self.cfg.train_indices is not None:
            self.indices = (self.cfg.train_indices[0], self.cfg.train_indices[1])
        elif self.split == "val" and self.cfg.val_indices is not None:
            self.indices = (self.cfg.val_indices[0], self.cfg.val_indices[1])
            # self.indices = (self.cfg.val_indices[0], self.cfg.val_indices[0]+10)
        elif self.split == "test" and self.cfg.test_indices is not None:
            # self.indices = (self.cfg.test_indices[0], self.cfg.test_indices[1])
            self.indices = (self.cfg.test_indices[0], self.cfg.test_indices[0]+10)
        else:
            self.indices = (0, len(self.all_scenes))

        repeat = self.cfg.repeat if self.split == "train" else 1

        self.all_scenes = self.all_scenes[self.indices[0] : self.indices[1]] * repeat
        self.all_objects = self.all_objects[self.indices[0] : self.indices[1]] * repeat
        self.all_images = self.all_images[self.indices[0] : self.indices[1]] * repeat

        if self.cfg.do_mix:
            self.mix_all_scenes = _parse_scene_list(
                self.cfg.mix_scene_list, self.cfg.mix_surface_root_dir
            )[: self.cfg.mix_length]
            self.mix_all_images = _parse_scene_list(
                self.cfg.mix_scene_list, self.cfg.mix_image_root_dir
            )[: self.cfg.mix_length]

    def __len__(self):
        return len(self.all_scenes)

    def get_bg_color(self, bg_color):
        if bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif bg_color == "random":
            bg_color = np.random.rand(3)
        elif bg_color == "random_gray":
            bg_color = random.uniform(0.3, 0.7)
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        elif isinstance(bg_color, float):
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        elif isinstance(bg_color, list) or isinstance(bg_color, tuple):
            bg_color = np.array(bg_color, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def get_intrinsics(self, meta: dict, height: int, width: int):
        camera_lens = meta["camera_lens"]
        sensor_width = meta["sensor_width"]
        intrinsics = np.array(
            [
                [camera_lens * width / sensor_width, 0, width / 2],
                [0, camera_lens * height / sensor_width, height / 2],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        return intrinsics
    
    def reconstruct_pcd(self, depth, intrinsic, cam2wrd_rot, scale=1, normalize=False, valid_mask=None):
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
        point_cloud_incam[:, 1:3] = -point_cloud_incam[:, 1:3].copy()
        # point_cloud_incam[:,0] = -point_cloud_incam[:,0].copy() # raw data

        if normalize:
            translation = (np.max(point_cloud_incam, axis=0) + np.min(point_cloud_incam, axis=0)) / 2
            point_cloud_incam = point_cloud_incam - translation
            scale = np.max(np.abs(point_cloud_incam[:, :2]))
            point_cloud_incam = point_cloud_incam / scale
            return point_cloud_incam, translation, scale
        else:
            return point_cloud_incam

    def load_surface(self, path, num_pc: int = 20480):
        if path.endswith(".npy"):
            data = np.load(path, allow_pickle=True).tolist()
            surface = data["surface_points"]  # Nx3
            normal = data["surface_normals"]  # Nx3
        elif path.endswith(".obj") or path.endswith(".glb"):
            import trimesh

            n_surf_sample = 500000
            scene = trimesh.load(path, process=False, force="scene")
            meshes = []

            for node_name in scene.graph.nodes_geometry:
                geom_name = scene.graph[node_name][1]
                geometry = scene.geometry[geom_name]
                transform = scene.graph[node_name][0]
                if isinstance(geometry, trimesh.Trimesh):
                    geometry.apply_transform(transform)
                    meshes.append(geometry)
            mesh = trimesh.util.concatenate(meshes)
            surface, face_indices = trimesh.sample.sample_surface(
                mesh, n_surf_sample, sample_color=False
            )
            normal = mesh.face_normals[face_indices]
        else:
            raise NotImplementedError(f"Unsupported file format: {path}")

        rng = np.random.default_rng()
        ind = rng.choice(surface.shape[0], num_pc, replace=False)
        surface = torch.FloatTensor(surface[ind])
        normal = torch.FloatTensor(normal[ind])
        surface = torch.cat([surface, normal], dim=-1)

        return surface

    def load_image(
        self,
        path,
        height,
        width,
        background_color,
        rescale: bool = False,
        return_mask: bool = False,
        remove_bg: bool = False,
        idmap_path: Optional[str] = None,
    ):
        image_pil = Image.open(path).resize((width, height))
        image = torch.from_numpy(np.array(image_pil)).float() / 255.0

        if image_pil.mode == "RGBA":
            image_bg = image[:, :, :3] * image[:, :, 3:4] + background_color * (
                1 - image[:, :, 3:4]
            )
            mask = (image[:, :, 3] > 0.5).float()
        elif remove_bg and idmap_path is not None:
            id_map = torch.from_numpy(
                np.array(Image.open(idmap_path).resize((width, height), Image.NEAREST))
            )
            mask = (id_map > 0).float()
            mask_ = mask.unsqueeze(-1).repeat(1, 1, 3)
            image_bg = image * mask_ + background_color * (1 - mask_)
        else:
            image_bg = image
            mask = torch.ones_like(image[:, :, 0]).float()

        if rescale:
            image_bg = image_bg * 2.0 - 1.0
        if return_mask:
            return image_bg, mask
        return image_bg

    def load_parts(
        self,
        rgb_path: str,
        idmap_path: str,
        depth_path: str,
        meta: dict,
        indexes: List[int],
        height: int,
        width: int,
        background_color: torch.Tensor,
        skip_small_object: bool = False,
        small_image_proportion: float = 0.005,
        morph_perturb: bool = False,  # Whether to apply morphological perturbation
        max_kernel_size: int = 5,
        p_dilation: float = 0.5,
    ):
        rgb_image = self.load_image(rgb_path, height, width, background_color)
        id_map = torch.from_numpy(np.array(Image.open(idmap_path).resize((width, height), Image.NEAREST)))
        
        # depth
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = depth / depth.max() * 10  # Normalize depth to [0, 1]
            
        height, width, _ = rgb_image.shape
        intrinsics = self.get_intrinsics(meta, height, width)
        
        mask_legal = torch.zeros(self.cfg.max_objs)
        rgb_list, mask_list = [], []
        scene_pcd_list, obj_pcd_list = [], []
        pcd_size_list, pcd_trans_list = [], []
        rotation_list, translation_list = [], []
        for object_ind, idx in enumerate(indexes):
            # break if number of objects exceeds max_objs
            if object_ind >= self.cfg.max_objs:
                break
            
            # mask
            mask = (id_map == idx).float()
            if morph_perturb:
                mask = random_morphological_transform(mask, max_kernel_size=max_kernel_size, p_dilation=p_dilation)               
                
            # reconstruct pcd
            try:
                obj_pcd_cam, pcd_tran, pcd_size = self.reconstruct_pcd(depth=depth, intrinsic=intrinsics, cam2wrd_rot=None, scale=1, normalize=True, valid_mask=mask.cpu().bool().numpy())
            except:
                obj_pcd_cam = torch.zeros(self.cfg.min_pcd, 3)
                pcd_tran = np.zeros(3)
                pcd_size = 1e-4
            
            if obj_pcd_cam.shape[0] <= self.cfg.min_pcd:
                obj_pcd_cam = torch.zeros(self.cfg.min_pcd, 3)
            elif obj_pcd_cam.shape[0] > self.cfg.min_pcd:
                kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(obj_pcd_cam[:, :3], self.cfg.min_pcd, h=5)
                obj_pcd_cam = obj_pcd_cam[kdline_fps_samples_idx]
                mask_legal[object_ind] = 1.0
            
            pcd_size_list.append(pcd_size)
            pcd_trans_list.append(pcd_tran)
            obj_pcd_list.append(obj_pcd_cam)
            scene_pcd_list.append(obj_pcd_cam*pcd_size+pcd_tran)

            # save mask
            mask_3c = mask.unsqueeze(-1).repeat(1, 1, 3)
            part_rgb = rgb_image * mask_3c + background_color * (1 - mask_3c)
            rgb_list.append(part_rgb)
            mask_list.append(mask)
            
            # pose
            transformation = np.array(meta["locations"][object_ind]['transform_matrix'])
            rotation_list.append(transformation[:3, :3])
            translation_list.append(transformation[:3, 3])
            
            # filter small objects
            if (skip_small_object and mask.sum() <= small_image_proportion * height * width):
                mask_legal[object_ind] = 0.0
            
            # viz
            # Image.fromarray((part_rgb*255).numpy().astype(np.uint8)).save(f"outputs/viz/mask_{object_ind}.png") 
            # trimesh.points.PointCloud(obj_pcd_cam.reshape(-1,3), colors=[255, 0, 0]).export(f"outputs/viz/depth_pcd/scene_pcd_{object_ind}.ply")

        return rgb_list, mask_list, pcd_size_list, pcd_trans_list, obj_pcd_list, scene_pcd_list, rotation_list, translation_list, mask_legal

    def crop_and_pad(self, rgbs, masks, height, width, padding_ratio=0.1):
        cropped_rgbs, cropped_masks = [], []

        for rgb, mask in zip(rgbs, masks):
            rgb = rgb.permute(2, 0, 1)

            # crop
            coords = torch.nonzero(mask == 1)
            y_min, x_min = coords.min(dim=0).values
            y_max, x_max = coords.max(dim=0).values

            cropped_rgb = rgb[:, y_min : y_max + 1, x_min : x_max + 1]
            cropped_mask = mask[y_min : y_max + 1, x_min : x_max + 1]

            h, w = cropped_rgb.shape[1:]

            # padding
            padding_size = [0, 0, 0, 0]  # left, right, top, bottom
            if w > h:
                padding_size[2] = padding_size[3] = int((w - h) / 2)
                h = w
            else:
                padding_size[0] = padding_size[1] = int((h - w) / 2)
                w = h

            padding_size = tuple([s + int(w * padding_ratio) for s in padding_size])
            padded_rgb = F.pad(cropped_rgb, padding_size, mode="constant", value=1)
            padded_mask = F.pad(cropped_mask, padding_size, mode="constant", value=0)

            # resize
            padded_rgb = F.interpolate(
                padded_rgb.unsqueeze(0), (height, width), mode="bilinear"
            )[0]
            padded_mask = F.interpolate(
                padded_mask.unsqueeze(0).unsqueeze(0), (height, width), mode="nearest"
            )[0][0]

            cropped_rgbs.append(padded_rgb)
            cropped_masks.append(padded_mask)

        return cropped_rgbs, cropped_masks

    def normalize_pts(self, pts):
        """
        Normalize points to the range [-1, 1] and return the size of the bounding box.
        :param pts: (N, 3) or (N, 6) tensor of points
        :return: normalized points, size of the bounding box
        """
        min_pt = pts.min(dim=0).values
        max_pt = pts.max(dim=0).values
        size = max_pt - min_pt
        center = (max_pt + min_pt) / 2.0
        pts_normalized = 1.9 * (pts - center) / size.max()
        
        return pts_normalized, size.max(), center

    def __getitem__(self, index):        
        # Background color
        background_color = torch.as_tensor(self.get_bg_color(self.cfg.background_color))

        # Surface
        scene = self.all_scenes[index]
        scene_objects = self.all_objects[index]
        surfaces, sizes, translation = [], [], []
        for scene_object in scene_objects:
            surface_path = os.path.join(scene, f"{scene_object}.{self.cfg.surface_suffix}")
            surface = self.load_surface(surface_path, self.cfg.n_samples)
            surface[:,:3], size, trans = self.normalize_pts(surface[:,:3])
            surfaces.append(surface)
            sizes.append(size)
            translation.append(trans)
        surfaces = torch.stack(surfaces)  # (num_instances, num_points, 6)
        num_instances = surfaces.shape[0]
        sharp_surfaces = surfaces.clone()
        sizes = torch.tensor(sizes, dtype=torch.float32).unsqueeze(1)  # (num_instances, 1)
        translation = torch.stack(translation)  # (num_instances, 3)
        # trimesh.points.PointCloud(surface[:,:3].reshape(-1,3), colors=[255, 0, 0]).export(f"outputs/viz/depth_pcd/scene_pcd.ply")

        # Image
        image_dir = self.all_images[index]
        image_name = (
            random.choice(self.cfg.image_names)
            if self.split == "train"
            else self.cfg.image_names[0]
        )
        image_prefix = (
            [self.cfg.image_prefix]
            if isinstance(self.cfg.image_prefix, str)
            else self.cfg.image_prefix
        )
        image_prefix = (
            random.choice(image_prefix) if self.split == "train" else image_prefix[0]
        )
        image_path = os.path.join(
            image_dir, f"{image_prefix}_{image_name}.{self.cfg.image_suffix}"
        )
        idmap_path = (
            os.path.join(
                image_dir,
                f"{self.cfg.idmap_prefix}_{image_name}.{self.cfg.idmap_suffix}",
            )
            .replace("_controlnet", "")
            .replace("_inpaint", "")
        )
        depth_path = os.path.join(
            image_dir, f"{self.cfg.depth_prefix}_{image_name}.{self.cfg.depth_suffix}",
        )
        
        # load meta information
        meta_path = os.path.join(image_dir, f"meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)

        # Load image and parts
        rgb_scene = (
            self.load_image(
                image_path,
                height=self.cfg.height,
                width=self.cfg.width,
                background_color=background_color,
                remove_bg=self.cfg.remove_scene_bg,
                idmap_path=idmap_path,
            )
        )
        rgbs, masks, pcd_size_list, pcd_trans_list, obj_pcd_list, scene_pcd_list, rotation_list, translation_list, mask_legal = self.load_parts(
            image_path,
            idmap_path,
            depth_path,
            meta,
            list(range(1, num_instances + 1)),
            self.cfg.height,
            self.cfg.width,
            background_color,
            skip_small_object=self.cfg.skip_small_object,
            small_image_proportion=self.cfg.small_image_proportion,
            morph_perturb=self.cfg.morph_perturb,
            max_kernel_size=self.cfg.max_kernel_size,
            p_dilation=self.cfg.p_dilation,
        )
        if len(rgbs) == 0:
            return self._getitem(random.randint(0, self.__len__() - 1))
        
        # Convert to tensors
        mask_legal = mask_legal.bool()
        rgb = torch.stack(rgbs).permute(0, 3, 1, 2).float() 
        masks = torch.stack(masks).float() 
        pcd_sizes = torch.from_numpy(np.stack(pcd_size_list)).unsqueeze(-1).float()   # (num_instances, 1)
        pcd_trans = torch.from_numpy(np.stack(pcd_trans_list)).float() 
        obj_pcds = torch.from_numpy(np.stack(obj_pcd_list)).float() 
        scene_pcds = torch.from_numpy(np.stack(scene_pcd_list)).float() 
        rotation = mat2repr6d(torch.eye(3).unsqueeze(0).repeat(len(rotation_list), 1, 1)).float() 
        # rotation = torch.from_numpy(mat2repr6d(np.stack(rotation_list)))
        # translation = torch.from_numpy(np.stack(translation_list))
        
        # normalize scene pcd
        mask_index = torch.nonzero(mask_legal).squeeze(-1)
        if mask_index.shape[0] > 0:
            scene_pcds_idx = scene_pcds[mask_index].reshape(-1, 3)
            scene_center = (torch.max(scene_pcds_idx, dim=0, keepdims=True)[0] + torch.min(scene_pcds_idx, dim=0, keepdims=True)[0]) / 2
            scene_pcds_idx = scene_pcds_idx - scene_center
            scene_pcds_idx = scene_pcds_idx / torch.max(torch.abs(scene_pcds_idx))
            scene_pcds[mask_index] = scene_pcds_idx.reshape(-1, self.cfg.min_pcd, 3)
        pcd_trans = (torch.max(scene_pcds, dim=1).values + torch.min(scene_pcds, dim=1).values) / 2.0
        pcd_sizes = torch.max(torch.max(scene_pcds, dim=1)[0] - torch.min(scene_pcds, dim=1)[0], dim=-1, keepdim=True).values + 1e-4  # (num_instances, 1)
        
        # # viz
        # trimesh.points.PointCloud(scene_pcds.reshape(-1,3), colors=[255, 0, 0]).export(f"outputs/viz/depth_pcd/scene_pcd.ply")

        # crop object image
        crop_imgs, crop_masks = self.crop_and_pad(rgbs, masks, self.cfg.height, self.cfg.width)
        crop_imgs = torch.stack(crop_imgs).permute(0, 2, 3, 1).float() 
        crop_masks = torch.stack(crop_masks).float() 
        
        # Scene id
        scene_id = "-".join(image_dir.split("/")[-2:])
        
        # padding to max_objs
        n_objects = obj_pcds.shape[0]
        class_list = [""] * n_objects
        if n_objects < self.cfg.max_objs:
            pad_len = self.cfg.max_objs - n_objects
            mask_legal = F.pad(mask_legal, (0, pad_len), value=False)
            surfaces = F.pad(surfaces, (0, 0, 0, 0, 0, pad_len))
            sharp_surfaces = F.pad(sharp_surfaces, (0, 0, 0, 0, 0, pad_len))
            crop_imgs = F.pad(crop_imgs, (0, 0, 0, 0, 0, 0, 0, pad_len))
            masks = F.pad(masks, (0, 0, 0, 0, 0, pad_len))
            sizes = F.pad(sizes, (0, 0, 0, pad_len))
            obj_pcds = F.pad(obj_pcds, (0, 0, 0, 0, 0, pad_len))
            rotation = F.pad(rotation, (0, 0, 0, pad_len))
            translation = F.pad(translation, (0, 0, 0, pad_len))     
            scene_pcds = F.pad(scene_pcds, (0, 0, 0, 0, 0, pad_len))
            pcd_sizes = F.pad(pcd_sizes, (0, 0, 0, pad_len))
            pcd_trans = F.pad(pcd_trans, (0, 0, 0, pad_len))
            class_list = class_list + [""] * pad_len
        # remove extra objects         
        elif n_objects > self.cfg.max_objs:
            mask_legal = mask_legal[:self.cfg.max_objs]
            surfaces = surfaces[:self.cfg.max_objs]
            sharp_surfaces = sharp_surfaces[:self.cfg.max_objs]
            crop_imgs = crop_imgs[:self.cfg.max_objs]
            masks = masks[:self.cfg.max_objs]
            sizes = sizes[:self.cfg.max_objs]
            obj_pcds = obj_pcds[:self.cfg.max_objs]
            rotation = rotation[:self.cfg.max_objs]
            translation = translation[:self.cfg.max_objs]
            scene_pcds = scene_pcds[:self.cfg.max_objs]
            pcd_sizes = pcd_sizes[:self.cfg.max_objs]
            pcd_trans = pcd_trans[:self.cfg.max_objs]
            class_list = class_list[:self.cfg.max_objs]

        data_dict = {"whole_img": rgb_scene, "pose": rotation, "translation": translation, "size": sizes, "surface": surfaces, "sharp_surface": sharp_surfaces, \
                "image": crop_imgs, "obj_pcds": obj_pcds, "scene_pcds": scene_pcds, "pcd_sizes": pcd_sizes, "pcd_trans": pcd_trans, "mask_legal": mask_legal.bool(), \
                "taskid": scene_id, "masks": masks, "class_list": class_list, "caption": class_list}

        return data_dict

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


@register("midi-datamodule")
class MultiObjectDataModule(pl.LightningDataModule):
    cfg: MultiObjectDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiObjectDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = MultiObjectDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MultiObjectDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = MultiObjectDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.val_dataset.collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.test_dataset.collate,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


if __name__ == "__main__":
    import torchvision
    from omegaconf import OmegaConf

    config_file = "configs/scenediff/training.yaml"
    data_cfg = OmegaConf.load(config_file)["data"]
    cfg: MultiObjectDataModuleConfig = MultiObjectDataModuleConfig(**data_cfg)
    data_module = MultiObjectDataModule(cfg)
    data_module.setup()

    for batch in data_module.test_dataloader():
        print(batch["num_instances"])

        for key in [
            "rgb",
            "mask",
            "rgb_scene",
            # "rgb_crop_padded",
            # "mask_crop_padded",
        ]:
            print(key, batch[key].shape, batch[key].min(), batch[key].max())
            torchvision.utils.save_image(
                batch[key], f"tmp/{key}.png", nrow=4, normalize=True
            )

        for key in ["rgb"]:
            for i in range(batch[key].shape[0]):
                torchvision.utils.save_image(
                    batch[key][i], f"tmp/{key}_{i}.png", normalize=True
                )

        break
