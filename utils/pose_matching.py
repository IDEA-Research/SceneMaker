
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from craftsman import CraftsManPipeline
import trimesh
import numpy as np
from scipy.spatial import KDTree
import argparse
import torch
from PIL import Image
from sklearn.utils import resample
# import pyrender
import fpsample
import torch.nn.functional as F
import rembg
import json
import cv2
import open3d as o3d

from utils.space_transform import get_yaw_from_R, get_rot_from_yaw

trimesh.util.attach_to_log()
trimesh.rendering.offscreen = True
os.environ["PYOPENGL_PLATFORM"] = "egl"

device = "cuda" if torch.cuda.is_available() else "cpu"

import pdb

def numerical_sort_key(filename):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

def convert_to_python(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def normalize_points(points):
    """Normalize the point cloud to keep XY in [-1, 1] while preserving relative XYZ scale."""
    # calculate pcd size of objects
    centroid = (np.max(points, axis=0) + np.min(points, axis=0)) / 2           
    points_centered = points - centroid  # translate to center
    xy_max = np.max(np.abs(points_centered[:, :2]))  # maximum absolute value on XY
    normalized_points = points_centered / xy_max * 0.95 # scale point cloud

    return normalized_points, centroid, xy_max

def crop_mask(mask, resolution=256):
    # crop  
    mask = Image.fromarray(mask)
    mask = mask.crop(mask.getbbox())
    
    # pad image to 1:1
    width, height = mask.size
    new_size = (width, width) if width > height else (height, height)
    if width > height: 
        new_size = (width, width)
        new_image = Image.new("L", new_size, (0,))
        new_image.paste(mask, (0, (width - height) // 2))
    else:
        new_size = (height, height)
        new_image = Image.new("L", new_size, (0,))
        new_image.paste(mask, ((height - width) // 2, 0))
    new_image.resize((resolution, resolution))
    
    return np.array(new_image)

def crop_square_with_padding(image, mask, bbox_ratio=0.9, pad_color=(255,255,255,255)):
    from PIL import Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)

    coordinates = mask.getbbox()
    if coordinates is None:
        return image

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


def cleanup_pcd(points: np.ndarray, num_nb: int = 15, std_ratio: float = 2.0,):
    # voxelize the point cloud
    pcd = o3d.geometry.PointCloud()
    # transform points to Open3D format
    pcd.points = o3d.utility.Vector3dVector(points)
    # remove outliers
    pcd, _ = pcd.remove_statistical_outlier(num_nb, std_ratio=std_ratio)
    # transform to numpy array
    pcd = np.asarray(pcd.points)
    
    return pcd

def reconstruct_pcd(depth, intrinsic, scale=1, normalize=False, valid_mask=None, depth_mode="depth-anything"):
    """
    Reconstruct a point cloud from a depth map, with optional support for querying based on a mask.

    Args:
        depth (np.ndarray): Depth map.
        intrinsic (np.ndarray): Camera intrinsic matrix.
        cam2wrd_rot (np.ndarray): Camera-to-world rotation matrix.
        scale (float): Scale factor for the intrinsic matrix.
        normalize (bool): Whether to normalize the point cloud.
        valid_mask (np.ndarray, optional): Mask to filter valid points in the depth map.

    Returns:
        tuple: (point_cloud, point_cloud_incam)
            - point_cloud: Reconstructed point cloud in world coordinates.
            - point_cloud_incam: Reconstructed point cloud in camera coordinates.
    """
    intrinsic[0] = intrinsic[0] / scale
    intrinsic[1] = intrinsic[1] / scale

    # Apply mask if provided
    # depth = depth / depth.max() * 10.0
    # if valid_mask is not None:
    #     valid_Y, valid_X = np.where((depth > 0) & (depth < 10) & valid_mask)
    # else:
    #     valid_Y, valid_X = np.where((depth > 0) & (depth < 10))
    
    if valid_mask is not None:
        valid_Y, valid_X = np.where(valid_mask)
    else:
        valid_Y, valid_X = np.where((depth > 0))

    max_num_points = 10000
    if valid_Y.shape[0] > max_num_points:
        random_ind = np.random.choice(valid_Y.shape[0], max_num_points, replace=False)
        valid_Y = valid_Y[random_ind]
        valid_X = valid_X[random_ind]
    valid_Z = depth[valid_Y, valid_X]

    unprojected_Y = valid_Y * depth[valid_Y, valid_X]
    unprojected_X = valid_X * depth[valid_Y, valid_X]
    unprojected_Z = depth[valid_Y, valid_X]
    point_cloud_xyz = np.concatenate(
        [unprojected_X[:, np.newaxis], unprojected_Y[:, np.newaxis], unprojected_Z[:, np.newaxis]], axis=1
    )
    
    # fx = intrinsic[0, 0]
    # fy = intrinsic[1, 1]
    # cx = intrinsic[0, 2]
    # cy = intrinsic[1, 2]

    # x_valid = (valid_X - cx) * valid_Z / fx
    # y_valid = (valid_Y - cy) * valid_Z / fy
    # point_cloud_xyz = np.stack((x_valid, y_valid, valid_Z), axis=-1)

    intrinsic_inv = np.linalg.inv(intrinsic[0:3, 0:3])
    point_cloud_xyz = np.dot(intrinsic_inv, point_cloud_xyz.T).T
    point_cloud_incam = point_cloud_xyz[:, :3].copy()

    # reverse the all axis
    point_cloud_incam = -point_cloud_incam
    if depth_mode == "moge":
        point_cloud_incam[:, -1] = -point_cloud_incam[:, -1]
        # point_cloud_incam[:, 0] = -point_cloud_incam[:, 0]
        # point_cloud_incam[:, -1] = -point_cloud_incam[:, -1]

    if normalize:
        translation = (np.max(point_cloud_incam, axis=0) + np.min(point_cloud_incam, axis=0)) / 2
        point_cloud_incam = point_cloud_incam - translation
        scale = np.max(np.abs(point_cloud_incam[:, :2]))
        point_cloud_incam = point_cloud_incam / scale * 0.95
        return point_cloud_incam, translation, scale
    else:
        return point_cloud_incam

def get_intrinsics(depth):
    intrinsics = np.zeros((3, 3))
    h, w = depth.shape
    intrinsics[0, 0] = 1168
    intrinsics[1, 1] = 1168
    intrinsics[0, 2] = w
    intrinsics[1, 2] = h
    intrinsics[2, 2] = 1
    
    return intrinsics

def dilate_mask(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
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

def refine_instance_mask(image: np.ndarray, instance_mask: np.ndarray):
    # Step 1: Get bounding box of the instance mask
    ys, xs = np.where(instance_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return instance_mask  # skip empty mask

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Step 2: Crop image
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]

    # Step 2.1: Pad to 1:1
    h, w, _ = cropped_image.shape
    side = max(h, w)
    pad_top = (side - h) // 2
    pad_bottom = side - h - pad_top
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left

    padded_image = np.pad(
        cropped_image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    # Step 3: Apply rembg to get new alpha mask
    padded_pil = Image.fromarray(padded_image).convert("RGBA")
    removed = rembg.remove(padded_pil)
    alpha = np.array(removed.split()[-1])
    new_mask = (alpha > 128).astype(np.uint8)  # binary mask

    # Step 4: Remove padding to match original crop size
    unpadded_mask = new_mask[pad_top:side - pad_bottom, pad_left:side - pad_right]

    # Step 5: Project new mask back to full image
    refined_mask = np.zeros_like(instance_mask, dtype=np.uint8)
    refined_mask[y_min:y_max+1, x_min:x_max+1] = unpadded_mask & instance_mask[y_min:y_max+1, x_min:x_max+1]

    return refined_mask * 255  # for visualization or consistency


def prepare_data(pipeline, mesh_paths, depth_map, output_path, batch_gt=None, intrinsics=None, \
                 depth_mode="moge", pcd_mode="pcd2", pred_mode="pose", num_objs=-1):
    # pcd path
    pcd_path = os.path.join(output_path, "depth_pcd")
    os.makedirs(pcd_path, exist_ok=True)
    
    # scene img
    scene_img = np.array(Image.open(os.path.join(output_path, "scene_image.png")).convert("RGB"))
    width, height = scene_img.shape[1], scene_img.shape[0]
    
    # depth mask
    if os.path.exists(os.path.join(output_path, "depth_mask.png")):
        depth_mask = np.array(Image.open(os.path.join(output_path, "depth_mask.png"))).astype(np.uint8)
    else:
        depth_mask = np.ones_like(depth_map) * 255.0
    
    # mask for legal object
    n_objects = len(mesh_paths)
    max_objs = pipeline.cfg.data.max_objs if num_objs == -1 else num_objs
    min_pcd = pipeline.cfg.data.min_pcd
    mask_legal = torch.zeros(max_objs)
    geo_shape, crop_imgs, obj_pcds, bbox_sizes, translations, scene_pcds, masks = [], [], [], [], [], [], []
    deocc_imgs = []
    for object_ind, mesh_path in enumerate(mesh_paths):
        if object_ind >= max_objs:
            break
        flag = 1
        
        # load mask
        mask = np.array(Image.open(mesh_path.replace("3d_models", "masks").replace("masked_image", "mask").replace(".obj", ".png").replace(".glb", ".png")))
        # dialate mask
        mask = ~dilate_mask(~mask, kernel_size=3, iterations=1)
        mask = np.where(mask > 0, depth_mask, 0) # mask with depth
        Image.fromarray(mask.astype(np.uint8)).save(mesh_path.replace("3d_models", "masks").replace("masked_image", "mask_refine").replace(".obj", ".png").replace(".glb", ".png"))
        masks.append(torch.from_numpy(mask)/255.0)
        
        # crop from scene image
        try:
            rgb, cropped_mask = crop_square_with_padding(image=scene_img, mask=mask, bbox_ratio=0.8)
            # apply mask with white background
            white_bg = np.ones_like(rgb) * 255
            rgb = rgb * (cropped_mask[..., None].astype(np.float32)/255) + white_bg * (1 - cropped_mask[..., None].astype(np.float32)/255) 
        except:
            flag = 0
            rgb = np.ones((512,512,3))
        crop_imgs.append(rgb/255.0)
        
        # load mesh
        num_samples = pipeline.cfg.data.n_samples
        try:
            mesh_data = trimesh.load(mesh_path)
            
            # Handle both single mesh and scene objects
            if isinstance(mesh_data, trimesh.Scene):
                # For GLB files, combine all geometries in the scene
                combined_vertices = []
                combined_faces = []
                vertex_offset = 0
                
                for geometry in mesh_data.geometry.values():
                    if hasattr(geometry, 'vertices') and hasattr(geometry, 'faces'):
                        combined_vertices.append(geometry.vertices)
                        # Adjust face indices for combined mesh
                        faces_adjusted = geometry.faces + vertex_offset
                        combined_faces.append(faces_adjusted)
                        vertex_offset += len(geometry.vertices)
                
                if combined_vertices:
                    # Create a single mesh from combined geometries
                    combined_vertices = np.vstack(combined_vertices)
                    combined_faces = np.vstack(combined_faces)
                    mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)
                else:
                    raise ValueError("No valid geometry found in scene")
            else:
                # Single mesh object
                mesh = mesh_data
            
            # sample points and corresponding normals on mesh
            points, face_indices = mesh.sample(num_samples, return_index=True)
            normals = mesh.face_normals[face_indices]
            surface = np.concatenate([points, normals], axis=-1)
        except:
            surface = np.zeros((num_samples, 6))
            flag = 0
        geo_shape.append(surface)
        
        # pcd
        try:
            obj_pcd_cam = reconstruct_pcd(depth_map, intrinsic=intrinsics, scale=1, normalize=False, valid_mask=mask, depth_mode=depth_mode)
            obj_pcd_cam = cleanup_pcd(obj_pcd_cam, num_nb=15, std_ratio=2.0) # cleanup pcd 
            # trimesh.points.PointCloud(normalize_points(obj_pcd_cam)[0], colors=[255, 0, 0]).export(os.path.join(pcd_path, f"{fname}.ply"))
            obj_pcd_cam, points_center, points_scale = normalize_points(obj_pcd_cam)
        except:
            obj_pcd_cam = np.zeros((min_pcd, 3))
            flag = 0

        # viz depth pcd
        fname = os.path.splitext(os.path.basename(mesh_path))[0]
        trimesh.points.PointCloud(obj_pcd_cam, colors=[255, 0, 0]).export(os.path.join(pcd_path, f"{fname}.ply"))
        

        if obj_pcd_cam.shape[0] <= min_pcd:
            obj_pcd_cam = np.zeros((min_pcd, 3))
            points_center = np.zeros(3,)
            points_scale = 1e-4
            flag = 0
        elif obj_pcd_cam.shape[0] > min_pcd:
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(obj_pcd_cam[:, :3], min_pcd, h=5)
            obj_pcd_cam = obj_pcd_cam[kdline_fps_samples_idx]
        
        mask_legal[object_ind] = flag
        obj_pcds.append(obj_pcd_cam)
        bbox_sizes.append(points_scale*np.ones(1,))
        translations.append(points_center)
        scene_pcds.append(obj_pcd_cam * points_scale + points_center)
        
        # deocc image
        if pred_mode == "scene":
            # apply mask to scene image
            deocc_image = scene_img * (mask[..., None].astype(np.float32)/255)
            deocc_imgs.append(deocc_image/255.0)
    
    # scene-level normalize
    scene_pcds = np.stack(scene_pcds)
    translations = np.stack(translations)
    bbox_sizes = np.stack(bbox_sizes)
    mask_index = torch.nonzero(mask_legal).numpy().squeeze(-1)
    if mask_index.shape[0] > 0:
        scene_pcds_legal = scene_pcds[mask_index].reshape(-1,3)
        pcd_trans = (np.max(scene_pcds_legal, axis=0) + np.min(scene_pcds_legal, axis=0)) / 2
        pcd_sizes = np.max(np.abs(scene_pcds_legal - pcd_trans))
        # modify scene pcds, translations, bbox_sizes
        # scene_pcds[mask_index] = ((scene_pcds[mask_index] - pcd_trans) / pcd_sizes * 0.95).reshape(-1,min_pcd,3)  
        scene_pcds = ((scene_pcds - pcd_trans) / pcd_sizes * 0.95).reshape(-1,min_pcd,3)  
        translations = (translations - pcd_trans[None,...]) / pcd_sizes[None,...] * 0.95
        bbox_sizes = bbox_sizes / pcd_sizes[None,...] * 0.95
    trimesh.points.PointCloud(scene_pcds.reshape(-1,3), colors=[0, 255, 0]).export(os.path.join(output_path, f"scene_pcd.ply")) # viz scene pcd
    
    # transfer to tensor
    scene_pcds = torch.from_numpy(scene_pcds).float()
    scene_img = torch.from_numpy(scene_img/255.0).float()
    crop_imgs = torch.from_numpy(np.stack(crop_imgs)).float()
    geo_shape = torch.from_numpy(np.stack(geo_shape)).float()
    obj_pcds = torch.from_numpy(np.stack(obj_pcds)).float()
    translations = torch.from_numpy(translations).float()
    bbox_sizes = torch.from_numpy(bbox_sizes).float()
    masks = torch.stack(masks)
    deocc_imgs = torch.from_numpy(np.stack(deocc_imgs)).float() if len(deocc_imgs) > 0 else torch.zeros(1, height, width, 3)
    
    # # apply mask on scene image
    # mask_union = torch.any(masks.bool(), dim=0)  # [H, W]
    # scene_img = scene_img * mask_union.unsqueeze(-1).float() + (1 - mask_union.unsqueeze(-1).float())  # white background
    # Image.fromarray((scene_img.numpy()*255).astype(np.uint8)).save(os.path.join(output_path, f"scene_image_masked.png"))

    # resize and crop image
    scene_img = F.interpolate(scene_img.permute(2, 0, 1).unsqueeze(0), size=(pipeline.cfg.data.image_height, pipeline.cfg.data.image_width), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0)
    masks = F.interpolate(masks.unsqueeze(1).float(), size=(pipeline.cfg.data.image_height, pipeline.cfg.data.image_width), mode="nearest").squeeze(1)     
    
    # padding to max_objs
    if n_objects < max_objs:
        pad_len = max_objs - n_objects
        geo_shape = F.pad(geo_shape, (0, 0, 0, 0, 0, pad_len))
        crop_imgs = F.pad(crop_imgs, (0, 0, 0, 0, 0, 0, 0, pad_len))
        bbox_sizes = F.pad(bbox_sizes, (0, 0, 0, pad_len))
        obj_pcds = F.pad(obj_pcds, (0, 0, 0, 0, 0, pad_len))
        translations = F.pad(translations, (0, 0, 0, pad_len))  
        masks = F.pad(masks, (0, 0, 0, 0, 0, pad_len))  
        scene_pcds = F.pad(scene_pcds, (0, 0, 0, 0, 0, pad_len))
        deocc_imgs = F.pad(deocc_imgs, (0, 0, 0, 0, 0, 0, 0, pad_len))
    # remove extra objects         
    elif n_objects > max_objs:
        geo_shape = geo_shape[:max_objs]
        crop_imgs = crop_imgs[:max_objs]
        bbox_sizes = bbox_sizes[:max_objs]
        obj_pcds = obj_pcds[:max_objs]
        translations = translations[:max_objs]
        mask_legal = mask_legal[:max_objs]
        masks = masks[:max_objs]
        deocc_imgs = deocc_imgs[:max_objs]
        
    # pcd sizes and translations
    pcd_sizes = bbox_sizes.clone()
    pcd_trans = translations.clone()
    bbox_sizes = bbox_sizes * 2  # to [0,2]

    data_dict = {"whole_img": scene_img.unsqueeze(0).to(device), "translation": translations.unsqueeze(0).to(device), "size": bbox_sizes.unsqueeze(0).to(device), \
        "surface": geo_shape.unsqueeze(0).to(device),  \
        "image": crop_imgs.unsqueeze(0).to(device), "obj_pcds": obj_pcds.unsqueeze(0).to(device), "mask_legal": mask_legal.unsqueeze(0).to(device), \
        "scene_pcds": scene_pcds, "pcd_sizes":pcd_sizes.unsqueeze(0).to(device), "pcd_trans": pcd_trans.unsqueeze(0).to(device), "masks": masks.unsqueeze(0).to(device), \
        "caption": batch_gt["caption"] if batch_gt is not None else None}

    if pred_mode == "scene":
        data_dict["deocc_images"] = deocc_imgs.unsqueeze(0).to(device)
    # data_dict["sharp_surface"] = geo_shape.unsqueeze(0).to(device)
    
    return data_dict

@torch.no_grad()
def match_mesh_to_pointcloud(pipeline, batch, mesh_paths, batch_gt=None, pcd_pose=False, pred_mode="pose", use_gt_depth=False, use_direct_pose=False, only_pitch=False, num_objs=-1):
    # gt pose
    # pred_pose, pred_trans, pred_size = batch_gt["pose"], batch_gt["translation"], batch_gt["size"]

    # input of gt depth
    if use_gt_depth:
        for key in batch_gt.keys():
            if isinstance(batch_gt[key], torch.Tensor):
                batch[key] = batch_gt[key].to(device)
                # print(f"Use gt {key} for pose estimation")
        # batch["pcd_sizes"] = batch["size"]
        # batch["pcd_trans"] = batch["translation"]

    # predict pose
    pipeline.system.cfg.max_objs = num_objs if num_objs != -1 else pipeline.system.cfg.max_objs
    pipeline.cfg.data.max_objs = num_objs if num_objs != -1 else pipeline.cfg.data.max_objs
    if pred_mode == "full":
        pred_pose, pred_trans, pred_size, pred_shape = pipeline.system.sample(batch)
    else:
        if pred_mode == "scene":
            pred_pose_latents, pred_scene_latents = pipeline.system.sample(batch)
            scene_pred_shape_latents = pred_scene_latents.reshape(pipeline.system.cfg.max_objs, -1, pred_scene_latents.shape[-1])
            scene_pred_shape = pipeline.system.shape_model.decode(scene_pred_shape_latents)
            scene_mesh_pred_v_f, has_surface = pipeline.system.shape_model.extract_geometry(latents=scene_pred_shape, extract_mesh_func=pipeline.system.cfg.extract_mesh_func)
        else:
            pred_pose_latents = pipeline.system.sample(batch)
        if use_direct_pose:
            pred_pose, pred_trans, pred_size = pred_pose_latents.split([6, 3, 1], dim=-1)  # [B, T, N, 6+3+1]
            pred_size = pred_size + 1  # normalize size to [0, 2]
        else:
            pred_pose, pred_trans, pred_size = pipeline.system.pose_model.decode(pred_pose_latents)
    
    pred_pose, pred_trans, pred_size = pred_pose.cpu(), pred_trans.cpu(), pred_size.cpu()

    if pcd_pose:
        pred_size = batch["bbox_sizes"].cpu()
        pred_trans = batch["translation"].cpu()
    
    # save
    max_objs = pipeline.cfg.data.max_objs if num_objs == -1 else num_objs
    mesh_list, transform_matrix_list, size_list = [], [], []
    scene_mesh_list = []
    for i, mesh_path in enumerate(mesh_paths):
        # mask
        if i >= max_objs:
            break
        if batch_gt is not None and batch_gt["mask_legal"][0, i] == 0:
            continue
        elif batch["mask_legal"][0, i] == 0:
            continue
        
        # load mesh
        mesh = trimesh.load(mesh_path)
    
        # scale
        size = pred_size[0, i].numpy()
        
        # normalize mesh at x-y plane - handle both Scene and Mesh objects
        if pcd_pose:
            if isinstance(mesh, trimesh.Scene):
                # For Scene objects, get vertices from all geometries
                all_vertices = []
                for geometry in mesh.geometry.values():
                    if hasattr(geometry, 'vertices'):
                        all_vertices.append(geometry.vertices)
                if all_vertices:
                    combined_vertices = np.vstack(all_vertices)
                    _, _, xy_max = normalize_points(combined_vertices)
                    size = size / xy_max.item()
            else:
                # For single mesh objects
                _, _, xy_max = normalize_points(mesh.vertices)
                size = size / xy_max.item()
        
        size = size / 2.0

        # transform
        transform_matrix = pipeline.system.recover_transform_matrix(pred_pose[0,i], pred_trans[0, i]).numpy()
        
        # apply only y-axis rotation
        if only_pitch:
            # transform_matrix[:3,:3] = get_rot_from_pitch(get_pitch_from_R(transform_matrix[:3,:3]))
            transform_matrix[:3,:3] = get_rot_from_yaw(get_yaw_from_R(transform_matrix[:3,:3]))
        
        # apply pose - handle both Scene and Mesh objects
        if isinstance(mesh, trimesh.Scene):
            # For Scene objects, apply transformations to each geometry
            for geometry in mesh.geometry.values():
                if hasattr(geometry, 'apply_scale') and hasattr(geometry, 'apply_transform'):
                    geometry.apply_scale(size)
                    geometry.apply_transform(transform_matrix)
        else:
            # For single mesh objects
            mesh.apply_scale(size)
            mesh.apply_transform(transform_matrix)
        
        # save
        transform_matrix_list.append(transform_matrix)
        mesh_list.append(mesh)        
        size_list.append(size)
        
        # scene mesh pred
        if pred_mode == "scene":
            mesh = scene_mesh_pred_v_f[i]
            scene_mesh_list.append(trimesh.Trimesh(vertices=mesh[0], faces=mesh[1]))
            
    return mesh_list, transform_matrix_list, size_list, scene_mesh_list


def estimate_pose(pipeline, mesh_path, output_path, batch_gt=None, pcd_pose=False, pred_mode="pose", use_gt_depth=False, intrinsics=None, \
                    depth_mode="moge", use_direct_pose=False, pcd_mode="pcd2", only_pitch=False, num_objs=-1):
    """Load a 3D mesh (OBJ) and match it to point clouds using the pretrained model."""
    # model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # pipeline = CraftsManPipeline.from_pretrained("ckpts/pose/michelangelo-autoencoder+n16384+lr0.0001+shift1.0+pose-LN-affine+pe+shape2vec-pcd+obj-attn", device=device, torch_dtype=torch.float32)
    
    # mesh
    if os.path.isfile(mesh_path):
        mesh_paths = [mesh_path]
    elif os.path.isdir(mesh_path):
        mesh_paths = [os.path.join(mesh_path, f) for f in sorted(os.listdir(mesh_path), key=numerical_sort_key) if f.endswith((".obj", ".glb"))]
    
    # depth
    depth_map = torch.load(os.path.join(output_path, "depth.pt")).numpy()
    if use_gt_depth:
        points = batch_gt["scene_pcds"].reshape(-1,3).cpu().numpy()
    elif depth_mode == "moge":
        intrinsics = np.loadtxt(os.path.join(output_path, "intrinsics.csv"), delimiter=",")
        depth_mask = np.array(Image.open(os.path.join(output_path, "depth_mask.png"))).astype(np.uint8)
        points = reconstruct_pcd(depth_map, intrinsic=intrinsics, scale=1, normalize=True, valid_mask=depth_mask, depth_mode=depth_mode)
        
    # viz depth
    # trimesh.points.PointCloud(points, colors=[255, 0, 0]).export(os.path.join(output_path, "depth_pts.ply"))

    # prepare data
    batch = prepare_data(pipeline=pipeline, mesh_paths=mesh_paths, depth_map=depth_map, output_path=output_path, \
        batch_gt=batch_gt, intrinsics=intrinsics, depth_mode=depth_mode, pcd_mode=pcd_mode, pred_mode=pred_mode, num_objs=num_objs)  
    
    # estimate pose
    all_mesh_list, all_transform_matrix_list, all_size_list, all_scene_mesh_list = match_mesh_to_pointcloud(pipeline=pipeline, batch=batch, batch_gt=batch_gt, mesh_paths=mesh_paths, \
        pcd_pose=pcd_pose, pred_mode=pred_mode, use_gt_depth=use_gt_depth, use_direct_pose=use_direct_pose, only_pitch=only_pitch, num_objs=num_objs)
    
    # save pose
    pose_path = os.path.join(output_path, "pose.json")
    pose_data = {
        "transform_matrix_list": all_transform_matrix_list,
        "size_list": all_size_list
    }
    pose_data = convert_to_python(pose_data)
    with open(pose_path, "w") as f:
        json.dump(pose_data, f, indent=4)
    print(f"Pose data saved to {pose_path}")
    
    if len(all_scene_mesh_list) > 0:
        # save scene mesh
        scene_mesh_path = os.path.join(output_path, "scene_mesh_pred.obj")
        trimesh.Scene(all_scene_mesh_list).export(scene_mesh_path)
        print(f"Scene mesh saved to {scene_mesh_path}")
    
    # Save final scene with textures preserved
    if len(all_mesh_list) > 0:
        # Check if any mesh has textures (GLB scene objects)
        has_textures = any(isinstance(mesh, trimesh.Scene) for mesh in all_mesh_list)
        
        if has_textures:
            # Save as GLB to preserve textures
            final_scene_path = os.path.join(output_path, "final_scene_with_textures.glb")
            final_scene = trimesh.Scene()
            for i, mesh in enumerate(all_mesh_list):
                if isinstance(mesh, trimesh.Scene):
                    # Add each geometry from the scene
                    for name, geometry in mesh.geometry.items():
                        final_scene.add_geometry(geometry, node_name=f"object_{i}_{name}")
                else:
                    # Add single mesh
                    final_scene.add_geometry(mesh, node_name=f"object_{i}")
            final_scene.export(final_scene_path)
            print(f"Final scene with textures saved to {final_scene_path}")
        else:
            # Save as OBJ for meshes without textures
            final_scene_path = os.path.join(output_path, "final_scene.obj")
            trimesh.Scene(all_mesh_list).export(final_scene_path)
            print(f"Final scene saved to {final_scene_path}")

    return all_mesh_list, all_transform_matrix_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./", help="Output directory")
    args = parser.parse_args()
    
    mesh_list, transform_matrix_list = estimate_pose(args.mesh_path, args.output_path)
    
    # The final scene export is now handled within estimate_pose function