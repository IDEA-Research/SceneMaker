import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import trimesh
from utils.transforms import *

import pdb

def compute_metrics(vertices_pred, vertices_gt, use_icp_obj: bool = True, use_icp_scene: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from craftsman.utils.metrics import (
        compute_chamfer_distance,
        compute_fscore,
        compute_volume_iou,
        icp,
    )

    def normalize_(tensor):
        min_vals = tensor.min(dim=1, keepdim=True)[0]
        max_vals = tensor.max(dim=1, keepdim=True)[0]

        ranges = max_vals - min_vals
        ranges = torch.where(ranges == 0, torch.ones_like(ranges), ranges)

        # respectively nrom
        normalized_tensor = 1.9 * (tensor - min_vals) / ranges.max() - 0.95
        
        # same scale norm
        # mean = (max_vals + min_vals) / 2
        # normalized_tensor = 1.9 * (tensor - mean) / ranges.max() - 0.95

        return normalized_tensor

    cd_scene, cd_scene_1, cd_scene_2, fscore_scene = [], [], [], []
    cd_object, fscore_object, iou_bbox = [], [], []
    volume_iou = []

    # preprocess: fps sampling
    vertices_pred = [torch.from_numpy(trimesh.sample.sample_surface(mesh, 20480)[0]) for mesh in vertices_pred]
    vertices_pred = torch.stack(vertices_pred).float().to(device)
    vertices_gt = [torch.from_numpy(trimesh.sample.sample_surface(mesh, 20480)[0]) for mesh in vertices_gt]
    vertices_gt = torch.stack(vertices_gt).float().to(device)

    # 1. scene
    B, N, C = vertices_pred.shape
    vertices_scene_pred = rearrange(vertices_pred, "B N C -> (B N) C").unsqueeze(0)
    vertices_scene_gt = rearrange(vertices_gt, "B N C -> (B N) C").unsqueeze(0)
    
    # normalize
    vertices_scene_pred = normalize_(vertices_scene_pred)
    vertices_scene_gt = normalize_(vertices_scene_gt)
    
    # # save scene vertices
    # vertices_scene_pred_np = vertices_scene_pred.squeeze(0).cpu().numpy()
    # vertices_scene_gt_np = vertices_scene_gt.squeeze(0).cpu().numpy()
    # trimesh.points.PointCloud(vertices_scene_pred_np).export("outputs/viz/eval/scene_pred2.ply")
    # trimesh.points.PointCloud(vertices_scene_gt_np).export("outputs/viz/eval/scene_gt2.ply")
    
    # object
    vertices_object_pred = vertices_scene_pred.reshape(B,N,C).clone().detach()
    vertices_object_gt = vertices_scene_gt.reshape(B,N,C).clone().detach()
    
    # # save object vertices
    # for idx, (v_pred, v_gt) in enumerate(zip(vertices_object_pred, vertices_object_gt)):
    #     vertices_object_pred_np = v_pred[idx].reshape(-1,3).cpu().numpy()
    #     vertices_object_gt_np = v_gt[idx].reshape(-1,3).cpu().numpy()
    #     trimesh.points.PointCloud(vertices_object_pred_np).export(f"outputs/viz/eval/object_pred_{idx}.ply")
    #     trimesh.points.PointCloud(vertices_object_gt_np).export(f"outputs/viz/eval/object_gt_{idx}.ply")


    ## 1.1 icp
    if use_icp_scene:
        vertices_scene_pred, R, t = icp(
            vertices_scene_pred, vertices_scene_gt, max_iterations=50
        )
    
    # viz
    # vertices_scene_pred_np = vertices_scene_pred.squeeze(0).cpu().numpy()
    # trimesh.points.PointCloud(vertices_scene_pred_np).export("outputs/viz/eval/scene_pred_icp.ply")

    ## 1.2 metrics
    cds = compute_chamfer_distance(vertices_scene_pred, vertices_scene_gt)
    cd_scene.append(cds[0])
    cd_scene_1.append(cds[1])
    cd_scene_2.append(cds[2])
    fscore_scene.append(compute_fscore(vertices_scene_pred, vertices_scene_gt))

    # # 2. object
    # vertices_object_pred = vertices_pred.reshape(B,N,C)
    # vertices_object_gt = vertices_gt.reshape(B,N,C)

    # 2.1. object iou in global scene
    iou_bbox.append(compute_volume_iou(vertices_scene_pred, vertices_scene_gt, mode="bbox"))

    # 2.2 object quality
    vertices_object_pred = normalize_(vertices_object_pred)
    vertices_object_gt = normalize_(vertices_object_gt)

    ## 2.2.1 icp
    if use_icp_obj:
        vertices_object_pred, _, _ = icp(
            vertices_object_pred, vertices_object_gt, max_iterations=50
        )

    ## 2.2.2 metrics
    # volume iou
    volume_iou.append(compute_volume_iou(vertices_object_pred, vertices_object_gt, mode="bbox"))
    cd_object.append(
        compute_chamfer_distance(vertices_object_pred, vertices_object_gt)[0]
    )
    fscore_object.append(
        compute_fscore(vertices_object_pred, vertices_object_gt)
    )

    for item in [
        cd_scene,
        cd_scene_1,
        cd_scene_2,
        fscore_scene,
        cd_object,
        fscore_object,
        iou_bbox,
        volume_iou
    ]:
        if len(item) == 0:
            return None

    mean_acc = lambda x: torch.cat(x).mean().cpu()

    return {
        "scene_cd": mean_acc(cd_scene),
        "scene_cd_1": mean_acc(cd_scene_1),
        "scene_cd_2": mean_acc(cd_scene_2),
        "scene_fscore": mean_acc(fscore_scene),
        "object_cd": mean_acc(cd_object),
        "object_fscore": mean_acc(fscore_object),
        "iou_bbox": mean_acc(iou_bbox),
        "volume_iou": mean_acc(volume_iou)
    }
