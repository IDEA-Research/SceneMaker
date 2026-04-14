import argparse
import os
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import cv2
import trimesh


def moge(model, image_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # image
    # Read the input image and convert to tensor (3, H, W) and normalize to [0, 1]
    input_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)                       
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

    # Infer 
    output = model.infer(input_image)
    pred_points, pred_depth, pred_mask, pred_intrinsics = output["points"], output["depth"], output["mask"], output["intrinsics"]
    
    # save depth
    os.makedirs(output_path, exist_ok=True)
    torch.save(pred_depth.detach().cpu(), os.path.join(output_path, "depth.pt"))
    
    # save intrinsics as csv
    pred_intrinsics = pred_intrinsics.cpu().numpy()
    pred_intrinsics[:1] = pred_intrinsics[:1] * pred_depth.shape[1]
    pred_intrinsics[1:2] = pred_intrinsics[1:2] * pred_depth.shape[0]
    np.savetxt(os.path.join(output_path, "intrinsics.csv"), pred_intrinsics, delimiter=",")
    
    # save mask
    pred_mask = pred_mask.squeeze(0).cpu().numpy()
    pred_mask = (pred_mask * 255).astype(np.uint8)
    pred_mask = Image.fromarray(pred_mask)
    pred_mask.save(os.path.join(output_path, "depth_mask.png"))
    
    # viz points
    max_num_points = 100000
    moge_pcd = pred_points.cpu().numpy().reshape(-1,3)
    moge_pcd = moge_pcd[np.isfinite(moge_pcd).all(axis=1)] # remove inf points
    if moge_pcd.shape[0] > max_num_points:
        random_ind = np.random.choice(moge_pcd.shape[0], max_num_points, replace=False)
        moge_pcd = moge_pcd[random_ind]
    trimesh.points.PointCloud(moge_pcd, colors=[255, 0, 0]).export(os.path.join(output_path, f"MoGE_pcd.ply"))
    
    # viz depth
    # depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
    depth = pred_depth.detach().cpu().numpy() * 255
    depth = Image.fromarray(depth.astype("uint8"))
    depth.save(os.path.join(output_path, "depth.png"))
    
    return pred_depth, pred_intrinsics
    
    