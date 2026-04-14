import argparse
import os
import torch
import numpy as np
from PIL import Image
import trimesh
import pdb
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "Step1X-3D"))
from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import Step1X3DTexturePipeline
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
from moge.model.v1 import MoGeModel

from utils.grounding_sam import grounding_sam
# from utils.dinox_seg import dinox_seg
from utils.step1x3d_gen import step1x3d_generate
from utils.depth_estimator import moge
from utils.pose_matching import estimate_pose


# device
device = "cuda" if torch.cuda.is_available() else "cpu"
# moge
MoGe_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
# Load Step1X-3D pipelines
geometry_pipeline = Step1X3DGeometryPipeline.from_pretrained("stepfun-ai/Step1X-3D", subfolder='Step1X-3D-Geometry-1300m').to(device)
texture_pipeline = Step1X3DTexturePipeline.from_pretrained("stepfun-ai/Step1X-3D", subfolder="Step1X-3D-Texture")

# set random seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def align_meshes_to_same_y_floor(mesh_list):
    """Shift each mesh so all meshes share the same minimum y value."""
    if not mesh_list:
        return mesh_list

    y_mins = []
    valid_indices = []
    for idx, mesh in enumerate(mesh_list):
        if mesh is None or getattr(mesh, "vertices", None) is None or len(mesh.vertices) == 0:
            continue
        y_mins.append(float(np.min(mesh.vertices[:, 1])))
        valid_indices.append(idx)

    if not y_mins:
        return mesh_list

    # Keep the global floor height unchanged, align other meshes to it.
    target_y = min(y_mins)
    for idx, y_min in zip(valid_indices, y_mins):
        delta = target_y - y_min
        if abs(delta) > 1e-8:
            mesh_list[idx].apply_translation([0.0, delta, 0.0])

    return mesh_list


def scene_generation(args):    
    # output path
    output_path = f'{args.output_path}_{args.seg_mode}'
    os.makedirs(output_path, exist_ok=True)
    
    # resize image
    image = Image.open(args.image_path).convert("RGB")
    if min(image.size) < 512:
        scale = 512 / min(image.size)
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.BICUBIC)
    image.save(os.path.join(output_path, "scene_image.png"))
    
    # segmentation
    if args.seg_mode == "click":
        pass
    else:
        image_array, detections, seg_map_pil = grounding_sam(detector_id=args.detector_id, segmenter_id=args.segmenter_id, image_path=os.path.join(output_path, f"scene_image.png"), \
            labels=args.labels, output_path=output_path, threshold=args.threshold, device="cpu")
    print("Finished segmentation")

    # resize mask
    seg_mask_path = os.path.join(output_path, "masks")
    for mask_file in os.listdir(seg_mask_path):
        mask = Image.open(os.path.join(seg_mask_path, mask_file)).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
            mask.save(os.path.join(seg_mask_path, mask_file))
    
    # depth-anything
    depth = moge(model=MoGe_model, image_path=os.path.join(output_path, f"scene_image.png"), output_path=output_path)
    print("Finished depth-anything")
    
    # # 3d generation with Step1X-3D
    # deocclusion_path = os.path.join(output_path, "deocclusion_images")
    # image_path = deocclusion_path if "flux" in args.deocc_mode else os.path.join(output_path, "masked_images")
    # step1x3d_generate(geometry_pipeline=geometry_pipeline, texture_pipeline=texture_pipeline, 
    #                  image_path=image_path, output_path=output_path)
    # print("Finished 3D generation")
    
    # pose estimation
    from craftsman import CraftsManPipeline
    pose_pipeline = CraftsManPipeline.from_pretrained(args.pose_model, device=device, torch_dtype=torch.float32)
    mesh_list, transform_matrix_list = estimate_pose(pipeline=pose_pipeline, mesh_path=os.path.join(output_path, "3d_models"), output_path=output_path, pcd_mode=args.pcd_mode, \
                                        pcd_pose=args.use_pcd_pose, pred_mode=args.pred_mode, use_gt_depth=args.use_gt_depth, depth_mode=args.depth_mode, use_direct_pose=args.use_direct_pose, \
                                        only_pitch=args.only_pitch, num_objs=args.num_objs)
    print("Finished pose estimation")

    # Post-process meshes so every object touches the same y-floor height.
    # mesh_list = align_meshes_to_same_y_floor(mesh_list)
    
    # composite
    trimesh.Scene(mesh_list).export(os.path.join(output_path, "final_scene.obj"))
    print(f"Finished composition, results have been saved to {output_path}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--depth_encoder", type=str, default="vitl")
    parser.add_argument("--labels", type=str, nargs="+")
    parser.add_argument("--output_path", type=str, default="./", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--pcd_mode", type=str, default="pcd2")
    parser.add_argument("--seg_mode", type=str, default="click")
    parser.add_argument("--deocc_mode", type=str, default="flux", help="deocclusion mode: skip, gt, flux")
    parser.add_argument("--detector_id", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--segmenter_id", type=str, default="facebook/sam-vit-base")
    parser.add_argument("--depth_mode", type=str, default="moge")
    parser.add_argument("--pose_model", type=str, default="ckpts/SceneMaker_openset_ckpts")
    parser.add_argument("--use_icp", action='store_true', help="Use ICP for alignment")
    parser.add_argument("--use_gt_mesh", action='store_true', help="Use ground truth mesh")
    parser.add_argument("--use_pcd_pose", action='store_true', help="Use PCD pose")
    parser.add_argument("--pred_mode", type=str, default="pose", help="Use full mode")
    parser.add_argument("--use_gt_depth", action='store_true', help="Use ground truth depth")
    parser.add_argument("--use_direct_pose", action='store_true', help="Use ground truth depth")
    parser.add_argument("--only_pitch", action='store_true', help="Only apply pitch rotation")
    parser.add_argument("--num_objs", type=int, default=-1, help="Random seed")
    args = parser.parse_args()
    
    # # random seed
    # set_seed(args.seed)
    
    scene_generation(args)
    
    