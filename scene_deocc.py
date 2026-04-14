import argparse
import os
import torch
import numpy as np
from PIL import Image
import trimesh
import pdb
import sys
sys.path.append("/comp_robot/shiyukai/CraftsMan_scaleup")
from diffusers import FluxKontextPipeline, FluxKontextInpaintPipeline
from diffusers.utils import load_image
from utils.flux_deocc import flux_deocc_crop_and_inpaint
from utils.grounding_sam import grounding_sam
from utils.dinox_seg import dinox_seg    


# set random seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# models
device = "cuda" if torch.cuda.is_available() else "cpu"

def scene_generation(args, deocc_pipe=None):
    # output path
    output_path = f'{args.output_path}_{args.seg_mode}'
    os.makedirs(output_path, exist_ok=True)
    
    # resize image
    image = Image.open(args.image_path).convert("RGB")
    if min(image.size) < 512:
        scale = 512 / min(image.size)
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.BICUBIC)
    scene_image_path = os.path.join(output_path, "scene_image.png")
    image.save(scene_image_path)

    # segmentation
    if args.seg_mode == "dinox":
        dinox_seg(prompt=args.labels, image_path=os.path.join(output_path, f"scene_image.png"), task_name='dinox_detection', output_path=output_path)
    elif args.seg_mode == "click":
        pass
    else:
        image_array, detections, seg_map_pil = grounding_sam(detector_id=args.detector_id, segmenter_id=args.segmenter_id, image_path=os.path.join(output_path, f"scene_image.png"), \
            labels=args.labels, output_path=output_path, threshold=args.threshold, device="cpu")
    print("Finished segmentation")
    
    # save image and masked images
    masked_path = os.path.join(output_path, "masked_images")
    os.makedirs(masked_path, exist_ok=True)
    mask_path = os.path.join(output_path, "masks")
    os.makedirs(mask_path, exist_ok=True)
    deocclusion_path = os.path.join(output_path, "deocclusion_images")
    os.makedirs(deocclusion_path, exist_ok=True)
    
    # deocclusion
    if os.path.exists(os.path.join(output_path, "detections.txt")):
        with open(os.path.join(output_path, "detections.txt"), "r") as f:
            lines = f.readlines()
            labels = [line.strip() for line in lines]
            
    for idx in range(len(labels)):
        # if labels[idx] != "chair":
        #     continue
        mask_save_path = os.path.join(mask_path, f"mask_{idx}_{labels[idx]}.png")
        prompt = f'complete the {labels[idx]} in the image. remove the white occlusion. smooth the edge. highly detailed geometry, realistic material with accurate reflections, global illumination, soft ambient occlusion, physically based rendering (PBR), rendered in a photorealistic 3D environment with balanced composition, 8k ultra-detailed quality'
        bbox_ratio = 0.7
        flux_deocc_crop_and_inpaint(pipe=deocc_pipe, image_path=scene_image_path, mask_path=mask_save_path, prompt=prompt, deocc_mode=args.deocc_mode, \
            save_path=os.path.join(deocclusion_path, f"masked_image_{idx}_{labels[idx]}.png"), bbox_ratio=bbox_ratio, pad_color=(255,255,255,255), guidance_scale=3.5)
    
    print("Finished deocclusion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--depth_encoder", type=str, default="vitl")
    parser.add_argument("--labels", type=str, nargs="+")
    parser.add_argument("--output_path", type=str, default="./", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--pcd_mode", type=str, default="pcd2")
    parser.add_argument("--seg_mode", type=str, default="grounded-sam")
    parser.add_argument("--detector_id", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--segmenter_id", type=str, default="facebook/sam-vit-huge")
    parser.add_argument("--depth_mode", type=str, default="depth-anything")
    parser.add_argument("--pose_model", type=str, default="ckpts/pose/rectify_obj_pcd2_caption")
    parser.add_argument("--use_icp", action='store_true', help="Use ICP for alignment")
    parser.add_argument("--use_gt_mesh", action='store_true', help="Use ground truth mesh")
    parser.add_argument("--use_pcd_pose", action='store_true', help="Use PCD pose")
    parser.add_argument("--deocc_mode", type=str, default="flux", help="deocclusion mode: skip, gt, flux, brushnet, sdxl")
    parser.add_argument("--pred_mode", type=str, default="pose", help="Use full mode")
    parser.add_argument("--use_gt_depth", action='store_true', help="Use ground truth depth")
    parser.add_argument("--use_direct_pose", action='store_true', help="Use ground truth depth")
    parser.add_argument("--lora_weights_path", type=str, default="/comp_robot/shiyukai/CraftsMan_scaleup/FLUX.1-Kontext-dev-Training/outputs/Deocc_Lora_Training_Kontext_dev_4batch_v3/checkpoint-10000/pytorch_lora_weights.safetensors")
    args = parser.parse_args()
    
    # # random seed
    # set_seed(args.seed)
    
    # deocc pipeline
    if args.deocc_mode == "flux":
        deocc_pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to(device)
    elif args.deocc_mode == "flux_deocc":
        deocc_pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
        deocc_pipe.load_lora_weights(args.lora_weights_path)
        deocc_pipe = deocc_pipe.to(device)
    else:
        deocc_pipe = FluxKontextInpaintPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to(device)
    
    scene_generation(args, deocc_pipe=deocc_pipe)
    
    