import gradio as gr
import os
import sys
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from huggingface_hub import hf_hub_download
import cv2
import json
import random
import trimesh
from datetime import datetime
import torch

# Prefer direct HF downloads and disable AWS metadata probes in restricted networks.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")

current_dir = os.path.dirname(os.path.abspath(__file__))
step1x_dir = os.path.join(current_dir, "Step1X-3D")
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if step1x_dir not in sys.path:
    sys.path.insert(0, step1x_dir)

# Add path for utility functions
from diffusers import FluxKontextPipeline
from utils.flux_deocc import crop_square_with_padding
from utils.step1x3d_gen import step1x3d_generate
from utils.pose_matching import estimate_pose
from utils.depth_estimator import moge as run_moge_depth
from moge.model.v1 import MoGeModel
from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import Step1X3DTexturePipeline  # type: ignore[import]
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline  # type: ignore[import]

# Set cache directory under the current path
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = CACHE_DIR

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models at module level to avoid Gradio State initialization issues
print("Loading SAM predictor...")
sam_checkpoint = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_h_4b8939.pth")
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# Keep SAM on CPU for all operations
sam = sam.to('cpu')
print("SAM predictor loaded successfully on CPU (will run all inference on CPU)!")
SAM_PREDICTOR = SamPredictor(sam)

print("Loading FLUX deocclusion model...")
try:
    # Load model (reference: official FLUX.1-Kontext demo)
    DEOCC_PIPELINE = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev", 
        torch_dtype=torch.bfloat16
    ).to("cuda")
    print("✓ FLUX model loaded on CUDA")
except Exception as e:
    print(f"Error loading deocc pipeline: {e}")
    import traceback
    traceback.print_exc()
    DEOCC_PIPELINE = None

print("Loading Step1X-3D pipelines...")
try:
    STEP1X3D_GEOMETRY_PIPELINE = Step1X3DGeometryPipeline.from_pretrained(
        "stepfun-ai/Step1X-3D",
        subfolder='Step1X-3D-Geometry-1300m'
    ).to(device)
    STEP1X3D_TEXTURE_PIPELINE = Step1X3DTexturePipeline.from_pretrained(
        "stepfun-ai/Step1X-3D",
        subfolder="Step1X-3D-Texture"
    )
    print(f"Step1X-3D pipelines loaded on {device}")
except Exception as e:
    print(f"Error loading Step1X-3D pipelines: {e}")
    import traceback
    traceback.print_exc()
    STEP1X3D_GEOMETRY_PIPELINE = None
    STEP1X3D_TEXTURE_PIPELINE = None


def get_sam_predictor():
    """Return the pre-loaded SAM predictor"""
    return SAM_PREDICTOR


def get_deocc_pipeline():
    """Return the pre-loaded deocclusion pipeline"""
    return DEOCC_PIPELINE


def _deocc_prompt(caption):
    return f'complete the {caption} in the image. remove the white occlusion. smooth the edge. highly detailed geometry, realistic material with accurate reflections, global illumination, soft ambient occlusion, physically based rendering (PBR), rendered in a photorealistic 3D environment with balanced composition, 8k ultra-detailed quality'


def reset_image(predictor, img):
    if img is None:
        return predictor, None, "Please upload an image.", [], []
    
    # Convert to numpy array
    img = np.array(img)
    
    # Resize if image is too large to prevent memory issues
    # Lower resolution for FLUX deocclusion to avoid OOM
    max_dimension = 1024  # Reduced from 1920 to prevent FLUX OOM
    h, w = img.shape[:2]
    if max(h, w) > max_dimension:
        ratio = max_dimension / max(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Image resized from ({w}, {h}) to ({new_w}, {new_h}) to prevent OOM")
    
    predictor.set_image(img)
    original_img = img.copy()
    return predictor, original_img, "Image loaded. Click to add points for segmentation.", [], []


def run_sam(img, predictor, selected_points):
    if len(selected_points) == 0:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    
    # SAM runs on CPU only
    input_points = [p for p in selected_points]
    input_labels = [1 for _ in range(len(selected_points))]
    masks, _, _ = predictor.predict(
        point_coords=np.array(input_points),
        point_labels=np.array(input_labels),
        multimask_output=False,
    )
    best_mask = masks[0].astype(np.uint8)
    
    # Apply morphological processing for multiple points
    if len(selected_points) > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        best_mask = cv2.dilate(best_mask, kernel, iterations=1)
        best_mask = cv2.erode(best_mask, kernel, iterations=1)
    
    return best_mask


def draw_points_on_image(image, points):
    image_with_points = image.copy()
    for point in points:
        x, y = point
        color = (255, 0, 0)  # red point
        cv2.circle(image_with_points, (int(x), int(y)), radius=8, color=color, thickness=-1)
        cv2.circle(image_with_points, (int(x), int(y)), radius=10, color=(255, 255, 255), thickness=2)
    return image_with_points


def apply_mask_overlay(image, mask, color=(0, 255, 0), alpha=0.5):
    """Overlay the mask on the image."""
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = color
    
    # Blend semi-transparent mask
    result = cv2.addWeighted(image, 1-alpha, mask_colored, alpha, 0)
    
    # Draw contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    
    return result


def compress_image_for_gradio(image, max_size=800):
    """Compress and resize image to prevent response size issues"""
    if image is None:
        return None
    
    # Convert to PIL if numpy array
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Resize if too large
    if max(pil_image.size) > max_size:
        ratio = max_size / max(pil_image.size)
        new_size = tuple(int(dim * ratio) for dim in pil_image.size)
        pil_image = pil_image.resize(new_size, Image.LANCZOS)
    
    # Convert back to numpy array
    return np.array(pil_image)


def get_point(image, selected_points, evt: gr.SelectData):
    """Add a clicked point."""
    if image is None:
        return image, selected_points, "Please upload an image first."
    
    x, y = evt.index
    selected_points.append([x, y])
    
    # Show points on the image
    updated_image = draw_points_on_image(np.array(image), selected_points)
    
    return updated_image, selected_points, f"Added point at ({x}, {y}). Total points: {len(selected_points)}"


def clear_points(original_image, selected_points):
    """Clear all selected points."""
    selected_points.clear()
    if original_image is not None:
        return original_image.copy(), selected_points, "All points cleared."
    return None, selected_points, "All points cleared."


def generate_mask(original_image, selected_points, predictor):
    """Generate a mask."""
    if original_image is None:
        return None, "Please upload an image first."
    
    if len(selected_points) == 0:
        return original_image, "Please add at least one point."
    
    # Generate mask
    mask = run_sam(original_image, predictor, selected_points)
    
    # Overlay mask on image
    result_image = apply_mask_overlay(original_image, mask)
    
    return result_image, f"Mask generated with {len(selected_points)} points."


def add_mask_caption_pair(mask_image, caption, data_pairs, original_image, selected_points, predictor):
    """Add a mask-caption pair to the dataset."""
    if original_image is None:
        return data_pairs, "", "Please upload an image first."
    
    if len(selected_points) == 0:
        return data_pairs, caption, "Please add at least one point to generate mask."
    
    if not caption.strip():
        return data_pairs, caption, "Please enter a caption."
    
    # Generate mask
    mask = run_sam(original_image, predictor, selected_points)
    
    # Create data pair
    data_pair = {
        "id": len(data_pairs) + 1,
        "caption": caption.strip(),
        "points": selected_points.copy(),
        "mask": mask.copy(),
        "deocc_image": None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    data_pairs.append(data_pair)
    
    # Clear current selection
    selected_points.clear()
    
    return data_pairs, "", f"Added pair #{len(data_pairs)}: '{caption}' (deocclusion pending). Points cleared for next mask."


def flux_infer_direct(pipe, input_image, prompt, guidance_scale=3.5, steps=28, seed=42):
    """
    Direct FLUX inference following official app.py logic
    Reference: FLUX.1-Kontext-Dev/app.py infer function
    """
    # Convert to RGB (as in official app)
    input_image_rgb = input_image.convert("RGB")
    
    # Call pipeline with official parameters
    result = pipe(
        image=input_image_rgb, 
        prompt=prompt,
        guidance_scale=guidance_scale,
        width=input_image_rgb.size[0],
        height=input_image_rgb.size[1],
        num_inference_steps=steps,
        generator=torch.Generator().manual_seed(seed),
    ).images[0]
    
    return result


def run_deocclusion(original_image, caption, selected_points, predictor):
    """Run deocclusion on current mask and caption"""
    if original_image is None:
        return None, None, "Please upload an image first."
    
    if len(selected_points) == 0:
        return None, None, "Please add at least one point to generate mask."
    
    if not caption.strip():
        return None, None, "Please enter a caption."
    
    if DEOCC_PIPELINE is None:
        return None, None, "Deocclusion model not loaded. Please wait for model loading."
    
    # Use inference_mode to completely disable gradient tracking
    with torch.inference_mode():
        try:
            # Generate mask
            mask = run_sam(original_image, predictor, selected_points)
            
            # Create masked image (white background where mask is 0)
            masked_image = original_image.copy()
            masked_image[mask == 0] = [255, 255, 255]
            
            # Convert to PIL Image
            masked_pil = Image.fromarray(masked_image)
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            
            # Apply crop and padding (following flux_deocc.py logic)
            bbox_ratio = 0.7
            cropped_img, cropped_mask = crop_square_with_padding(
                masked_pil, 
                mask_pil, 
                bbox_ratio=bbox_ratio, 
                pad_color=(255, 255, 255, 255)
            )
            
            # Convert to RGB and apply mask
            cropped_img_rgb = cropped_img.convert("RGB")
            from PIL import ImageOps
            cropped_mask_inv = ImageOps.invert(cropped_mask.convert("L"))
            cropped_img_rgb.paste((255, 255, 255), mask=cropped_mask_inv)
            
            # Generate deocclusion prompt
            prompt = f'complete the {caption.strip()} in the image. remove the white occlusion. smooth the edge. highly detailed geometry, realistic material with accurate reflections, global illumination, soft ambient occlusion, physically based rendering (PBR), rendered in a photorealistic 3D environment with balanced composition, 8k ultra-detailed quality'
            
            # Aggressive GPU memory cleanup before FLUX inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                # Print memory stats for debugging
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU Memory before FLUX: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            # Run FLUX inference directly (following official app.py)
            result_pil = flux_infer_direct(
                pipe=DEOCC_PIPELINE,
                input_image=cropped_img_rgb,
                prompt=prompt,
                guidance_scale=3.5,
                steps=28,
                seed=42
            )
            
            # Clear GPU memory immediately after FLUX
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            
            # Convert result to numpy
            deocc_image = np.array(result_pil)
            
            # Create preview with mask overlay
            preview_image = apply_mask_overlay(original_image, mask, color=(0, 255, 0), alpha=0.3)
            
            # Compress both images for Gradio
            deocc_compressed = compress_image_for_gradio(deocc_image, max_size=800)
            preview_compressed = compress_image_for_gradio(preview_image, max_size=800)
            
            return deocc_compressed, preview_compressed, f"Deocclusion completed for '{caption.strip()}'. Review the result and click 'Confirm & Add' if satisfied."
                
        except Exception as e:
            import traceback
            error_msg = f"Error during deocclusion: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return None, None, f"Error during deocclusion: {str(e)}"


def run_batch_deocclusion(data_pairs, original_image):
    """Run deocclusion for all existing mask+caption pairs in batch."""
    if original_image is None:
        return data_pairs, None, None, "Please upload an image first."

    if not data_pairs:
        return data_pairs, None, None, "No data pairs found. Please add mask+caption pairs first."

    pending_jobs = []
    updated_pairs = []
    skipped = 0
    failed = 0
    completed = 0

    for pair in data_pairs:
        pair_copy = pair.copy()
        mask = pair_copy.get("mask")
        caption = str(pair_copy.get("caption", "")).strip()

        if mask is None or not caption:
            failed += 1
            updated_pairs.append(pair_copy)
            continue

        if pair_copy.get("deocc_image") is not None:
            skipped += 1
            updated_pairs.append(pair_copy)
            continue

        pending_jobs.append({
            "id": pair_copy["id"],
            "caption": caption,
            "mask": mask,
            "seed": 42 + pair_copy["id"],
        })
        updated_pairs.append(pair_copy)

    if not pending_jobs:
        status = f"Batch deocclusion finished. completed={completed}, skipped={skipped}, failed={failed}."
        return updated_pairs, None, None, status

    if DEOCC_PIPELINE is None:
        return data_pairs, None, None, "Deocclusion model not loaded."

    results_by_id = {}
    with torch.inference_mode():
        for job in pending_jobs:
            try:
                mask = job["mask"]
                caption = job["caption"]
                masked_image = original_image.copy()
                masked_image[mask == 0] = [255, 255, 255]

                masked_pil = Image.fromarray(masked_image)
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                cropped_img, cropped_mask = crop_square_with_padding(
                    masked_pil,
                    mask_pil,
                    bbox_ratio=0.7,
                    pad_color=(255, 255, 255, 255),
                )

                from PIL import ImageOps

                cropped_img_rgb = cropped_img.convert("RGB")
                cropped_mask_inv = ImageOps.invert(cropped_mask.convert("L"))
                cropped_img_rgb.paste((255, 255, 255), mask=cropped_mask_inv)

                result_pil = flux_infer_direct(
                    pipe=DEOCC_PIPELINE,
                    input_image=cropped_img_rgb,
                    prompt=_deocc_prompt(caption),
                    guidance_scale=3.5,
                    steps=28,
                    seed=job["seed"],
                )
                results_by_id[job["id"]] = {
                    "id": job["id"],
                    "deocc_image": np.array(result_pil),
                    "error": None,
                }
            except Exception as e:
                results_by_id[job["id"]] = {"id": job["id"], "deocc_image": None, "error": str(e)}

    last_deocc_preview = None
    last_preview_mask = None
    for pair in updated_pairs:
        result = results_by_id.get(pair["id"])
        if result is None:
            continue

        if result.get("error"):
            failed += 1
            print(f"Batch deocclusion failed on pair {pair['id']}: {result['error']}")
            continue

        deocc_image = result.get("deocc_image")
        if deocc_image is None:
            failed += 1
            continue

        pair["deocc_image"] = deocc_image
        completed += 1
        last_deocc_preview = compress_image_for_gradio(deocc_image, max_size=800)
        last_preview_mask = compress_image_for_gradio(
            apply_mask_overlay(original_image, pair["mask"], color=(0, 255, 0), alpha=0.3),
            max_size=800,
        )

    if last_deocc_preview is None:
        for pair in updated_pairs:
            existing = pair.get("deocc_image")
            if existing is not None:
                last_deocc_preview = compress_image_for_gradio(existing, max_size=800)
                last_preview_mask = compress_image_for_gradio(
                    apply_mask_overlay(original_image, pair["mask"], color=(0, 255, 0), alpha=0.3),
                    max_size=800,
                )
                break

    status = f"Batch deocclusion finished. completed={completed}, skipped={skipped}, failed={failed}."
    return updated_pairs, last_deocc_preview, last_preview_mask, status


def _pair_display_label(pair):
    return f"{pair['id']}: {pair['caption']}"


def build_deocc_gallery_and_choices(data_pairs):
    gallery_items = []
    choices = []
    for pair in data_pairs:
        deocc_image = pair.get("deocc_image")
        if deocc_image is None:
            continue
        label = _pair_display_label(pair)
        gallery_items.append((deocc_image, label))
        choices.append(label)
    return gallery_items, gr.update(choices=choices, value=[])


def run_batch_deocclusion_ui(data_pairs, original_image, save_path):
    """UI wrapper for batch deocclusion without preview widgets."""
    updated_pairs, last_deocc_preview, _last_preview_mask, status = run_batch_deocclusion(data_pairs, original_image)
    save_status = _save_masks_and_masked_images(updated_pairs, original_image, save_path)
    return updated_pairs, last_deocc_preview, f"{status} {save_status}"


def rerun_selected_deocclusion(data_pairs, selected_labels, original_image):
    """Re-run deocclusion for selected pairs and refresh visualization."""
    if original_image is None:
        gallery, choices_update = build_deocc_gallery_and_choices(data_pairs)
        return data_pairs, None, None, None, gallery, choices_update, "Please upload an image first."

    if not data_pairs:
        return data_pairs, None, None, None, [], gr.update(choices=[], value=[]), "No pairs found."

    if not selected_labels:
        gallery, choices_update = build_deocc_gallery_and_choices(data_pairs)
        return data_pairs, None, None, None, gallery, choices_update, "Please select one or more deocclusion results to re-run."

    selected_ids = set()
    for label in selected_labels:
        head = str(label).split(":", 1)[0].strip()
        try:
            selected_ids.add(int(head))
        except ValueError:
            continue

    staged_pairs = []
    for pair in data_pairs:
        pair_copy = pair.copy()
        if pair_copy.get("id") in selected_ids:
            pair_copy["deocc_image"] = None
        staged_pairs.append(pair_copy)

    updated_pairs, last_deocc_preview, last_preview_mask, status = run_batch_deocclusion(staged_pairs, original_image)
    gallery, choices_update = build_deocc_gallery_and_choices(updated_pairs)

    return (
        updated_pairs,
        last_deocc_preview,
        last_preview_mask,
        last_deocc_preview,
        gallery,
        choices_update,
        f"Re-ran deocclusion for {len(selected_ids)} selected pair(s). {status}",
    )


def rerun_selected_deocclusion_ui(data_pairs, selected_labels, original_image):
    """UI wrapper for selective rerun without preview widgets."""
    updated_pairs, current_deocc, _mask_preview, _img_preview, gallery, choices_update, status = rerun_selected_deocclusion(
        data_pairs, selected_labels, original_image
    )
    return updated_pairs, current_deocc, gallery, choices_update, status


def confirm_and_add_pair(deocc_image, caption, data_pairs, original_image, selected_points, predictor):
    """Confirm deocclusion result and add to data pairs"""
    if deocc_image is None:
        return data_pairs, caption, None, None, "No deocclusion result to confirm. Please run deocclusion first."
    
    if original_image is None:
        return data_pairs, caption, None, None, "Please upload an image first."
    
    if len(selected_points) == 0:
        return data_pairs, caption, None, None, "Please add at least one point to generate mask."
    
    if not caption.strip():
        return data_pairs, caption, None, None, "Please enter a caption."
    
    # Generate mask
    mask = run_sam(original_image, predictor, selected_points)
    
    # Create data pair with deocclusion result
    data_pair = {
        "id": len(data_pairs) + 1,
        "caption": caption.strip(),
        "points": selected_points.copy(),
        "mask": mask.copy(),
        "deocc_image": deocc_image.copy(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    data_pairs.append(data_pair)
    
    # Clear current selection
    selected_points.clear()
    
    return data_pairs, "", None, None, f"Added pair #{len(data_pairs)}: '{caption}' with deocclusion result. Points cleared for next mask."


def save_masks_and_images(data_pairs, original_image, save_path):
    """Save masks and images to the specified path."""
    if not data_pairs:
        return "No data pairs to save."
    
    if original_image is None:
        return "No original image to save."
    
    if not save_path.strip():
        return "Please enter a valid save path."
    
    try:
        # Create save directory structure
        save_dir = save_path.strip()
        masks_dir = os.path.join(save_dir, "masks")
        masked_images_dir = os.path.join(save_dir, "masked_images")
        deocc_images_dir = os.path.join(save_dir, "deocclusion_images")
        
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(masked_images_dir, exist_ok=True)
        os.makedirs(deocc_images_dir, exist_ok=True)
        
        # Save original image to root directory
        original_pil = Image.fromarray(original_image)
        image_filename = "scene_image.png"
        original_pil.save(os.path.join(save_dir, image_filename))
        
        # Save mask and image for each data pair
        saved_files = []
        
        for i, pair in enumerate(data_pairs):
            pair_id = pair["id"]
            caption = pair["caption"]
            mask = pair["mask"]
            deocc_image = pair.get("deocc_image", None)
            
            # Sanitize caption for filename
            safe_caption = "".join(c for c in caption if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_caption = safe_caption.replace(' ', '_')[:50]  # limit length
            
            # Save mask to masks directory
            mask_filename = f"mask_{i}_{safe_caption}.png"
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            mask_pil.save(os.path.join(masks_dir, mask_filename))
            
            # Create masked image and save to masked_images directory
            masked_image = original_image.copy()
            masked_image[mask == 0] = [255, 255, 255]  # set background to white
            masked_img_filename = f"masked_image_{i}_{safe_caption}.png"
            masked_pil = Image.fromarray(masked_image)
            masked_pil.save(os.path.join(masked_images_dir, masked_img_filename))
            
            # Save deocclusion image (if present)
            if deocc_image is not None:
                deocc_filename = f"masked_image_{i}_{safe_caption}.png"
                deocc_pil = Image.fromarray(deocc_image)
                deocc_pil.save(os.path.join(deocc_images_dir, deocc_filename))
                saved_files.append(deocc_filename)
            
            saved_files.extend([mask_filename, masked_img_filename])
        
        # Save captions to detections.txt
        detections_filename = "detections.txt"
        with open(os.path.join(save_dir, detections_filename), 'w', encoding='utf-8') as f:
            for i, pair in enumerate(data_pairs):
                f.write(f"{pair['caption']}\n")
        
        # Save data metadata file
        data_info = {
            "image_file": image_filename,
            "detections_file": detections_filename,
            "total_pairs": len(data_pairs),
            "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "directory_structure": {
                "masks": "masks/",
                "masked_images": "masked_images/",
                "deocclusion_images": "deocclusion_images/"
            },
            "data_pairs": [
                {
                    "id": pair["id"],
                    "caption": pair["caption"],
                    "mask_file": f"masks/mask_{i}_{(''.join(c for c in pair['caption'] if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')[:50])}.png",
                    "masked_image_file": f"masked_images/masked_image_{i}_{(''.join(c for c in pair['caption'] if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')[:50])}.png",
                    "deocc_image_file": f"deocclusion_images/masked_image_{i}_{(''.join(c for c in pair['caption'] if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')[:50])}.png" if pair.get("deocc_image") is not None else None,
                    "points": pair["points"],
                    "timestamp": pair["timestamp"]
                }
                for i, pair in enumerate(data_pairs)
            ]
        }
        
        info_filename = "data_info.json"
        with open(os.path.join(save_dir, info_filename), 'w', encoding='utf-8') as f:
            json.dump(data_info, f, indent=2, ensure_ascii=False)
        
        # Generate segmentation images
        # Create global segmentation maps
        seg_map = np.zeros(original_image.shape[:2], dtype=np.uint8)
        seg_map_objs = np.zeros(original_image.shape[:2], dtype=np.uint8)
        for i, pair in enumerate(data_pairs):
            mask = pair["mask"]
            seg_map[mask > 0] = i + 1
            seg_map_objs[mask > 0] = i + 1
        
        # Generate colored segmentation maps
        seg_map_pil = generate_colored_segmentation(seg_map)
        seg_map_pil.save(os.path.join(save_dir, "segmentation.png"))
        
        seg_map_pil_objs = generate_colored_segmentation(seg_map_objs)
        seg_map_pil_objs.save(os.path.join(save_dir, "segmentation_max_objs.png"))
        
        deocc_count = sum(1 for pair in data_pairs if pair.get("deocc_image") is not None)

        # After saving files, automatically run 3D generation (deocclusion images only)
        step1x3d_status = "Skipped Step1X-3D generation (no deocclusion images)."
        if deocc_count > 0:
            try:
                deocclusion_path = os.path.join(save_dir, "deocclusion_images")
                if STEP1X3D_GEOMETRY_PIPELINE is None or STEP1X3D_TEXTURE_PIPELINE is None:
                    step1x3d_status = "Step1X-3D pipelines not loaded, skipped 3D generation."
                else:
                    step1x3d_generate(
                        geometry_pipeline=STEP1X3D_GEOMETRY_PIPELINE,
                        texture_pipeline=STEP1X3D_TEXTURE_PIPELINE,
                        image_path=deocclusion_path,
                        output_path=save_dir,
                    )
                    step1x3d_status = "Finished 3D generation in save flow (models in 3d_models/)."
            except Exception as gen_e:
                step1x3d_status = f"Step1X-3D generation failed: {str(gen_e)}"

        return f"Successfully saved to '{save_dir}':\n📁 Directory structure:\n  ├── {image_filename} (original image)\n  ├── {detections_filename} (captions list)\n  ├── segmentation.png (colored segmentation)\n  ├── segmentation_max_objs.png (colored segmentation max objs)\n  ├── masks/ ({len(data_pairs)} mask files)\n  ├── masked_images/ ({len(data_pairs)} masked image files)\n  ├── deocclusion_images/ ({deocc_count} deocclusion files)\n  └── {info_filename}\n\n✅ Total files: {len(saved_files) + 5}\n🧊 3D Generation: {step1x3d_status}"
        
    except Exception as e:
        return f"Error saving files: {str(e)}"


def _safe_caption(caption):
    safe = "".join(c for c in caption if c.isalnum() or c in (' ', '-', '_')).rstrip()
    return safe.replace(' ', '_')[:50]


def _save_masks_and_masked_images(data_pairs, original_image, save_path):
    """Persist masks and masked images to output path after deocclusion."""
    if original_image is None:
        return "Skipped saving masks/masked images (missing original image)."

    base_dir = save_path.strip() if save_path and save_path.strip() else "./output_masks"
    masks_dir = os.path.join(base_dir, "masks")
    masked_images_dir = os.path.join(base_dir, "masked_images")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masked_images_dir, exist_ok=True)

    saved_count = 0
    for i, pair in enumerate(data_pairs):
        mask = pair.get("mask")
        caption = str(pair.get("caption", "object"))
        if mask is None:
            continue

        safe_caption = _safe_caption(caption)
        pair_id = int(pair.get("id", i + 1))
        mask_filename = f"pair_{pair_id}_{safe_caption}.png"
        masked_filename = f"masked_image_{i}_{safe_caption}.png"

        mask_uint8 = (np.array(mask) > 0).astype(np.uint8) * 255
        Image.fromarray(mask_uint8, mode='L').save(os.path.join(masks_dir, mask_filename))

        masked_image = original_image.copy()
        masked_image[np.array(mask) == 0] = [255, 255, 255]
        Image.fromarray(masked_image).save(os.path.join(masked_images_dir, masked_filename))
        saved_count += 1

    return f"Saved {saved_count} masks and masked images to {base_dir}."


def _build_combined_scene(model_paths, output_path):
    """Build a combined preview scene so all generated objects can be viewed together."""
    scene = trimesh.Scene()
    for idx, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            continue
        mesh_or_scene = trimesh.load(model_path, force='scene')
        if isinstance(mesh_or_scene, trimesh.Scene):
            for geom in mesh_or_scene.geometry.values():
                g = geom.copy()
                g.apply_translation([idx * 1.2, 0.0, 0.0])
                scene.add_geometry(g)
        else:
            g = mesh_or_scene.copy()
            g.apply_translation([idx * 1.2, 0.0, 0.0])
            scene.add_geometry(g)
    scene.export(output_path)
    return output_path


def _mesh_item_min_y(mesh_item):
    """Get minimum y of a mesh or scene item."""
    if mesh_item is None:
        return None

    if isinstance(mesh_item, trimesh.Scene):
        y_candidates = []
        for geom in mesh_item.geometry.values():
            vertices = getattr(geom, "vertices", None)
            if vertices is None or len(vertices) == 0:
                continue
            y_candidates.append(float(np.min(vertices[:, 1])))
        return min(y_candidates) if y_candidates else None

    vertices = getattr(mesh_item, "vertices", None)
    if vertices is None or len(vertices) == 0:
        return None
    return float(np.min(vertices[:, 1]))


def _translate_mesh_item_y(mesh_item, delta_y):
    """Translate a mesh or scene item along y axis."""
    if abs(delta_y) <= 1e-8:
        return

    if isinstance(mesh_item, trimesh.Scene):
        for geom in mesh_item.geometry.values():
            geom.apply_translation([0.0, delta_y, 0.0])
    else:
        mesh_item.apply_translation([0.0, delta_y, 0.0])


def align_meshes_to_same_y_floor(mesh_list):
    """Shift each mesh/scene so all objects share the same minimum y floor."""
    if not mesh_list:
        return mesh_list

    valid = []
    for idx, mesh_item in enumerate(mesh_list):
        min_y = _mesh_item_min_y(mesh_item)
        if min_y is None:
            continue
        valid.append((idx, min_y))

    if not valid:
        return mesh_list

    target_y = min(min_y for _, min_y in valid)
    for idx, min_y in valid:
        _translate_mesh_item_y(mesh_list[idx], target_y - min_y)

    return mesh_list


POSE_PIPELINE = None
POSE_PIPELINE_PATH = None
MOGE_MODEL = None


def get_pose_pipeline(pose_model_path):
    """Lazily load and cache pose pipeline."""
    global POSE_PIPELINE, POSE_PIPELINE_PATH

    requested_path = pose_model_path.strip()
    local_path = requested_path
    if not os.path.isabs(local_path):
        local_path = os.path.join(current_dir, local_path)

    if not os.path.isdir(local_path):
        raise FileNotFoundError(
            f"Pose model directory not found: {local_path}. "
            f"Please provide a valid local path containing model.ckpt/model.bin and config.yaml."
        )

    if POSE_PIPELINE is not None and POSE_PIPELINE_PATH == local_path:
        return POSE_PIPELINE

    from craftsman import CraftsManPipeline

    POSE_PIPELINE = CraftsManPipeline.from_pretrained(
        local_path,
        device=device,
        torch_dtype=torch.float32,
    )
    POSE_PIPELINE_PATH = local_path
    return POSE_PIPELINE


def resolve_pose_model_path(scene_type):
    """Resolve pose checkpoint path from UI scene type selection."""
    mapping = {
        "Indoor Scenes": "ckpts/SceneMaker_indoor_ckpts",
        "Open-set Scenes": "ckpts/SceneMaker_openset_ckpts",
    }
    return mapping.get(scene_type, "ckpts/SceneMaker_openset_ckpts")


def get_moge_model():
    """Lazily load and cache MoGe depth model."""
    global MOGE_MODEL
    if MOGE_MODEL is None:
        MOGE_MODEL = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
    return MOGE_MODEL


def _prepare_pose_assets(base_dir, original_image, data_pairs):
    """Prepare scene image, masks and depth assets required by pose matching."""
    os.makedirs(base_dir, exist_ok=True)
    scene_image_path = os.path.join(base_dir, "scene_image.png")

    # Keep pose flow compatible with scene_generation: scene image must exist.
    if original_image is not None:
        Image.fromarray(original_image).save(scene_image_path)
    elif not os.path.exists(scene_image_path):
        return False, "Pose estimation requires scene image. Please upload an image first."

    masks_dir = os.path.join(base_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    if data_pairs:
        for pair in data_pairs:
            mask = pair.get("mask")
            if mask is None:
                continue
            caption = _safe_caption(str(pair.get("caption", "object")))
            pair_id = int(pair.get("id", 0))
            mask_name = f"pair_{pair_id}_{caption}.png"
            mask_uint8 = (np.array(mask) > 0).astype(np.uint8) * 255
            Image.fromarray(mask_uint8, mode='L').save(os.path.join(masks_dir, mask_name))

    depth_path = os.path.join(base_dir, "depth.pt")
    intrinsics_path = os.path.join(base_dir, "intrinsics.csv")
    depth_mask_path = os.path.join(base_dir, "depth_mask.png")
    if not (os.path.exists(depth_path) and os.path.exists(intrinsics_path) and os.path.exists(depth_mask_path)):
        model = get_moge_model()
        run_moge_depth(model=model, image_path=scene_image_path, output_path=base_dir)

    return True, "Pose assets are ready."


def run_pose_estimation_for_scene(save_path, pose_scene_type, original_image, data_pairs):
    """Estimate object poses and visualize the composed final scene."""
    try:
        base_dir = save_path.strip() if save_path and save_path.strip() else "./output_masks"
        mesh_dir = os.path.join(base_dir, "3d_models")

        if not os.path.isdir(mesh_dir):
            return None, "No 3D models found. Please run Submit Batch 3D first."

        model_files = [f for f in os.listdir(mesh_dir) if f.endswith(".glb")]
        if not model_files:
            return None, "No .glb files found in 3d_models."

        ok, prep_status = _prepare_pose_assets(base_dir, original_image, data_pairs)
        if not ok:
            return None, prep_status

        output_dir = base_dir

        pose_model_path = resolve_pose_model_path(pose_scene_type)
        pose_pipeline = get_pose_pipeline(pose_model_path)
        mesh_list, _ = estimate_pose(
            pipeline=pose_pipeline,
            mesh_path=mesh_dir,
            output_path=output_dir,
            pcd_mode="pcd2",
            pcd_pose=False,
            pred_mode="pose",
            use_gt_depth=False,
            depth_mode="moge",
            use_direct_pose=True,
            only_pitch=False,
            num_objs=-1,
        )

        # Align all objects to a consistent y-floor like scene_generation.py.
        # mesh_list = align_meshes_to_same_y_floor(mesh_list)

        # Export aligned scene. Prefer GLB when scene items are present to keep textures.
        if any(isinstance(mesh, trimesh.Scene) for mesh in mesh_list):
            final_scene_path = os.path.join(output_dir, "final_scene_with_textures.glb")
        else:
            final_scene_path = os.path.join(output_dir, "final_scene.obj")
        trimesh.Scene(mesh_list).export(final_scene_path)
        return final_scene_path, f"{prep_status} Pose estimation finished. Final scene saved to {final_scene_path}."
    except Exception as e:
        return None, f"Pose estimation failed: {str(e)}"


def submit_batch_3d_generation(data_pairs, save_path):
    """After all deocclusion is done, submit for batch Step1X-3D generation."""
    if not data_pairs:
        return None, gr.update(choices=[], value=[]), [], "No data pairs found. Please add pairs first."

    try:
        base_dir = save_path.strip() if save_path and save_path.strip() else "./output_masks"
        input_dir = os.path.join(base_dir, "deocclusion_images")
        os.makedirs(input_dir, exist_ok=True)

        batch_items = []
        for pair in data_pairs:
            deocc_image = pair.get("deocc_image")
            if deocc_image is None:
                continue
            safe_caption = _safe_caption(pair["caption"])
            image_name = f"pair_{pair['id']}_{safe_caption}.png"
            image_path = os.path.join(input_dir, image_name)
            Image.fromarray(deocc_image).save(image_path)
            batch_items.append({
                "id": pair["id"],
                "caption": pair["caption"],
                "image_name": image_name,
                "image_path": image_path,
                "model_name": f"pair_{pair['id']}_{safe_caption}.glb"
            })

        if not batch_items:
            return None, gr.update(choices=[], value=[]), [], "No deocclusion images available for 3D generation."

        if STEP1X3D_GEOMETRY_PIPELINE is None or STEP1X3D_TEXTURE_PIPELINE is None:
            return None, gr.update(choices=[], value=[]), [], "Step1X-3D pipelines are not loaded."

        step1x3d_generate(
            geometry_pipeline=STEP1X3D_GEOMETRY_PIPELINE,
            texture_pipeline=STEP1X3D_TEXTURE_PIPELINE,
            image_path=input_dir,
            output_path=base_dir,
            seed_base=2025,
        )

        model_dir = os.path.join(base_dir, "3d_models")
        select_choices = []
        model_paths = []
        for item in batch_items:
            model_path = os.path.join(model_dir, item["model_name"])
            item["model_path"] = model_path
            label = f"{item['id']}: {item['caption']}"
            item["label"] = label
            select_choices.append(label)
            if os.path.exists(model_path):
                model_paths.append(model_path)

        if not model_paths:
            return None, gr.update(choices=select_choices, value=[]), batch_items, "Batch generation finished, but no model files were found."

        scene_path = os.path.join(base_dir, "combined_scene.glb")
        _build_combined_scene(model_paths, scene_path)

        return scene_path, gr.update(choices=select_choices, value=[]), batch_items, f"Batch 3D generation completed for {len(model_paths)} objects."
    except Exception as e:
        return None, gr.update(choices=[], value=[]), [], f"Batch 3D generation failed: {str(e)}"


def regenerate_selected_3d(selected_labels, batch_items):
    """Regenerate selected objects and rebuild combined 3D scene."""
    if not batch_items:
        return None, [], "No batch generation result found. Click Submit Batch 3D first."

    if not selected_labels:
        return None, batch_items, "Please select one or more objects to regenerate."

    try:
        selected_set = set(selected_labels)
        selected_items = [item for item in batch_items if item.get("label") in selected_set]
        regenerated_count = len(selected_items)
        if regenerated_count == 0:
            return None, batch_items, "No valid selected objects found."

        base_dir = os.path.dirname(os.path.dirname(selected_items[0]["image_path"]))
        if STEP1X3D_GEOMETRY_PIPELINE is None or STEP1X3D_TEXTURE_PIPELINE is None:
            return None, batch_items, "Step1X-3D pipelines are not loaded."

        for item in selected_items:
            step1x3d_generate(
                geometry_pipeline=STEP1X3D_GEOMETRY_PIPELINE,
                texture_pipeline=STEP1X3D_TEXTURE_PIPELINE,
                image_path=item["image_path"],
                output_path=base_dir,
                seed_base=random.randint(1, 1_000_000_000),
            )

        model_paths = [item["model_path"] for item in batch_items if os.path.exists(item.get("model_path", ""))]
        if not model_paths:
            return None, batch_items, "Regeneration finished, but no model files were found."

        scene_path = os.path.join(base_dir, "combined_scene.glb")
        _build_combined_scene(model_paths, scene_path)

        return scene_path, batch_items, f"Regenerated {regenerated_count} selected object(s)."
    except Exception as e:
        return None, batch_items, f"Regeneration failed: {str(e)}"

def generate_colored_segmentation(label_image):
    """Generate a colored segmentation image."""
    # Create color palette
    palette = [
        0,0,0, 255,0,0, 0,255,0, 0,0,255, 255,255,0, 255,0,255, 0,255,255,
        128,0,0, 0,128,0, 0,0,128, 128,128,0, 128,0,128, 0,128,128,
        64,0,0, 0,64,0, 0,0,64, 64,64,0, 64,0,64, 0,64,64,
        192,192,192, 128,128,128, 255,165,0, 75,0,130, 238,130,238
    ]
    palette.extend([0] * (768 - len(palette)))
    
    label_image_pil = Image.fromarray(label_image.astype(np.uint8), mode="P")
    label_image_pil.putpalette(palette)
    return label_image_pil


def delete_data_pair(data_pairs, pair_id):
    """Delete the specified data pair."""
    if not data_pairs:
        return data_pairs, "No data pairs to delete."
    
    try:
        pair_id = int(pair_id)
        if 1 <= pair_id <= len(data_pairs):
            deleted_pair = data_pairs.pop(pair_id - 1)
            # Re-index IDs
            for i, pair in enumerate(data_pairs):
                pair["id"] = i + 1
            return data_pairs, f"Deleted pair: '{deleted_pair['caption']}'"
        else:
            return data_pairs, f"Invalid pair ID. Valid range: 1-{len(data_pairs)}"
    except ValueError:
        return data_pairs, "Please enter a valid number."


# Gradio interface
with gr.Blocks(
    title="SceneMaker: Open-set 3D Scene Generation with Decoupled De-occlusion and Pose Estimation Model",
    theme=gr.themes.Default(),
    css=None,
    analytics_enabled=False,
) as demo:
    gr.Markdown("""
    <div style="
        border-radius: 16px;
        padding: 18px 20px;
        background: linear-gradient(135deg, #f8fbff 0%, #f4fff8 100%);
        border: 1px solid #d9e9ff;
        margin-bottom: 8px;
    ">
        <h1 style="margin: 0 0 8px 0; font-size: 28px;">SceneMaker Workflow</h1>
        <p style="margin: 0; color: #475569; font-size: 15px;">
            Open-set 3D scene generation with decoupled de-occlusion and pose estimation.
        </p>
    </div>

    <div style="display: grid; grid-template-columns: repeat(5, minmax(170px, 1fr)); gap: 10px; margin: 8px 0 12px 0;">
        <div style="border: 1px solid #e2e8f0; border-radius: 12px; padding: 10px 12px; background: #ffffff;">
            <div style="font-size: 18px; font-weight: 600;">1) 🎯 Masking</div>
            <div style="font-size: 13px; color: #475569; margin-top: 4px;">Upload image, click points, generate object masks.</div>
        </div>
        <div style="border: 1px solid #e2e8f0; border-radius: 12px; padding: 10px 12px; background: #ffffff;">
            <div style="font-size: 18px; font-weight: 600;">2) 🏷️ Pairing</div>
            <div style="font-size: 13px; color: #475569; margin-top: 4px;">Add multiple mask-caption pairs for one scene.</div>
        </div>
        <div style="border: 1px solid #e2e8f0; border-radius: 12px; padding: 10px 12px; background: #ffffff;">
            <div style="font-size: 18px; font-weight: 600;">3) 🪄 Deocclusion</div>
            <div style="font-size: 13px; color: #475569; margin-top: 4px;">Run batch deocclusion, review gallery, re-run unsatisfied results.</div>
        </div>
        <div style="border: 1px solid #e2e8f0; border-radius: 12px; padding: 10px 12px; background: #ffffff;">
            <div style="font-size: 18px; font-weight: 600;">4) 🧊 3D Gen</div>
            <div style="font-size: 13px; color: #475569; margin-top: 4px;">Submit batch 3D generation and selectively regenerate objects.</div>
        </div>
        <div style="border: 1px solid #e2e8f0; border-radius: 12px; padding: 10px 12px; background: #ffffff;">
            <div style="font-size: 18px; font-weight: 600;">5) 🧭 Pose</div>
            <div style="font-size: 13px; color: #475569; margin-top: 4px;">Run pose estimation to compose and visualize the final scene.</div>
        </div>
    </div>

    <div style="font-size: 13px; color: #64748b; margin-top: 2px;">
        💡 Tip: complete all pairs first, then run batch deocclusion for better consistency.
    </div>
    """)
    
    # State variables - use pre-loaded models
    predictor = gr.State(value=SAM_PREDICTOR)
    original_image = gr.State(value=None)
    selected_points = gr.State(value=[])
    data_pairs = gr.State(value=[])
    current_deocc_image = gr.State(value=None)
    batch_3d_items = gr.State(value=[])
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 1. Upload Image and Select Segmentation Area")
            
            # Image upload and display
            input_image = gr.Image(
                type='pil', 
                label='Upload Image', 
                interactive=True,
                height=400
            )
            
            # Image processing area
            with gr.Row():
                image_display = gr.Image(
                    type='numpy',
                    label='Click Image to Add Segmentation Points',
                    interactive=True,
                    height=400
                )
            
            # Control buttons
            with gr.Row():
                clear_btn = gr.Button("Clear Points", variant="secondary")
                generate_mask_btn = gr.Button("Generate Mask", variant="primary")
            
            # Status info
            status_text = gr.Textbox(
                label="Status",
                value="Please upload an image to start",
                interactive=False
            )

            gr.Markdown("### 5. Pose Estimation")
            with gr.Row():
                pose_scene_type_input = gr.Radio(
                    label="Pose Scene Type",
                    choices=["Indoor Scenes", "Open-set Scenes"],
                    value="Open-set Scenes"
                )
                run_pose_btn = gr.Button("Run Pose Estimation", variant="primary")

            final_scene_viewer = gr.Model3D(
                label="Final Composed Scene",
                clear_color=[1.0, 1.0, 1.0, 1.0],
                height=320
            )

            pose_status = gr.Textbox(
                label="Pose Estimation Status",
                value="",
                interactive=False,
                lines=2
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 2. Add Caption and Mask Pairs")
            
            # Caption input
            caption_input = gr.Textbox(
                label="Caption",
                placeholder="Enter description for current mask...",
                lines=3
            )
            
            # Add pair button
            with gr.Row():
                confirm_add_btn = gr.Button("Add Pair", variant="primary")

            # Data management
            gr.Markdown("### Data Management")

            data_info = gr.Textbox(
                label="Data Pairs Info",
                value="No data yet",
                interactive=False,
                lines=5
            )

            # Delete function
            with gr.Row():
                delete_id = gr.Number(
                    label="Delete ID",
                    value=1,
                    precision=0
                )
                delete_btn = gr.Button("Delete", variant="secondary")
            
            gr.Markdown("#### 3. Deocclusion Results")
            deocc_gallery = gr.Gallery(
                label="Deocclusion Gallery",
                columns=2,
                height=260,
                object_fit="contain"
            )
            deocc_rerun_selection = gr.CheckboxGroup(
                label="Select Deocclusion Results To Re-run",
                choices=[],
                value=[]
            )
            with gr.Row():
                run_deocc_btn = gr.Button("Run Batch Deocclusion", variant="primary")
                rerun_deocc_btn = gr.Button("Re-run Selected Deocclusion", variant="secondary")

            save_path_input = gr.Textbox(
                label="Save Path",
                placeholder="Enter save directory path, e.g.: ./output_masks",
                value="./output_masks",
                lines=1
            )

            gr.Markdown("#### 3D Batch Generation")
            with gr.Row():
                submit_batch_3d_btn = gr.Button("Submit Batch 3D", variant="primary")
                regenerate_selected_btn = gr.Button("Regenerate Selected", variant="secondary")

            regenerate_selection = gr.CheckboxGroup(
                label="Select Objects To Regenerate",
                choices=[],
                value=[]
            )

            model_3d_viewer = gr.Model3D(
                label="Combined 3D Viewer (All Objects)",
                clear_color=[1.0, 1.0, 1.0, 1.0],
                height=280
            )
    
    # Event binding
    
    # Image upload
    input_image.upload(
        reset_image,
        inputs=[predictor, input_image],
        outputs=[predictor, original_image, status_text, selected_points, data_pairs]
    ).then(
        lambda img: img,
        inputs=[original_image],
        outputs=[image_display]
    )
    
    # Click image to add points
    image_display.select(
        get_point,
        inputs=[original_image, selected_points],
        outputs=[image_display, selected_points, status_text]
    )
    
    # Clear points
    clear_btn.click(
        clear_points,
        inputs=[original_image, selected_points],
        outputs=[image_display, selected_points, status_text]
    )
    
    # Generate mask
    generate_mask_btn.click(
        generate_mask,
        inputs=[original_image, selected_points, predictor],
        outputs=[image_display, status_text]
    )
    
    # Run batch deocclusion for all existing pairs
    run_deocc_btn.click(
        run_batch_deocclusion_ui,
        inputs=[data_pairs, original_image, save_path_input],
        outputs=[data_pairs, current_deocc_image, status_text]
    ).then(
        lambda pairs: f"Current data pairs: {len(pairs)}\n" + "\n".join([f"{i+1}. {pair['caption']}" + (" (with deocc)" if pair.get('deocc_image') is not None else "") for i, pair in enumerate(pairs)]) if pairs else "No data yet",
        inputs=[data_pairs],
        outputs=[data_info]
    ).then(
        build_deocc_gallery_and_choices,
        inputs=[data_pairs],
        outputs=[deocc_gallery, deocc_rerun_selection]
    )
    
    # Add mask-caption pair only (deocclusion deferred to batch step)
    confirm_add_btn.click(
        add_mask_caption_pair,
        inputs=[image_display, caption_input, data_pairs, original_image, selected_points, predictor],
        outputs=[data_pairs, caption_input, status_text]
    ).then(
        lambda img: img if img is not None else None,
        inputs=[original_image],
        outputs=[image_display]
    ).then(
        lambda pairs: f"Current data pairs: {len(pairs)}\n" + "\n".join([f"{i+1}. {pair['caption']}" + (" (with deocc)" if pair.get('deocc_image') is not None else "") for i, pair in enumerate(pairs)]) if pairs else "No data yet",
        inputs=[data_pairs],
        outputs=[data_info]
    ).then(
        build_deocc_gallery_and_choices,
        inputs=[data_pairs],
        outputs=[deocc_gallery, deocc_rerun_selection]
    )
    
    # Delete data pair
    delete_btn.click(
        delete_data_pair,
        inputs=[data_pairs, delete_id],
        outputs=[data_pairs, status_text]
    ).then(
        lambda pairs: f"Current data pairs: {len(pairs)}\n" + "\n".join([f"{i+1}. {pair['caption']}" + (" (with deocc)" if pair.get('deocc_image') is not None else "") for i, pair in enumerate(pairs)]) if pairs else "No data yet",
        inputs=[data_pairs],
        outputs=[data_info]
    ).then(
        build_deocc_gallery_and_choices,
        inputs=[data_pairs],
        outputs=[deocc_gallery, deocc_rerun_selection]
    )

    # Re-run selected deocclusion results
    rerun_deocc_btn.click(
        rerun_selected_deocclusion_ui,
        inputs=[data_pairs, deocc_rerun_selection, original_image],
        outputs=[
            data_pairs,
            current_deocc_image,
            deocc_gallery,
            deocc_rerun_selection,
            status_text,
        ]
    ).then(
        lambda pairs: f"Current data pairs: {len(pairs)}\n" + "\n".join([f"{i+1}. {pair['caption']}" + (" (with deocc)" if pair.get('deocc_image') is not None else "") for i, pair in enumerate(pairs)]) if pairs else "No data yet",
        inputs=[data_pairs],
        outputs=[data_info]
    )
    
    # Submit all deocclusion results for batch 3D generation
    submit_batch_3d_btn.click(
        submit_batch_3d_generation,
        inputs=[data_pairs, save_path_input],
        outputs=[model_3d_viewer, regenerate_selection, batch_3d_items, status_text]
    )

    # Regenerate selected objects
    regenerate_selected_btn.click(
        regenerate_selected_3d,
        inputs=[regenerate_selection, batch_3d_items],
        outputs=[model_3d_viewer, batch_3d_items, status_text]
    )

    # Pose estimation and final scene visualization
    run_pose_btn.click(
        run_pose_estimation_for_scene,
        inputs=[
            save_path_input,
            pose_scene_type_input,
            original_image,
            data_pairs,
        ],
        outputs=[final_scene_viewer, pose_status]
    )


if __name__ == "__main__":
    demo.queue(max_size=10)  # Add queue to handle long-running operations
    demo.launch(
        share=False, 
        server_port=7860,
        show_error=True,
        quiet=False,
        server_name="192.168.80.51",
        max_threads=10  # Increase thread pool size for better handling
    )