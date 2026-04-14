import os
import torch
import numpy as np
from PIL import Image
import trimesh
from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face  # type: ignore[import]


def step1x3d_generate(geometry_pipeline, texture_pipeline, image_path, output_path="./", seed_base=2025, num_inference_steps=25, guidance_scale=7.5):
    """
    Generate 3D models using Step1X-3D pipeline
    Args:
        geometry_pipeline: Step1X3DGeometryPipeline for geometry generation
        texture_pipeline: Step1X3DTexturePipeline for texture synthesis
        image_path: Path to input image(s) - can be single file or directory
        output_path: Output directory for generated 3D models
        seed_base: Base seed for reproducible generation
    """
    geometry_save_path = os.path.join(output_path, "geometry")
    textured_save_path = os.path.join(output_path, "3d_models")
    os.makedirs(geometry_save_path, exist_ok=True)
    os.makedirs(textured_save_path, exist_ok=True)
    
    device = geometry_pipeline.device
    
    # Generation
    if os.path.isfile(image_path):
        # Single image generation
        file_name = os.path.basename(image_path)
        base_name = os.path.splitext(file_name)[0]
        
        # Step 1: Generate geometry
        generator = torch.Generator(device=device)
        generator.manual_seed(seed_base)
        
        print(f"Generating geometry for {file_name}...")
        out = geometry_pipeline(
            image_path, 
            guidance_scale=guidance_scale, 
            num_inference_steps=num_inference_steps, 
            generator=generator
        )
        
        # Save geometry mesh
        geometry_glb_path = os.path.join(geometry_save_path, f"{base_name}.glb")
        out.mesh[0].export(geometry_glb_path)
        
        # Step 2: Add texture
        print(f"Adding texture for {file_name}...")
        mesh = trimesh.load(geometry_glb_path)
        mesh = remove_degenerate_face(mesh)
        mesh = reduce_face(mesh)
        textured_mesh = texture_pipeline(image_path, mesh, seed=seed_base)
        
        # Export final textured mesh
        output_glb_path = os.path.join(textured_save_path, f"{base_name}.glb")
        textured_mesh.export(output_glb_path)
        
        print(f"Saved textured model to {output_glb_path}")
        
    elif os.path.isdir(image_path):
        # Batch generation for directory
        image_files = sorted([f for f in os.listdir(image_path) 
                            if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        print(f"Found {len(image_files)} images to process")
        
        for idx, file_name in enumerate(image_files):
            full_image_path = os.path.join(image_path, file_name)
            base_name = os.path.splitext(file_name)[0]
            
            try:
                # Step 1: Generate geometry
                generator = torch.Generator(device=device)
                generator.manual_seed(seed_base + idx)  # Different seed for each image
                
                print(f"[{idx+1}/{len(image_files)}] Generating geometry for {file_name}...")
                out = geometry_pipeline(
                    full_image_path, 
                    guidance_scale=guidance_scale, 
                    num_inference_steps=num_inference_steps, 
                    generator=generator
                )
                
                # Save geometry mesh
                geometry_glb_path = os.path.join(geometry_save_path, f"{base_name}.glb")
                out.mesh[0].export(geometry_glb_path)
                
                # Step 2: Add texture
                print(f"[{idx+1}/{len(image_files)}] Adding texture for {file_name}...")
                mesh = trimesh.load(geometry_glb_path)
                mesh = remove_degenerate_face(mesh)
                mesh = reduce_face(mesh)
                textured_mesh = texture_pipeline(full_image_path, mesh, seed=seed_base + idx)
                
                # Export final textured mesh
                output_glb_path = os.path.join(textured_save_path, f"{base_name}.glb")
                textured_mesh.export(output_glb_path)
                
                print(f"[{idx+1}/{len(image_files)}] Saved textured model to {output_glb_path}")
                
                # Clear CUDA cache to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                continue
    else:
        raise ValueError(f"Invalid image path: {image_path}")
    
    print(f"Finished Step1X-3D generation")
    print(f"  - Geometry saved to {geometry_save_path}")
    print(f"  - Textured models saved to {textured_save_path}")
