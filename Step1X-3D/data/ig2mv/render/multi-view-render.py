# scripts modified fron https://github.com/huanngzh/bpy-renderer

import os
import json
import argparse
from render_utils import (
    add_camera,
    init_render_engine,
    set_background_color,
    set_env_map,
    load_file,
    SceneManager,
    enable_color_output,
    enable_depth_output,
    enable_normals_output,
    convert_normal_to_webp,
    get_camera_positions_on_sphere
)

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-view rendering script for Blender.")
    parser.add_argument("--save_dir", type=str, default="save_outputs", help="Output directory for renders")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the 3D model file")
    parser.add_argument("--env_path", type=str, help="Path to the environment map (EXR/HDR) file", default="./env.exr")
    return parser.parse_args()

def main():
    args = parse_args()
    MODEL_PATH = args.model_path
    uid = MODEL_PATH.split('/')[-1].split('.')[0]
    OUTPUT_DIR = os.path.join(args.save_dir, uid)
    ENV_PATH = args.env_path

    # Step 1: Initialize
    init_render_engine("BLENDER_EEVEE")
    scene_mgr = SceneManager()
    scene_mgr.clear(reset_keyframes=True)

    # Step 2: Import model
    load_file(MODEL_PATH)
    scene_mgr.smooth()
    scene_mgr.clear_normal_map()
    scene_mgr.set_material_transparency(False)
    scene_mgr.set_materials_opaque()
    scene_mgr.normalize_scene(1.0)

    # Step 3: Set environment
    set_env_map(ENV_PATH)
    # set_bg_color([1.0, 1.0, 1.0, 1.0])

    # Step 4: Camera setup
    cam_positions, cam_matrices, elevs, azims = get_camera_positions_on_sphere(
        center=(0, 0, 0),
        radius=1.8,
        elevations=[0, 0, 0, 0, 89.99, -89.99],
        azimuths=[x - 90 for x in [0, 90, 180, 270, 180, 0]]
    )
    cameras = []
    for idx, mat in enumerate(cam_matrices):
        cam = add_camera(mat, "ORTHO", add_frame=idx < len(cam_matrices) - 1)
        cameras.append(cam)

        # Step 5: Output settings
    width, height = 1024, 1024
    enable_color_output(width, height, OUTPUT_DIR, file_format="WEBP", output_mode="IMAGE", film_transparent=True)
    enable_normals_output(OUTPUT_DIR)
    enable_depth_output(OUTPUT_DIR)
    scene_mgr.render()

    # Step 6: Convert normal maps
    for fname in os.listdir(OUTPUT_DIR):
        if fname.startswith("normal_") and fname.endswith(".exr"):
            src = os.path.join(OUTPUT_DIR, fname)
            render_src = src.replace("normal_", "color_").replace(".exr", ".webp")
            convert_normal_to_webp(
                src,
                src.replace(".exr", ".webp"),
                render_src
            )
            os.remove(src)

    # Optional. save metadata
    metadata = {"width": width, "height": height, "locations": []}
    for i, cam in enumerate(cameras):
        metadata["locations"].append({
            "index": f"{i:04d}",
            "projection_type": cam.data.type,
            "ortho_scale": cam.data.ortho_scale,
            "camera_angle_x": cam.data.angle_x,
            "elevation": elevs[i],
            "azimuth": azims[i],
            "transform_matrix": cam_matrices[i].tolist()
        })
        with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
            json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    main()