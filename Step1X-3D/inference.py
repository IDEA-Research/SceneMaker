import warnings

warnings.filterwarnings("ignore")
import os
import trimesh
from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import (
    Step1X3DTexturePipeline,
)

from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
import torch

def geometry_pipeline(input_image_path, save_glb_path):
    """
    The base geometry model, input image generate glb
    """
    pipeline = Step1X3DGeometryPipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", subfolder='Step1X-3D-Geometry-1300m'
    ).to("cuda")

    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(2025)
    out = pipeline(input_image_path, guidance_scale=7.5, num_inference_steps=50, generator=generator)

    os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
    out.mesh[0].export(save_glb_path)


def geometry_label_pipeline(input_image_path, save_glb_path):
    """
    The label geometry model, support using label to control generation, input image generate glb
    """
    pipeline = Step1X3DGeometryPipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", subfolder='Step1X-3D-Geometry-Label-1300m'
    ).to("cuda")
    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(2025)

    out = pipeline(
        input_image_path,
        label={"symmetry": "x", "edge_type": "sharp"},
        guidance_scale=7.5,
        octree_resolution=384,
        max_facenum=400000,
        num_inference_steps=50,
        generator=generator
    )

    os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
    out.mesh[0].export(save_glb_path)


def texture_pipeline(input_image_path, input_glb_path, save_glb_path):
    """
    The texture model, input image and glb generate textured glb
    """
    mesh = trimesh.load(input_glb_path)
    pipeline = Step1X3DTexturePipeline.from_pretrained("stepfun-ai/Step1X-3D", subfolder="Step1X-3D-Texture")
    mesh = remove_degenerate_face(mesh)
    mesh = reduce_face(mesh)
    textured_mesh = pipeline(input_image_path, mesh, seed=2025)
    os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
    textured_mesh.export(save_glb_path)


if __name__ == "__main__":
    image_path = "/comp_robot/shiyukai/CraftsMan_scaleup/outputs/viz/obj_image/masked_image_2_tree.png"
    geometry_pipeline(image_path, "output/masked_image_2_tree.glb")
    geometry_label_pipeline(image_path, "output/masked_image_2_tree-label.glb")
    texture_pipeline(image_path, "output/masked_image_2_tree.glb", "output/masked_image_2_tree-textured.glb")
    texture_pipeline(
        image_path, "output/masked_image_2_tree-label.glb", "output/masked_image_2_tree-label-textured.glb"
    )
