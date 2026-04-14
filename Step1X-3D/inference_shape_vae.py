import warnings

warnings.filterwarnings("ignore")
import os
import trimesh
import fpsample
import numpy as np

from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
import torch

def geometry_vae_pipeline(input_shape_path, save_glb_path, n_samples=32768):
    """
    The base geometry VAE model, input is a shape npz file, output is a glb file.
    """
    pipeline = Step1X3DGeometryPipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", subfolder='Step1X-3D-Geometry-1300m'
    ).to("cuda")

    # Load the input shape and do random sampling or FPS sampling
    input_shape = np.load(input_shape_path)
    surface = input_shape['surface']  # 1000000 x 6 (xyz + normal)
    sharp_surface = input_shape['sharp_surface'] # 1000000 x 6
    kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(
        surface[:, :3], n_samples, h=5
    )
    surface = surface[kdline_fps_samples_idx]
    kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(
        sharp_surface[:, :3], n_samples, h=5
    )
    sharp_surface = sharp_surface[kdline_fps_samples_idx]
    surface = torch.tensor(surface, dtype=torch.float32, device='cuda')  # [n_samples, 6]
    sharp_surface = torch.tensor(sharp_surface, dtype=torch.float32, device='cuda')  # [n_samples, 6]

    # Encode the surface and decode to get the shape latents
    shape_latents, kl_embed, posterior = pipeline.vae.encode(
        surface.unsqueeze(0), # 1 x 32768 x 6
        sharp_surface=sharp_surface.unsqueeze(0), # 1 x 32768 x 6
        sample_posterior=True
    )
    shape_latents = pipeline.vae.decode(kl_embed)  # [B, num_latents, width]

    # Extract the mesh from the shape latents
    mesh = pipeline.vae.extract_geometry(
        shape_latents,
        mc_level=0.0,
        bounds=1.05,
        octree_resolution=512,
        enable_pbar=True,
    )

    os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
    mesh = trimesh.Trimesh(
        vertices=mesh[0].verts.cpu().numpy(),
        faces=mesh[0].faces.cpu().numpy(),
    )
    mesh.fix_normals()
    mesh.face_normals
    mesh.vertex_normals
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=(255, 255, 255),
            main_color=(255, 255, 255),
            metallicFactor=0.05,
            roughnessFactor=1.0,
        )
    )
    mesh.export(save_glb_path)

if __name__ == "__main__":
    image_path = "./data/shape_autoencoder/objaverse/surfaces/000-000/00a1a602456f4eb188b522d7ef19e81b.npz"
    geometry_vae_pipeline(image_path, "output/00a1a602456f4eb188b522d7ef19e81b.glb")