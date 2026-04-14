import argparse

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, UNet2DConditionModel
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation

from step1x3d_texture.models.attention_processor import (
    DecoupledMVRowColSelfAttnProcessor2_0,
)
from step1x3d_texture.pipelines.ig2mv_sdxl_pipeline import IG2MVSDXLPipeline
from step1x3d_texture.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from step1x3d_texture.utils import (
    get_orthogonal_camera,
    make_image_grid,
    tensor_to_image,
)
from step1x3d_texture.utils.render import NVDiffRastContextWrapper, load_mesh, render
from step1x3d_texture.differentiable_renderer.mesh_render import MeshRender
import trimesh
import xatlas
import scipy.sparse
from scipy.sparse.linalg import spsolve
from step1x3d_geometry.models.pipelines.pipeline_utils import smart_load_model


class Step1X3DTextureConfig:
    def __init__(self):
        # prepare pipeline params
        self.base_model = "stabilityai/stable-diffusion-xl-base-1.0"
        self.vae_model = "madebyollin/sdxl-vae-fp16-fix"
        self.unet_model = None
        self.lora_model = None
        self.adapter_path = "stepfun-ai/Step1X-3D"
        self.scheduler = "ddpm"
        self.num_views = 6
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.lora_scale = None

        # run pipeline params
        self.text = "high quality"
        self.num_inference_steps = 50
        self.guidance_scale = 3.0
        self.seed = -1
        self.reference_conditioning_scale = 1.0
        self.negative_prompt = "watermark, ugly, deformed, noisy, blurry, low contrast"
        self.azimuth_deg = [0, 45, 90, 180, 270, 315]

        # texture baker params
        self.selected_camera_azims = [0, 90, 180, 270, 180, 180]
        self.selected_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.selected_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
        self.camera_distance = 1.8
        self.render_size = 2048
        self.texture_size = 2048
        self.bake_exp = 4
        self.merge_method = "fast"

        # texture sync params
        self.texture_sync_config = {
            "texture_size": 1536,
            "latent_size": 768//8,
            "elevations": [0, 0, 0, 0, 90, -90],
            "azimuths": [0, 90, 180, 270, 0, 0],
            "texture_sync_ratio": 0.5,
            "exp_end": 6.0,
            "exp_start": 0,
            "shuffle_background_change": 0.4,
            "shuffle_background_end": 0.99,
            "camera_distance": 1.8
        }


class Step1X3DTexturePipeline:
    def __init__(self, config):
        self.config = config
        self.mesh_render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            camera_distance=self.config.camera_distance,
        )

        self.ig2mv_pipe = self.prepare_ig2mv_pipeline(
            base_model=self.config.base_model,
            vae_model=self.config.vae_model,
            unet_model=self.config.unet_model,
            lora_model=self.config.lora_model,
            adapter_path=self.config.adapter_path,
            scheduler=self.config.scheduler,
            num_views=self.config.num_views,
            device=self.config.device,
            dtype=self.config.dtype,
        )

    @classmethod
    def from_pretrained(cls, model_path, subfolder):
        config = Step1X3DTextureConfig()
        local_model_path = smart_load_model(model_path, subfolder=subfolder)
        print(f'Local model path: {local_model_path}')
        config.adapter_path = local_model_path
        return cls(config)

    def mesh_uv_wrap(self, mesh):
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
        vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
        mesh.vertices = mesh.vertices[vmapping]
        mesh.faces = indices
        mesh.visual.uv = uvs

        return mesh

    def prepare_ig2mv_pipeline(
        self,
        base_model,
        vae_model,
        unet_model,
        lora_model,
        adapter_path,
        scheduler,
        num_views,
        device,
        dtype,
    ):
        # Load vae and unet if provided
        pipe_kwargs = {}
        if vae_model is not None:
            pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model)
        if unet_model is not None:
            pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(unet_model)

        # Prepare pipeline
        pipe = IG2MVSDXLPipeline.from_pretrained(base_model, **pipe_kwargs)

        # Load scheduler if provided
        scheduler_class = None
        if scheduler == "ddpm":
            scheduler_class = DDPMScheduler
        elif scheduler == "lcm":
            scheduler_class = LCMScheduler

        pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            pipe.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=scheduler_class,
        )
        pipe.init_custom_adapter(
            num_views=num_views,
            self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0,
        )
        pipe.load_custom_adapter(adapter_path, "step1x-3d-ig2v.safetensors")
        pipe.to(device=device, dtype=dtype)
        pipe.cond_encoder.to(device=device, dtype=dtype)

        # load lora if provided
        if lora_model is not None:
            model_, name_ = lora_model.rsplit("/", 1)
            pipe.load_lora_weights(model_, weight_name=name_)

        return pipe

    def remove_bg(self, image, net, transform, device):
        image_size = image.size
        input_images = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = net(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image

    def preprocess_image(self, image, height, width):
        image = np.array(image)
        alpha = image[..., 3] > 0
        H, W = alpha.shape
        # get the bounding box of alpha
        y, x = np.where(alpha)
        y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
        x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
        image_center = image[y0:y1, x0:x1]
        # resize the longer side to H * 0.9
        H, W, _ = image_center.shape
        if H > W:
            W = int(W * (height * 0.9) / H)
            H = int(height * 0.9)
        else:
            H = int(H * (width * 0.9) / W)
            W = int(width * 0.9)
        image_center = np.array(Image.fromarray(image_center).resize((W, H)))
        # pad to H, W
        start_h = (height - H) // 2
        start_w = (width - W) // 2
        image = np.zeros((height, width, 4), dtype=np.uint8)
        image[start_h : start_h + H, start_w : start_w + W] = image_center
        image = image.astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = (image * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)

        return image

    def run_ig2mv_pipeline(
        self,
        pipe,
        mesh,
        num_views,
        text,
        image,
        height,
        width,
        num_inference_steps,
        guidance_scale,
        seed,
        remove_bg_fn=None,
        reference_conditioning_scale=1.0,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        lora_scale=1.0,
        device="cuda",
    ):
        # Prepare cameras
        cameras = get_orthogonal_camera(
            elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
            distance=[1.8] * num_views,
            left=-0.55,
            right=0.55,
            bottom=-0.55,
            top=0.55,
            azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
            device=device,
        )
        ctx = NVDiffRastContextWrapper(device=device, context_type="cuda")

        mesh, mesh_bp = load_mesh(mesh, rescale=True, device=device)
        render_out = render(
            ctx,
            mesh,
            cameras,
            height=height,
            width=width,
            render_attr=False,
            normal_background=0.0,
        )
        pos_images = tensor_to_image((render_out.pos + 0.5).clamp(0, 1), batched=True)
        normal_images = tensor_to_image(
            (render_out.normal / 2 + 0.5).clamp(0, 1), batched=True
        )
        control_images = (
            torch.cat(
                [
                    (render_out.pos + 0.5).clamp(0, 1),
                    (render_out.normal / 2 + 0.5).clamp(0, 1),
                ],
                dim=-1,
            )
            .permute(0, 3, 1, 2)
            .to(device)
        )

        # Prepare image
        reference_image = Image.open(image) if isinstance(image, str) else image
        if len(reference_image.split()) == 1:
            reference_image = reference_image.convert("RGBA")
        if remove_bg_fn is not None and reference_image.mode == "RGB":
            reference_image = remove_bg_fn(reference_image)
            reference_image = self.preprocess_image(reference_image, height, width)
        elif reference_image.mode == "RGBA":
            reference_image = self.preprocess_image(reference_image, height, width)

        pipe_kwargs = {}
        if seed != -1 and isinstance(seed, int):
            pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

        images = pipe(
            text,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_views,
            control_image=control_images,
            control_conditioning_scale=1.0,
            reference_image=reference_image,
            reference_conditioning_scale=reference_conditioning_scale,
            negative_prompt=negative_prompt,
            cross_attention_kwargs={"scale": lora_scale},
            mesh=mesh_bp,
            texture_sync_config=self.config.texture_sync_config,
            **pipe_kwargs,
        ).images

        return images, pos_images, normal_images, reference_image, mesh, mesh_bp

    def bake_from_multiview(
        self,
        render,
        views,
        camera_elevs,
        camera_azims,
        view_weights,
        method="graphcut",
        bake_exp=4,
    ):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        for view, camera_elev, camera_azim, weight in zip(
            views, camera_elevs, camera_azims, view_weights
        ):
            project_texture, project_cos_map, project_boundary_map = (
                render.back_project(view, camera_elev, camera_azim)
            )
            project_cos_map = weight * (project_cos_map**bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)

        if method == "fast":
            texture, ori_trust_map = render.fast_bake_texture(
                project_textures, project_weighted_cos_maps
            )
        else:
            raise f"no method {method}"
        return texture, ori_trust_map > 1e-8

    def texture_inpaint(self, render, texture, mask):
        texture_np = render.uv_inpaint(texture, mask)
        texture = torch.tensor(texture_np / 255).float().to(texture.device)

        return texture

    @torch.no_grad()
    def __call__(self, image, mesh, remove_bg=True, seed=2025):
        if remove_bg:
            birefnet = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet", trust_remote_code=True
            )
            birefnet.to(self.config.device)
            transform_image = transforms.Compose(
                [
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            remove_bg_fn = lambda x: self.remove_bg(
                x, birefnet, transform_image, self.config.device
            )
        else:
            remove_bg_fn = None

        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()

        # multi-view generation pipeline
        images, pos_images, normal_images, reference_image, textured_mesh, mesh_bp = (
            self.run_ig2mv_pipeline(
                self.ig2mv_pipe,
                mesh=mesh,
                num_views=self.config.num_views,
                text=self.config.text,
                image=image,
                height=768,
                width=768,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                seed= seed if seed is not None else self.config.seed,
                lora_scale=self.config.lora_scale,
                reference_conditioning_scale=self.config.reference_conditioning_scale,
                negative_prompt=self.config.negative_prompt,
                device=self.config.device,
                remove_bg_fn=remove_bg_fn,
            )
        )

        for i in range(len(images)):
            images[i] = images[i].resize(
                (self.config.render_size, self.config.render_size),
                Image.Resampling.LANCZOS,
            )

        mesh = self.mesh_uv_wrap(mesh_bp)
        self.mesh_render.load_mesh(mesh, auto_center=False, scale_factor=1.0)

        # texture baker
        texture, mask = self.bake_from_multiview(
            self.mesh_render,
            images,
            self.config.selected_camera_elevs,
            self.config.selected_camera_azims,
            self.config.selected_view_weights,
            method="fast",
        )
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        # texture inpaint
        texture = self.texture_inpaint(self.mesh_render, texture, mask_np)

        self.mesh_render.set_texture(texture)
        textured_mesh = self.mesh_render.save_mesh()

        return textured_mesh
