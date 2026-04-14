# Some parts of this file are refer to Hugging Face Diffusers library.
import os
import json
import warnings
from typing import Callable, List, Optional, Union, Dict, Any
import PIL.Image
import trimesh
import rembg
import torch
import numpy as np
from huggingface_hub import hf_hub_download

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.loaders import (
    FluxIPAdapterMixin,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
)
from .pipeline_utils import (
    TransformerDiffusionMixin,
    preprocess_image,
    retrieve_timesteps,
    remove_floater,
    remove_degenerate_face,
    reduce_face,
    smart_load_model,
)
from transformers import (
    BitImageProcessor,
)

import step1x3d_geometry
from step1x3d_geometry.models.autoencoders.surface_extractors import MeshExtractResult
from step1x3d_geometry.utils.config import ExperimentConfig, load_config
from ..autoencoders.michelangelo_autoencoder import MichelangeloAutoencoder
from ..conditional_encoders.dinov2_encoder import Dinov2Encoder
from ..conditional_encoders.t5_encoder import T5Encoder
from ..conditional_encoders.label_encoder import LabelEncoder
from ..transformers.flux_transformer_1d import FluxDenoiser


class Step1X3DGeometryPipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `torch.Tensor`):
            List of PIL images or a tensor representing the input images.
        meshes (`List[trimesh.Trimesh]` or `np.ndarray`)
            List of denoised trimesh meshes of length `batch_size` or a tuple of NumPy array with shape `((vertices, 3), (faces, 3)) of length `batch_size``.
    """

    image: PIL.Image.Image
    mesh: Union[trimesh.Trimesh, MeshExtractResult, np.ndarray]


class Step1X3DGeometryPipeline(
    DiffusionPipeline, FromSingleFileMixin, TransformerDiffusionMixin
):
    """
    Step1X-3D Geometry Pipeline, generate high-quality meshes conditioned on image/caption/label inputs

    Args:
        scheduler (FlowMatchEulerDiscreteScheduler):
            The diffusion scheduler controlling the denoising process
        vae (MichelangeloAutoencoder):
            Variational Autoencoder for latent space compression/reconstruction
        transformer (FluxDenoiser):
            Transformer-based denoising model
        visual_encoder (Dinov2Encoder):
            Pretrained visual encoder for image feature extraction
        caption_encoder (T5Encoder):
            Text encoder for processing natural language captions
        label_encoder (LabelEncoder):
            Auxiliary text encoder for label conditioning
        visual_eature_extractor (BitImageProcessor):
            Preprocessor for input images

    Note:
        - CPU offloading sequence: visual_encoder → caption_encoder → label_encoder → transformer → vae
        - Optional components: visual_encoder, visual_eature_extractor, caption_encoder, label_encoder
    """

    model_cpu_offload_seq = (
        "visual_encoder->caption_encoder->label_encoder->transformer->vae"
    )
    _optional_components = [
        "visual_encoder",
        "visual_eature_extractor",
        "caption_encoder",
        "label_encoder",
    ]

    @classmethod
    def from_pretrained(cls, model_path, subfolder='.', **kwargs):
        local_model_path = smart_load_model(model_path, subfolder)
        return super().from_pretrained(local_model_path, **kwargs)

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: MichelangeloAutoencoder,
        transformer: FluxDenoiser,
        visual_encoder: Dinov2Encoder,
        caption_encoder: T5Encoder,
        label_encoder: LabelEncoder,
        visual_eature_extractor: BitImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            visual_encoder=visual_encoder,
            caption_encoder=caption_encoder,
            label_encoder=label_encoder,
            visual_eature_extractor=visual_eature_extractor,
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def check_inputs(
        self,
        image,
    ):
        r"""
        Check if the inputs are valid. Raise an error if not.
        """
        if isinstance(image, str):
            assert os.path.isfile(image) or image.startswith(
                "http"
            ), "Input image must be a valid URL or a file path."
        elif isinstance(image, (torch.Tensor, PIL.Image.Image)):
            raise ValueError(
                "Input image must be a `torch.Tensor` or `PIL.Image.Image`."
            )

    def encode_image(self, image, device, num_meshes_per_prompt):
        dtype = next(self.visual_encoder.parameters()).dtype

        image_embeds = self.visual_encoder.encode_image(image)
        image_embeds = image_embeds.repeat_interleave(num_meshes_per_prompt, dim=0)

        uncond_image_embeds = self.visual_encoder.empty_image_embeds.repeat(
            image_embeds.shape[0], 1, 1
        ).to(image_embeds)

        return image_embeds, uncond_image_embeds

    def encode_caption(self, caption, device, num_meshes_per_prompt):
        dtype = next(self.label_encoder.parameters()).dtype

        caption_embeds = self.caption_encoder.encode_text([caption])
        caption_embeds = caption_embeds.repeat_interleave(num_meshes_per_prompt, dim=0)

        uncond_caption_embeds = self.caption_encoder.empty_text_embeds.repeat(
            caption_embeds.shape[0], 1, 1
        ).to(caption_embeds)

        return caption_embeds, uncond_caption_embeds

    def encode_label(self, label, device, num_meshes_per_prompt):
        dtype = next(self.label_encoder.parameters()).dtype

        label_embeds = self.label_encoder.encode_label([label])
        label_embeds = label_embeds.repeat_interleave(num_meshes_per_prompt, dim=0)

        uncond_label_embeds = self.label_encoder.empty_label_embeds.repeat(
            label_embeds.shape[0], 1, 1
        ).to(label_embeds)

        return label_embeds, uncond_label_embeds

    def prepare_latents(
        self,
        batch_size,
        num_tokens,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents: Optional[torch.Tensor] = None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (batch_size, num_tokens, num_channels_latents)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image, str],
        label: Optional[str] = None,
        caption: Optional[str] = None,
        num_inference_steps: int = 30,
        timesteps: List[int] = None,
        num_meshes_per_prompt: int = 1,
        guidance_scale: float = 7.5,
        generator: Optional[int] = None,
        latents: Optional[torch.FloatTensor] = None,
        force_remove_background: bool = False,
        background_color: List[int] = [255, 255, 255],
        foreground_ratio: float = 0.95,
        surface_extractor_type: Optional[str] = None,
        bounds: float = 1.05,
        mc_level: float = 0.0,
        octree_resolution: int = 384,
        output_type: str = "trimesh",
        do_remove_floater: bool = True,
        do_remove_degenerate_face: bool = False,
        do_reduce_face: bool = True,
        do_shade_smooth: bool = True,
        max_facenum: int = 200000,
        return_dict: bool = True,
        use_zero_init: Optional[bool] = True,
        zero_steps: Optional[int] = 0,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.FloatTensor` or `PIL.Image.Image` or `str`):
                `Image`, or tensor representing an image batch, or path to an image file. The image will be encoded to
                its CLIP/DINO-v2 embedding which the DiT will be conditioned on.
            label (`str`):
                The label of the generated mesh, like {"symmetry": "asymmetry", "edge_type": "smooth"}
            num_inference_steps (`int`, *optional*, defaults to 30):
                The number of denoising steps. More denoising steps usually lead to a higher quality mesh at the expense
                of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not provided, will use equally spaced timesteps.
            num_meshes_per_prompt (`int`, *optional*, defaults to 1):
                The number of meshes to generate per input image.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                Higher guidance scale encourages generation that closely matches the input image.
            generator (`int`, *optional*):
                A seed to make the generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents to use as inputs for mesh generation.
            force_remove_background (`bool`, *optional*, defaults to `False`):
                Whether to force remove the background from the input image before processing.
            background_color (`List[int]`, *optional*, defaults to `[255, 255, 255]`):
                RGB color values for the background if it needs to be removed or modified.
            foreground_ratio (`float`, *optional*, defaults to 0.95):
                Ratio of the image to consider as foreground when processing.
            surface_extractor_type (`str`, *optional*, defaults to "mc"):
                Type of surface extraction method to use ("mc" for Marching Cubes or other available methods).
            bounds (`float`, *optional*, defaults to 1.05):
                Bounding box size for the generated mesh.
            mc_level (`float`, *optional*, defaults to 0.0):
                Iso-surface level value for Marching Cubes extraction.
            octree_resolution (`int`, *optional*, defaults to 256):
                Resolution of the octree used for mesh generation.
            output_type (`str`, *optional*, defaults to "trimesh"):
                Type of output mesh format ("trimesh" or other supported formats).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a `MeshPipelineOutput` instead of a plain tuple.

        Returns:
            [`MeshPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`MeshPipelineOutput`] is returned, otherwise a `tuple` is returned where the
                first element is a list of generated meshes and the second element is a list of corresponding metadata.
        """
        # 0. Check inputs. Raise error if not correct
        self.check_inputs(
            image=image,
        )
        device = self._execution_device
        self._guidance_scale = guidance_scale

        # 1. Define call parameters
        if isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        elif isinstance(image, PIL.Image.Image) or isinstance(image, str):
            batch_size = 1

        # 2. Preprocess input image
        if isinstance(image, torch.Tensor):
            assert image.ndim == 3  # H, W, 3
            image_pil = TF.to_pil_image(image)
        elif isinstance(image, PIL.Image.Image):
            image_pil = image
        elif isinstance(image, str):
            if image.startswith("http"):
                import requests

                image_pil = PIL.Image.open(requests.get(image, stream=True).raw)
            else:
                image_pil = PIL.Image.open(image)
        image_pil = preprocess_image(image_pil, force=force_remove_background, background_color=background_color, foreground_ratio=foreground_ratio)  # remove the background images

        # 3. Encode condition
        image_embeds, negative_image_embeds = self.encode_image(
            image_pil, device, num_meshes_per_prompt
        )
        if self.do_classifier_free_guidance and image_embeds is not None:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)
        # 3.1 Encode label condition
        label_embeds = None
        if self.transformer.cfg.use_label_condition:
            if label is not None:
                label_embeds, negative_label_embeds = self.encode_label(
                    label, device, num_meshes_per_prompt
                )
                if self.do_classifier_free_guidance:
                    label_embeds = torch.cat(
                        [negative_label_embeds, label_embeds], dim=0
                    )
            else:
                uncond_label_embeds = self.label_encoder.empty_label_embeds.repeat(
                    num_meshes_per_prompt, 1, 1
                ).to(image_embeds)
                if self.do_classifier_free_guidance:
                    label_embeds = torch.cat(
                        [uncond_label_embeds, uncond_label_embeds], dim=0
                    )
        # 3.3 Encode caption condition
        caption_embeds = None
        if self.transformer.cfg.use_caption_condition:
            if caption is not None:
                caption_embeds, negative_caption_embeds = self.encode_caption(
                    caption, device, num_meshes_per_prompt
                )
                if self.do_classifier_free_guidance:
                    caption_embeds = torch.cat(
                        [negative_caption_embeds, caption_embeds], dim=0
                    )
            else:
                uncond_caption_embeds = self.caption_encoder.empty_text_embeds.repeat(
                    num_meshes_per_prompt, 1, 1
                ).to(image_embeds)
                if self.do_classifier_free_guidance:
                    caption_embeds = torch.cat(
                        [uncond_caption_embeds, uncond_caption_embeds], dim=0
                    )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_latents = self.vae.cfg.num_latents
        num_channels_latents = self.transformer.cfg.input_channels
        latents = self.prepare_latents(
            batch_size * num_meshes_per_prompt,
            num_latents,
            num_channels_latents,
            image_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    visual_condition=image_embeds,
                    label_condition=label_embeds,
                    caption_condition=caption_embeds,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )

                if (i <= zero_steps) and use_zero_init:
                    noise_pred = noise_pred * 0.0

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # 4. Post-processing
        if not output_type == "latent":
            if latents.dtype == torch.bfloat16:
                self.vae.to(torch.float16)
                latents = latents.to(torch.float16)
            mesh = self.vae.extract_geometry(
                self.vae.decode(latents),
                surface_extractor_type=surface_extractor_type,
                bounds=bounds,
                mc_level=mc_level,
                octree_resolution=octree_resolution,
                enable_pbar=False,
            )
            if output_type != "raw":
                mesh_list = []
                for i, cur_mesh in enumerate(mesh):
                    print(f"Generating mesh {i+1}/{num_meshes_per_prompt}")
                    if output_type == "trimesh":
                        import trimesh

                        cur_mesh = trimesh.Trimesh(
                            vertices=cur_mesh.verts.cpu().numpy(),
                            faces=cur_mesh.faces.cpu().numpy(),
                        )
                        cur_mesh.fix_normals()
                        cur_mesh.face_normals
                        cur_mesh.vertex_normals
                        cur_mesh.visual = trimesh.visual.TextureVisuals(
                            material=trimesh.visual.material.PBRMaterial(
                                baseColorFactor=(255, 255, 255),
                                main_color=(255, 255, 255),
                                metallicFactor=0.05,
                                roughnessFactor=1.0,
                            )
                        )
                        if do_remove_floater:
                            cur_mesh = remove_floater(cur_mesh)
                        if do_remove_degenerate_face:
                            cur_mesh = remove_degenerate_face(cur_mesh)
                        if do_reduce_face and max_facenum > 0:
                            cur_mesh = reduce_face(cur_mesh, max_facenum)
                        if do_shade_smooth:
                            cur_mesh = cur_mesh.smooth_shaded
                        mesh_list.append(cur_mesh)
                    elif output_type == "np":
                        if do_remove_floater:
                            print(
                                'remove floater is NOT used when output_type is "np". '
                            )
                        if do_remove_degenerate_face:
                            print(
                                'remove degenerate face is NOT used when output_type is "np". '
                            )
                        if do_reduce_face:
                            print(
                                'reduce floater is NOT used when output_type is "np". '
                            )
                        if do_shade_smooth:
                            print('shade smooth is NOT used when output_type is "np". ')
                        mesh_list.append(
                            [
                                cur_mesh[0].verts.cpu().numpy(),
                                cur_mesh[0].faces.cpu().numpy(),
                            ]
                        )
                mesh = mesh_list
            else:
                if do_remove_floater:
                    print('remove floater is NOT used when output_type is "raw". ')
                if do_remove_degenerate_face:
                    print(
                        'remove degenerate face is NOT used when output_type is "raw". '
                    )
                if do_reduce_face:
                    print('reduce floater is NOT used when output_type is "raw". ')

        else:
            mesh = latents

        if not return_dict:
            return tuple(image_pil), tuple(mesh)
        return Step1X3DGeometryPipelineOutput(image=image_pil, mesh=mesh)
