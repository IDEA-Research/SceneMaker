from dataclasses import dataclass, field

import numpy as np
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
from einops import repeat
from tqdm import tqdm
from PIL import Image

from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
    KarrasVeScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
import step1x3d_geometry
from step1x3d_geometry.systems.base import BaseSystem
from step1x3d_geometry.utils.misc import get_rank
from step1x3d_geometry.utils.typing import *
from step1x3d_geometry.systems.utils import read_image, preprocess_image, flow_sample


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(timesteps.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


@step1x3d_geometry.register("rectified-flow-system")
class RectifiedFlowSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        skip_validation: bool = True
        val_samples_json: str = ""
        bounds: float = 1.05
        mc_level: float = 0.0
        octree_resolution: int = 256

        # diffusion config
        guidance_scale: float = 7.5
        num_inference_steps: int = 30
        eta: float = 0.0
        snr_gamma: float = 5.0

        # flow
        weighting_scheme: str = "logit_normal"
        logit_mean: float = 0
        logit_std: float = 1.0
        mode_scale: float = 1.29
        precondition_outputs: bool = True
        precondition_t: int = 1000

        # shape vae model
        shape_model_type: str = None
        shape_model: dict = field(default_factory=dict)

        # condition model
        visual_condition_type: Optional[str] = None
        visual_condition: dict = field(default_factory=dict)
        caption_condition_type: Optional[str] = None
        caption_condition: dict = field(default_factory=dict)
        label_condition_type: Optional[str] = None
        label_condition: dict = field(default_factory=dict)

        # diffusion model
        denoiser_model_type: str = None
        denoiser_model: dict = field(default_factory=dict)

        # noise scheduler
        noise_scheduler_type: str = None
        noise_scheduler: dict = field(default_factory=dict)

        # denoise scheduler
        denoise_scheduler_type: str = None
        denoise_scheduler: dict = field(default_factory=dict)

        # lora
        use_lora: bool = False
        lora_layers: Optional[str] = None
        rank: int = 128  # The dimension of the LoRA update matrices.
        alpha: int = 128

    cfg: Config

    def configure(self):
        super().configure()

        self.shape_model = step1x3d_geometry.find(self.cfg.shape_model_type)(
            self.cfg.shape_model
        )
        self.shape_model.eval()
        self.shape_model.requires_grad_(False)

        if self.cfg.visual_condition_type is not None:
            self.visual_condition = step1x3d_geometry.find(
                self.cfg.visual_condition_type
            )(self.cfg.visual_condition)
            self.visual_condition.requires_grad_(False)

        if self.cfg.caption_condition_type is not None:
            self.caption_condition = step1x3d_geometry.find(
                self.cfg.caption_condition_type
            )(self.cfg.caption_condition)
            self.caption_condition.requires_grad_(False)

        if self.cfg.label_condition_type is not None:
            self.label_condition = step1x3d_geometry.find(
                self.cfg.label_condition_type
            )(self.cfg.label_condition)

        self.denoiser_model = step1x3d_geometry.find(self.cfg.denoiser_model_type)(
            self.cfg.denoiser_model
        )
        if self.cfg.use_lora:  # We only train the additional adapter LoRA layers
            self.denoiser_model.requires_grad_(False)

        self.noise_scheduler = step1x3d_geometry.find(self.cfg.noise_scheduler_type)(
            **self.cfg.noise_scheduler
        )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)

        self.denoise_scheduler = step1x3d_geometry.find(
            self.cfg.denoise_scheduler_type
        )(**self.cfg.denoise_scheduler)

        if self.cfg.use_lora:
            from peft import LoraConfig, set_peft_model_state_dict

            if self.cfg.lora_layers is not None:
                self.target_modules = [
                    layer.strip() for layer in self.cfg.lora_layers.split(",")
                ]
            else:
                self.target_modules = [
                    "attn.to_k",
                    "attn.to_q",
                    "attn.to_v",
                    "attn.to_out.0",
                    "attn.add_k_proj",
                    "attn.add_q_proj",
                    "attn.add_v_proj",
                    "attn.to_add_out",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "ff_context.net.0.proj",
                    "ff_context.net.2",
                ]
                self.transformer_lora_config = LoraConfig(
                    r=self.cfg.rank,
                    lora_alpha=self.cfg.alpha,
                    init_lora_weights="gaussian",
                    target_modules=self.target_modules,
                )
                self.denoiser_model.dit_model.add_adapter(self.transformer_lora_config)

    def forward(self, batch: Dict[str, Any], skip_noise=False) -> Dict[str, Any]:
        # 1. encode shape latents
        if "sharp_surface" in batch.keys():
            sharp_surface = batch["sharp_surface"][
                ..., : 3 + self.cfg.shape_model.point_feats
            ]
        else:
            sharp_surface = None
        shape_embeds, latents, _ = self.shape_model.encode(
            batch["surface"][..., : 3 + self.cfg.shape_model.point_feats],
            sample_posterior=True,
            sharp_surface=sharp_surface,
        )

        # 2. gain visual condition
        visual_cond = None
        if self.cfg.visual_condition_type is not None:
            assert "image" in batch.keys(), "image is required for label encoder"
            if "image" in batch and batch["image"].dim() == 5:
                if self.training:
                    bs, n_images = batch["image"].shape[:2]
                    batch["image"] = batch["image"].view(
                        bs * n_images, *batch["image"].shape[-3:]
                    )
                else:
                    batch["image"] = batch["image"][:, 0, ...]
                    n_images = 1
                    bs = batch["image"].shape[0]
                visual_cond = self.visual_condition(batch).to(latents)
                latents = latents.unsqueeze(1).repeat(1, n_images, 1, 1)
                latents = latents.view(bs * n_images, *latents.shape[-2:])
            else:
                visual_cond = self.visual_condition(batch).to(latents)
                bs = visual_cond.shape[0]
                n_images = 1

        ## 2.1 text condition if provided
        caption_cond = None
        if self.cfg.caption_condition_type is not None:
            assert "caption" in batch.keys(), "caption is required for caption encoder"
            assert bs == len(
                batch["caption"]
            ), "Batch size must be the same as the caption length."
            caption_cond = (
                self.caption_condition(batch)
                .repeat_interleave(n_images, dim=0)
                .to(latents)
            )

        ## 2.2 label condition if provided
        label_cond = None
        if self.cfg.label_condition_type is not None:
            assert "label" in batch.keys(), "label is required for label encoder"
            assert bs == len(
                batch["label"]
            ), "Batch size must be the same as the label length."
            label_cond = (
                self.label_condition(batch)
                .repeat_interleave(n_images, dim=0)
                .to(latents)
            )

        # 3. sample noise that we"ll add to the latents
        noise = torch.randn_like(latents).to(
            latents
        )  # [batch_size, n_token, latent_dim]

        # 4. Sample a random timestep
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.cfg.weighting_scheme,
            batch_size=bs * n_images,
            logit_mean=self.cfg.logit_mean,
            logit_std=self.cfg.logit_std,
            mode_scale=self.cfg.mode_scale,
        )
        indices = (u * self.cfg.noise_scheduler.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(
            device=latents.device
        )

        # 5. add noise
        sigmas = get_sigmas(
            self.noise_scheduler_copy, timesteps, n_dim=3, dtype=latents.dtype
        )
        noisy_z = (1.0 - sigmas) * latents + sigmas * noise

        # 6. diffusion model forward
        output = self.denoiser_model(
            noisy_z, timesteps.long(), visual_cond, caption_cond, label_cond
        ).sample

        # 7. compute loss
        if self.cfg.precondition_outputs:
            output = output * (-sigmas) + noisy_z
        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.cfg.weighting_scheme, sigmas=sigmas
        )
        # flow matching loss
        if self.cfg.precondition_outputs:
            target = latents
        else:
            target = noise - latents

        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (output.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            1,
        )
        loss = loss.mean()

        return {
            "loss_diffusion": loss,
            "latents": latents,
            "x_t": noisy_z,
            "noise": noise,
            "noise_pred": output,
            "timesteps": timesteps,
        }

    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.0
        for name, value in out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
            if name.startswith("log_"):
                self.log(f"log/{name.replace('log_', '')}", value.mean())

        for name, value in self.cfg.loss.items():
            if name.startswith("lambda_"):
                self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.cfg.skip_validation:
            return {}
        self.eval()

        if get_rank() == 0:
            sample_inputs = json.loads(
                open(self.cfg.val_samples_json).read()
            )  # condition
            sample_inputs_ = copy.deepcopy(sample_inputs)
            sample_outputs = self.sample(sample_inputs)  # list
            for i, latents in enumerate(sample_outputs["latents"]):
                meshes = self.shape_model.extract_geometry(
                    latents,
                    bounds=self.cfg.bounds,
                    mc_level=self.cfg.mc_level,
                    octree_resolution=self.cfg.octree_resolution,
                    enable_pbar=False,
                )

                for j in range(len(meshes)):
                    name = ""
                    if "image" in sample_inputs_:
                        name += (
                            sample_inputs_["image"][j]
                            .split("/")[-1]
                            .replace(".png", "")
                        )

                    elif "mvimages" in sample_inputs_:
                        name += (
                            sample_inputs_["mvimages"][j][0]
                            .split("/")[-2]
                            .replace(".png", "")
                        )

                    if "caption" in sample_inputs_:
                        name += "_" + sample_inputs_["caption"][j].replace(
                            " ", "_"
                        ).replace(".", "")

                    if "label" in sample_inputs_:
                        name += (
                            "_"
                            + sample_inputs_["label"][j]["symmetry"]
                            + sample_inputs_["label"][j]["edge_type"]
                        )

                    if (
                        meshes[j].verts is not None
                        and meshes[j].verts.shape[0] > 0
                        and meshes[j].faces is not None
                        and meshes[j].faces.shape[0] > 0
                    ):
                        self.save_mesh(
                            f"it{self.true_global_step}/{name}_{i}.obj",
                            meshes[j].verts,
                            meshes[j].faces,
                        )
                        torch.cuda.empty_cache()

        out = self(batch)
        if self.global_step == 0:
            latents = self.shape_model.decode(out["latents"])
            meshes = self.shape_model.extract_geometry(
                latents,
                bounds=self.cfg.bounds,
                mc_level=self.cfg.mc_level,
                octree_resolution=self.cfg.octree_resolution,
                enable_pbar=False,
            )

            for i, mesh in enumerate(meshes):
                self.save_mesh(
                    f"it{self.true_global_step}/{batch['uid'][i]}.obj",
                    mesh.verts,
                    mesh.faces,
                )

        return {"val/loss": out["loss_diffusion"]}

    @torch.no_grad()
    def sample(
        self,
        sample_inputs: Dict[str, Union[torch.FloatTensor, List[str]]],
        sample_times: int = 1,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        eta: float = 0.0,
        seed: Optional[int] = None,
        **kwargs,
    ):

        if steps is None:
            steps = self.cfg.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.cfg.guidance_scale
        do_classifier_free_guidance = guidance_scale != 1.0

        # conditional encode
        visal_cond = None
        if "image" in sample_inputs:
            sample_inputs["image"] = [
                Image.open(img) if type(img) == str else img
                for img in sample_inputs["image"]
            ]
            sample_inputs["image"] = preprocess_image(sample_inputs["image"], **kwargs)
            cond = self.visual_condition.encode_image(sample_inputs["image"])
            if do_classifier_free_guidance:
                un_cond = self.visual_condition.empty_image_embeds.repeat(
                    len(sample_inputs["image"]), 1, 1
                ).to(cond)
                visal_cond = torch.cat([un_cond, cond], dim=0)
        caption_cond = None
        if "caption" in sample_inputs:
            cond = self.label_condition.encode_label(sample_inputs["caption"])
            if do_classifier_free_guidance:
                un_cond = self.caption_condition.empty_caption_embeds.repeat(
                    len(sample_inputs["caption"]), 1, 1
                ).to(cond)
                caption_cond = torch.cat([un_cond, cond], dim=0)
        label_cond = None
        if "label" in sample_inputs:
            cond = self.label_condition.encode_label(sample_inputs["label"])
            if do_classifier_free_guidance:
                un_cond = self.label_condition.empty_label_embeds.repeat(
                    len(sample_inputs["label"]), 1, 1
                ).to(cond)
                label_cond = torch.cat([un_cond, cond], dim=0)

        latents_list = []
        if seed != None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None

        for _ in range(sample_times):
            sample_loop = flow_sample(
                self.denoise_scheduler,
                self.denoiser_model.eval(),
                shape=self.shape_model.latent_shape,
                visual_cond=visal_cond,
                caption_cond=caption_cond,
                label_cond=label_cond,
                steps=steps,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=self.device,
                eta=eta,
                disable_prog=False,
                generator=generator,
            )
            for sample, t in sample_loop:
                latents = sample
            latents_list.append(self.shape_model.decode(latents))

        return {"latents": latents_list, "inputs": sample_inputs}

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        return
