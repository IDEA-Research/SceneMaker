from dataclasses import dataclass, field

from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
import numpy as np
import json
import copy
import torch
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
    compute_snr,
    free_memory,
)
import step1x3d_geometry
from step1x3d_geometry.systems.base import BaseSystem
from step1x3d_geometry.utils.misc import get_rank
from step1x3d_geometry.utils.typing import *
from diffusers import DDIMScheduler
from step1x3d_geometry.systems.utils import read_image, ddim_sample


# DEBUG = True
@step1x3d_geometry.register("diffusion-system")
class DiffusionSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        val_samples_json: str = ""
        bounds: float = 1.05
        mc_level: float = 0.0
        octree_resolution: int = 256
        skip_validation: bool = True

        # diffusion config
        z_scale_factor: float = 1.0
        guidance_scale: float = 7.5
        num_inference_steps: int = 50
        eta: float = 0.0
        snr_gamma: float = 5.0

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

        if self.cfg.caption_condition_type is not None:
            self.caption_condition = step1x3d_geometry.find(
                self.cfg.caption_condition_type
            )(self.cfg.caption_condition)

        if self.cfg.label_condition_type is not None:
            self.label_condition = step1x3d_geometry.find(
                self.cfg.label_condition_type
            )(self.cfg.label_condition)

        self.denoiser_model = step1x3d_geometry.find(self.cfg.denoiser_model_type)(
            self.cfg.denoiser_model
        )

        self.noise_scheduler = step1x3d_geometry.find(self.cfg.noise_scheduler_type)(
            **self.cfg.noise_scheduler
        )

        self.denoise_scheduler = step1x3d_geometry.find(
            self.cfg.denoise_scheduler_type
        )(**self.cfg.denoise_scheduler)

    def forward(self, batch: Dict[str, Any], skip_noise=False) -> Dict[str, Any]:
        # 1. encode shape latents
        if "sharp_surface" in batch.keys():
            sharp_surface = batch["sharp_surface"][
                ..., : 3 + self.cfg.shape_model.point_feats
            ]
        else:
            sharp_surface = None
        shape_embeds, kl_embed, _ = self.shape_model.encode(
            batch["surface"][..., : 3 + self.cfg.shape_model.point_feats],
            sample_posterior=True,
            sharp_surface=sharp_surface,
        )

        latents = kl_embed * self.cfg.z_scale_factor

        # 2. gain visual condition
        visual_cond_latents = None
        if self.cfg.visual_condition_type is not None:
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
                visual_cond_latents = self.visual_condition(batch).to(latents)
                latents = latents.unsqueeze(1).repeat(1, n_images, 1, 1)
                latents = latents.view(bs * n_images, *latents.shape[-2:])
            else:
                visual_cond_latents = self.visual_condition(batch).to(latents)

        ## 2.1 text condition if provided
        caption_cond_latents = None
        if self.cfg.caption_condition_type is not None:
            assert "caption" in batch.keys(), "caption is required for caption encoder"
            assert bs == len(
                batch["caption"]
            ), "Batch size must be the same as the caption length."
            caption_cond_latents = (
                self.caption_condition(batch)
                .repeat_interleave(n_images, dim=0)
                .to(latents)
            )

        ## 2.2 label condition if provided
        label_cond_latents = None
        if self.cfg.label_condition_type is not None:
            assert "label" in batch.keys(), "label is required for label encoder"
            assert bs == len(
                batch["label"]
            ), "Batch size must be the same as the label length."
            label_cond_latents = (
                self.label_condition(batch)
                .repeat_interleave(n_images, dim=0)
                .to(latents)
            )

        # 3. sample noise that we"ll add to the latents
        noise = torch.randn_like(latents).to(
            latents
        )  # [batch_size, n_token, latent_dim]
        bs = latents.shape[0]

        # 4. Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.cfg.noise_scheduler.num_train_timesteps,
            (bs,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # 5. add noise
        noisy_z = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 6. diffusion model forward
        output = self.denoiser_model(
            noisy_z,
            timesteps.long(),
            visual_cond_latents,
            caption_cond_latents,
            label_cond_latents,
        ).sample

        # 7. compute loss
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Prediction Type: {self.noise_scheduler.prediction_type} not supported."
            )
        if self.cfg.snr_gamma == 0:
            if self.cfg.loss.loss_type == "l1":
                loss = F.l1_loss(output, target, reduction="mean")
            elif self.cfg.loss.loss_type in ["mse", "l2"]:
                loss = F.mse_loss(output, target, reduction="mean")
            else:
                raise ValueError(f"Loss Type: {self.cfg.loss.loss_type} not supported.")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            mse_loss_weights = torch.stack(
                [snr, self.cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
            ).min(dim=1)[0]
            if self.noise_scheduler.config.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)

            if self.cfg.loss.loss_type == "l1":
                loss = F.l1_loss(output, target, reduction="none")
            elif self.cfg.loss.loss_type in ["mse", "l2"]:
                loss = F.mse_loss(output, target, reduction="none")
            else:
                raise ValueError(f"Loss Type: {self.cfg.loss.loss_type} not supported.")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
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
                        name += "_" + sample_inputs_["caption"][j].replace(" ", "_")

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
            sample_inputs["image"] = Step1X3DGeometryPipeline.preprocess_image(
                sample_inputs["image"], **kwargs
            )
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
                    len(sample_inputs["label"]), 1
                ).to(cond)
                label_cond = torch.cat([un_cond, cond], dim=0)

        latents_list = []
        if seed != None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None

        for _ in range(sample_times):
            sample_loop = ddim_sample(
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
