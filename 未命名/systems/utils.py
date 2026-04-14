import torch
import numpy as np
import math

from PIL import Image
from tqdm import tqdm
from diffusers import DDIMScheduler
from torchvision import transforms
from craftsman.utils.typing import *
from utils.transforms import *

import pdb

def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(timesteps.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u

def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """
    Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting

# from: https://github.com/genmoai/models/blob/075b6e36db58f1242921deff83a1066887b9c9e1/src/mochi_preview/infer.py#L77
def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [
        i * threshold_noise / linear_steps for i in range(linear_steps)
    ]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps *
                                                  quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (
        quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const
        for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

@torch.no_grad()
def ddim_sample(ddim_scheduler: DDIMScheduler,
                diffusion_model: torch.nn.Module,
                shape: Union[List[int], Tuple[int]],
                cond: torch.FloatTensor,
                steps: int,
                eta: float = 0.0,
                guidance_scale: float = 3.0,
                do_classifier_free_guidance: bool = True,
                generator: Optional[torch.Generator] = None,
                device: torch.device = "cuda:0",
                disable_prog: bool = True):

    assert steps > 0, f"{steps} must > 0."

    # init latents
    bsz = cond.shape[0]
    if do_classifier_free_guidance:
        bsz = bsz // 2
    latents = torch.randn(
        (bsz, *shape),
        generator=generator,
        device=cond.device,
        dtype=cond.dtype,
    )
    try:
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma
    except AttributeError:
        pass

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    extra_step_kwargs = {
        "generator": generator
    }
    
    # set timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        steps,
        device,
    )
    if eta > 0:
        assert 0 <= eta <= 1, f"eta must be between [0, 1]. Got {eta}."
        assert scheduler.__class__.__name__ == "DDIMScheduler", f"eta is only used with the DDIMScheduler."
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs["eta"] = eta

    # reverse
    for i, t in enumerate(tqdm(timesteps, disable=disable_prog, desc="DDIM Sampling:", leave=False)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2)
            if do_classifier_free_guidance
            else latents
        )

        # predict the noise residual
        timestep_tensor = torch.tensor([t], dtype=torch.long, device=device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        noise_pred = diffusion_model.forward(latent_model_input, timestep_tensor, cond)

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )

        # compute the previous noisy sample x_t -> x_t-1
        latents = ddim_scheduler.step(
            noise_pred, t, latents,  **extra_step_kwargs
        ).prev_sample

        yield latents, t

@torch.no_grad()
def flow_sample(scheduler: DDIMScheduler,
                diffusion_model: torch.nn.Module,
                shape: Union[List[int], Tuple[int]],
                cond: torch.FloatTensor,
                steps: int,
                eta: float = 0.0,
                guidance_scale: float = 3.0,
                do_classifier_free_guidance: bool = True,
                generator: Optional[torch.Generator] = None,
                device: torch.device = "cuda:0",
                disable_prog: bool = True):


    assert steps > 0, f"{steps} must > 0."

    # init latents
    bsz = cond.shape[0]
    if do_classifier_free_guidance:
        bsz = bsz // 2
    latents = torch.randn(
        (bsz, *shape),
        generator=generator,
        device=cond.device,
        dtype=cond.dtype,
    )
    try:
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma
    except AttributeError:
        pass

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    extra_step_kwargs = {
        "generator": generator
    }
    
    # set timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        steps + 1,
        device,
    )
    if eta > 0:
        assert 0 <= eta <= 1, f"eta must be between [0, 1]. Got {eta}."
        assert scheduler.__class__.__name__ == "DDIMScheduler", f"eta is only used with the DDIMScheduler."
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs["eta"] = eta

    # reverse
    distance = (timesteps[:-1] - timesteps[1:]) / scheduler.config.num_train_timesteps
    for i, t in enumerate(tqdm(timesteps[:-1], disable=disable_prog, desc="Flow Sampling:", leave=False)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2)
            if do_classifier_free_guidance
            else latents
        )
        # predict the noise residual
        timestep_tensor = torch.tensor([t], dtype=latents.dtype, device=device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        noise_pred = diffusion_model.forward(latent_model_input, timestep_tensor, cond)

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )

        # compute the previous noisy sample x_t -> x_t-1
        latents = latents - distance[i] * noise_pred

        yield latents, t

@torch.no_grad()
def flow_sample_pose(scheduler: DDIMScheduler,
                diffusion_model: torch.nn.Module,
                shape: Union[List[int], Tuple[int]],
                cond: torch.FloatTensor,
                cond_cat: torch.FloatTensor,
                steps: int,
                eta: float = 0.0,
                guidance_scale: float = 3.0,
                do_classifier_free_guidance: bool = True,
                generator: Optional[torch.Generator] = None,
                device: torch.device = "cuda:0",
                disable_prog: bool = True,
                attention_kwargs: Dict[str, torch.Tensor] = None,):


    assert steps > 0, f"{steps} must > 0."

    # init latents
    bsz = cond.shape[0]
    if do_classifier_free_guidance:
        bsz = bsz // 2
    latents = torch.randn(
        (bsz, *shape),
        generator=generator,
        device=cond.device,
        dtype=cond.dtype,
    )
    try:
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma
    except AttributeError:
        pass

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    extra_step_kwargs = {
        "generator": generator
    }
    
    # set timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        steps + 1,
        device,
    )
    if eta > 0:
        assert 0 <= eta <= 1, f"eta must be between [0, 1]. Got {eta}."
        assert scheduler.__class__.__name__ == "DDIMScheduler", f"eta is only used with the DDIMScheduler."
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs["eta"] = eta

    # reverse
    distance = (timesteps[:-1] - timesteps[1:]) / scheduler.config.num_train_timesteps
    for i, t in enumerate(tqdm(timesteps[:-1], disable=disable_prog, desc="Flow Sampling:", leave=False)):
        # concat condition
        B, T, N, C = latents.shape
        latents = torch.cat([latents, cond_cat], dim=-2)
        
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        # predict the noise residual
        timestep_tensor = torch.tensor([t], dtype=latents.dtype, device=device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        noise_pred = diffusion_model(latent_model_input, timestep_tensor, cond, attention_kwargs=attention_kwargs)

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )

        # compute the previous noisy sample x_t -> x_t-1
        latents = latents - distance[i] * noise_pred
        
        # split condition
        latents, _ = latents.split([N, cond_cat.shape[-2]], dim=-2)

        yield latents, t

@torch.no_grad()
def flow_sample_pose_direct(scheduler: DDIMScheduler,
                diffusion_model: torch.nn.Module,
                pose_ae: torch.nn.Module,
                shape: Union[List[int], Tuple[int]],
                cond: torch.FloatTensor,
                cond_cat: torch.FloatTensor,
                steps: int,
                eta: float = 0.0,
                guidance_scale: float = 3.0,
                do_classifier_free_guidance: bool = True,
                generator: Optional[torch.Generator] = None,
                device: torch.device = "cuda:0",
                disable_prog: bool = True,
                attention_kwargs: Dict[str, torch.Tensor] = None,):


    assert steps > 0, f"{steps} must > 0."

    # init latents
    bsz = cond.shape[0]
    if do_classifier_free_guidance:
        bsz = bsz // 2
    latents = torch.randn(
        (bsz, *shape),
        generator=generator,
        device=cond.device,
        dtype=cond.dtype,
    )
    try:
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma
    except AttributeError:
        pass

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    extra_step_kwargs = {
        "generator": generator
    }
    
    # set timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        steps + 1,
        device,
    )
    if eta > 0:
        assert 0 <= eta <= 1, f"eta must be between [0, 1]. Got {eta}."
        assert scheduler.__class__.__name__ == "DDIMScheduler", f"eta is only used with the DDIMScheduler."
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs["eta"] = eta

    # reverse
    distance = (timesteps[:-1] - timesteps[1:]) / scheduler.config.num_train_timesteps
    for i, t in enumerate(tqdm(timesteps[:-1], disable=disable_prog, desc="Flow Sampling:", leave=False)):
        # encode
        pose_latents = pose_ae.encode(rotation=latents[..., :6], translation=latents[..., 6:9], size=latents[..., 9:])
    
        # concat condition
        B, T, N, C = pose_latents.shape
        pose_latents = torch.cat([pose_latents, cond_cat], dim=-2)
        
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([pose_latents] * 2) if do_classifier_free_guidance else pose_latents

        # predict the noise residual
        timestep_tensor = torch.tensor([t], dtype=pose_latents.dtype, device=device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        noise_pred = diffusion_model(latent_model_input, timestep_tensor, cond, attention_kwargs=attention_kwargs)
        
        # split condition
        noise_pred, _ = noise_pred.split([N, cond_cat.shape[-2]], dim=-2)
        
        # decode
        noise_pred = pose_ae.decode(noise_pred)

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = latents - distance[i] * noise_pred

        yield latents, t

@torch.no_grad()
def flow_sample_pose_direct_midi(scheduler: DDIMScheduler,
                diffusion_model: torch.nn.Module,
                pose_ae: torch.nn.Module,
                shape: Union[List[int], Tuple[int]],
                shape_scene: Union[List[int], Tuple[int]],
                cond: torch.FloatTensor,
                cond_cat: torch.FloatTensor,
                steps: int,
                eta: float = 0.0,
                guidance_scale: float = 3.0,
                do_classifier_free_guidance: bool = True,
                generator: Optional[torch.Generator] = None,
                device: torch.device = "cuda:0",
                disable_prog: bool = True,
                attention_kwargs: Dict[str, torch.Tensor] = None,):

    assert steps > 0, f"{steps} must > 0."
    
    # init latents
    bsz = cond.shape[0]
    if do_classifier_free_guidance:
        bsz = bsz // 2
    latents = torch.randn((bsz, *shape), generator=generator, device=cond.device, dtype=cond.dtype,)
    scene_latents = torch.randn((*shape_scene,), generator=generator, device=cond.device, dtype=cond.dtype,)
    try:
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma
        scene_latents = scene_latents * scheduler.init_noise_sigma
    except AttributeError:
        pass

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    extra_step_kwargs = {
        "generator": generator
    }
    
    # set timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        steps + 1,
        device,
    )
    if eta > 0:
        assert 0 <= eta <= 1, f"eta must be between [0, 1]. Got {eta}."
        assert scheduler.__class__.__name__ == "DDIMScheduler", f"eta is only used with the DDIMScheduler."
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs["eta"] = eta

    # reverse
    distance = (timesteps[:-1] - timesteps[1:]) / scheduler.config.num_train_timesteps
    for i, t in enumerate(tqdm(timesteps[:-1], disable=disable_prog, desc="Flow Sampling:", leave=False)):
        # encode
        pose_latents = pose_ae.encode(rotation=latents[..., :6], translation=latents[..., 6:9], size=latents[..., 9:])
    
        # concat condition
        B, T, N, C = pose_latents.shape
        pose_latents = torch.cat([pose_latents, scene_latents, cond_cat], dim=-2)
        
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([pose_latents] * 2) if do_classifier_free_guidance else pose_latents

        # predict the noise residual
        timestep_tensor = torch.tensor([t], dtype=pose_latents.dtype, device=device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        noise_pred = diffusion_model(latent_model_input, timestep_tensor, cond, attention_kwargs=attention_kwargs)
        
        # split condition
        noise_pred, noise_pred_scene, _ = noise_pred.split([N, scene_latents.shape[-2], cond_cat.shape[-2]], dim=-2)
        
        # decode
        noise_pred = pose_ae.decode(noise_pred)

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # scene
            noise_pred_scene_uncond, noise_pred_scene_text = noise_pred_scene.chunk(2)
            noise_pred_scene = noise_pred_scene_uncond + guidance_scale * (noise_pred_scene_text - noise_pred_scene_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = latents - distance[i] * noise_pred
        scene_latents = scene_latents - distance[i] * noise_pred_scene

        yield latents, scene_latents, t
        
@torch.no_grad()
def flow_sample_pose_direct_transform(scheduler: DDIMScheduler,
                diffusion_model: torch.nn.Module,
                pose_ae: torch.nn.Module,
                shape_model: torch.nn.Module,
                shape: Union[List[int], Tuple[int]],
                cond: torch.FloatTensor,
                cond_cat: torch.FloatTensor,
                steps: int,
                surface: torch.FloatTensor=None,
                sharp_surface: torch.FloatTensor=None,
                eta: float = 0.0,
                guidance_scale: float = 3.0,
                do_classifier_free_guidance: bool = True,
                generator: Optional[torch.Generator] = None,
                device: torch.device = "cuda:0",
                disable_prog: bool = True,
                attention_kwargs: Dict[str, torch.Tensor] = None,):


    assert steps > 0, f"{steps} must > 0."

    # init latents
    bsz = cond.shape[0]
    if do_classifier_free_guidance:
        bsz = bsz // 2
    latents = torch.randn(
        (bsz, *shape),
        generator=generator,
        device=cond.device,
        dtype=cond.dtype,
    )
    try:
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma
    except AttributeError:
        pass

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    extra_step_kwargs = {
        "generator": generator
    }
    
    # set timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        steps + 1,
        device,
    )
    if eta > 0:
        assert 0 <= eta <= 1, f"eta must be between [0, 1]. Got {eta}."
        assert scheduler.__class__.__name__ == "DDIMScheduler", f"eta is only used with the DDIMScheduler."
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs["eta"] = eta

    # reverse
    distance = (timesteps[:-1] - timesteps[1:]) / scheduler.config.num_train_timesteps
    for i, t in enumerate(tqdm(timesteps[:-1], disable=disable_prog, desc="Flow Sampling:", leave=False)):
        # apply noisy
        bs, T, _ = latents.shape
        noisy_rot, noisy_trans, noisy_size = latents.reshape(-1, 10).split([6, 3, 1], dim=-1)
        transform_matrix_noisy = recover_transform_matrix(noisy_rot, noisy_trans).to(latents).unsqueeze(1)
        # apply on surface
        surface_pts = ((surface[...,:3] * (noisy_size[:,None,:]+1)/2).unsqueeze(-2) @ transform_matrix_noisy[...,:3,:3].transpose(-1,-2)) + transform_matrix_noisy[...,None,:3,3]
        surface_normal = surface[...,3:].unsqueeze(-2) @ transform_matrix_noisy[...,:3,:3].transpose(-1,-2)
        surface_input = torch.cat([surface_pts, surface_normal], dim=-1).reshape(*surface.shape)
        # apply on sharp surface
        if sharp_surface is not None:
            sharp_surface_pts = ((sharp_surface[...,:3] * (noisy_size[:,None,:]+1)/2).unsqueeze(-2) @ transform_matrix_noisy[...,:3,:3].transpose(-1,-2)) + transform_matrix_noisy[...,None,:3,3]
            sharp_surface_normal = sharp_surface[...,3:].unsqueeze(-2) @ transform_matrix_noisy[...,:3,:3].transpose(-1,-2)
            sharp_surface_input = torch.cat([sharp_surface_pts, sharp_surface_normal], dim=-1).reshape(*sharp_surface.shape)
        # encode
        shape_embeds, kl_embed, _ = shape_model.encode(
            surface_input, 
            sample_posterior=True,
            sharp_surface=sharp_surface_input, 
        )
        shape_latents = kl_embed
        shape_latents = shape_latents.view(bs, T, *shape_latents.shape[-2:]) # [B, T, N, C]
        
        # encode
        pose_latents = pose_ae.encode(rotation=latents[..., :6], translation=latents[..., 6:9], size=latents[..., 9:])
    
        # concat condition
        B, T, N, C = pose_latents.shape
        pose_latents = torch.cat([pose_latents, cond_cat], dim=-2)
        
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([pose_latents] * 2) if do_classifier_free_guidance else pose_latents

        # predict the noise residual
        timestep_tensor = torch.tensor([t], dtype=pose_latents.dtype, device=device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        noise_pred = diffusion_model(latent_model_input, timestep_tensor, cond, attention_kwargs=attention_kwargs)
        
        # split condition
        noise_pred, _ = noise_pred.split([N, cond_cat.shape[-2]], dim=-2)
        
        # decode
        noise_pred = pose_ae.decode(noise_pred)

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = latents - distance[i] * noise_pred

        yield latents, t
        
@torch.no_grad()
def flow_sample_pixart_scene(scheduler: DDIMScheduler,
                diffusion_model: torch.nn.Module,
                shape: Union[List[int], Tuple[int]],
                cond: torch.FloatTensor,
                cond_cat: torch.FloatTensor,
                steps: int,
                eta: float = 0.0,
                guidance_scale: float = 3.0,
                do_classifier_free_guidance: bool = True,
                generator: Optional[torch.Generator] = None,
                device: torch.device = "cuda:0",
                disable_prog: bool = True,
                attention_kwargs: Dict[str, torch.Tensor] = None,):


    assert steps > 0, f"{steps} must > 0."

    # init latents
    bsz = cond.shape[0]
    if do_classifier_free_guidance:
        bsz = bsz // 2
    latents = torch.randn(
        (bsz, *shape),
        generator=generator,
        device=cond.device,
        dtype=cond.dtype,
    )
    try:
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma
    except AttributeError:
        pass

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    extra_step_kwargs = {
        "generator": generator
    }
    
    # set timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        steps + 1,
        device,
    )
    if eta > 0:
        assert 0 <= eta <= 1, f"eta must be between [0, 1]. Got {eta}."
        assert scheduler.__class__.__name__ == "DDIMScheduler", f"eta is only used with the DDIMScheduler."
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs["eta"] = eta

    # reverse
    distance = (timesteps[:-1] - timesteps[1:]) / scheduler.config.num_train_timesteps
    for i, t in enumerate(tqdm(timesteps[:-1], disable=disable_prog, desc="Flow Sampling:", leave=False)):
        # concat condition
        B, T, N, C = latents.shape
        concat_latents = torch.cat([latents, cond_cat], dim=-2)
        
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([concat_latents] * 2) if do_classifier_free_guidance else concat_latents

        # predict the noise residual
        timestep_tensor = torch.tensor([t], dtype=latents.dtype, device=device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        noise_pred = diffusion_model(latent_model_input, timestep_tensor, cond, attention_kwargs=attention_kwargs)
        
        # split condition
        noise_pred, _ = noise_pred.split([N, cond_cat.shape[-2]], dim=-2)

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = latents - distance[i] * noise_pred

        yield latents, t

                
@torch.no_grad()
def flow_sample_scene(scheduler: DDIMScheduler,
                diffusion_model: torch.nn.Module,
                pose_ae: torch.nn.Module,
                pose_shape: Union[List[int], Tuple[int]],
                shape: Union[List[int], Tuple[int]],
                cond: torch.FloatTensor,
                steps: int,
                eta: float = 0.0,
                guidance_scale: float = 3.0,
                do_classifier_free_guidance: bool = True,
                generator: Optional[torch.Generator] = None,
                device: torch.device = "cuda:0",
                disable_prog: bool = True,
                attention_kwargs: Dict[str, torch.Tensor] = None,):


    assert steps > 0, f"{steps} must > 0."

    # init latents
    bsz = cond.shape[0]
    if do_classifier_free_guidance:
        bsz = bsz // 2
    shape_latents = torch.randn((bsz, *shape), generator=generator, device=cond.device, dtype=cond.dtype,)
    latents = torch.randn((bsz, *pose_shape), generator=generator, device=cond.device, dtype=cond.dtype,)
    try:
        # scale the initial noise by the standard deviation required by the scheduler
        shape_latents = shape_latents * scheduler.init_noise_sigma
        latents = latents * scheduler.init_noise_sigma
    except AttributeError:
        pass

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    extra_step_kwargs = {
        "generator": generator
    }
    
    # set timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        steps + 1,
        device,
    )
    if eta > 0:
        assert 0 <= eta <= 1, f"eta must be between [0, 1]. Got {eta}."
        assert scheduler.__class__.__name__ == "DDIMScheduler", f"eta is only used with the DDIMScheduler."
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs["eta"] = eta

    # reverse
    distance = (timesteps[:-1] - timesteps[1:]) / scheduler.config.num_train_timesteps
    for i, t in enumerate(tqdm(timesteps[:-1], disable=disable_prog, desc="Flow Sampling:", leave=False)):
        # encode
        pose_latents = pose_ae.encode(rotation=latents[..., :6], translation=latents[..., 6:9], size=latents[..., 9:])
    
        # concat condition
        B, T, N, C = pose_latents.shape
        pose_latents = torch.cat([pose_latents, shape_latents], dim=-2)
        
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([pose_latents] * 2) if do_classifier_free_guidance else pose_latents

        # predict the noise residual
        timestep_tensor = torch.tensor([t], dtype=pose_latents.dtype, device=device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        noise_pred = diffusion_model(latent_model_input, timestep_tensor, cond, attention_kwargs=attention_kwargs)
        
        # split condition
        noise_pred, noise_pred_shape = noise_pred.split([N, shape_latents.shape[-2]], dim=-2)
        
        # decode
        noise_pred = pose_ae.decode(noise_pred)

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # also apply guidance to shape latents
            noise_pred_shape_uncond, noise_pred_shape_text = noise_pred_shape.chunk(2)
            noise_pred_shape = noise_pred_shape_uncond + guidance_scale * (noise_pred_shape_text - noise_pred_shape_uncond)
            
        # compute the previous noisy sample x_t -> x_t-1
        latents = latents - distance[i] * noise_pred
        shape_latents = shape_latents - distance[i] * noise_pred_shape

        yield latents, shape_latents, t

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def read_image(img, img_size=224):
    transform = transforms.Compose(
            [
                transforms.Resize(img_size, transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(img_size),  # crop a (224, 224) square
                transforms.ToTensor()
            ]
        )
    rgb = Image.open(img)
    rgb = transform(rgb)[:3,...].permute(1, 2, 0)
    return rgb

def get_val_data(file_list, n_supervision=16384, geometry_type="occupancies"):
    batch_rand_points = []
    batch_occupancies = []
    for file in file_list:
        points = np.load(file)
        rand_points = np.asarray(points['points']) * 2 # range from -1.1 to 1.1
        rand_points = torch.from_numpy(rand_points)
        rand_points = torch.split(rand_points, n_supervision, dim=0)
        rand_points = torch.stack(rand_points[0:-1])
        occupancies = np.asarray(points[geometry_type])
        occupancies = np.unpackbits(occupancies)
        occupancies = torch.from_numpy(occupancies)
        occupancies = torch.split(occupancies, n_supervision, dim=0)
        occupancies = torch.stack(occupancies[0:-1])
        batch_rand_points.append(rand_points)
        batch_occupancies.append(occupancies)
    batch_rand_points = torch.stack(batch_rand_points)
    batch_occupancies= torch.stack(batch_occupancies)
    B, M, N, _ = batch_rand_points.shape
    return batch_rand_points.view(B*M, N, 3), batch_occupancies.view(B*M, N)

def compute_metric(shape_model, sample_inputs, sample_outputs, device):
    threshold = 0
    if len(sample_inputs['occupancy']) == 0:
        return 0, 0
    queries, labels = get_val_data(sample_inputs['occupancy'])
    latent = sample_outputs[0][:len(sample_inputs['occupancy']),...]
    queries = queries.to(latent)
    labels = labels.to(latent)
    latent = latent.unsqueeze(1).repeat(1, queries.shape[0] // latent.shape[0], 1, 1).view(queries.shape[0], latent.shape[1], latent.shape[2])
    outputs = shape_model.query(queries, latent)
    pred = torch.zeros_like(outputs)
    pred[outputs>=threshold] = 1
    torch.cuda.empty_cache()
    accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
    accuracy = accuracy.mean()
    intersection = (pred * labels).sum(dim=1)
    union = (pred + labels).gt(0).sum(dim=1)
    iou = intersection * 1.0 / union + 1e-5
    iou = iou.mean()

    return accuracy, iou
