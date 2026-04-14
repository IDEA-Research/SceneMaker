import torch
import numpy as np

import rembg
from PIL import Image
from tqdm import tqdm
from diffusers import DDIMScheduler
from torchvision import transforms

from step1x3d_geometry.utils.typing import *
from step1x3d_geometry.utils.misc import get_device


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
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
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
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
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
def ddim_sample(
    ddim_scheduler: DDIMScheduler,
    diffusion_model: torch.nn.Module,
    shape: Union[List[int], Tuple[int]],
    visual_cond: torch.FloatTensor,
    caption_cond: torch.FloatTensor,
    label_cond: torch.FloatTensor,
    steps: int,
    eta: float = 0.0,
    guidance_scale: float = 3.0,
    do_classifier_free_guidance: bool = True,
    generator: Optional[torch.Generator] = None,
    device: torch.device = "cuda:0",
    disable_prog: bool = True,
):

    assert steps > 0, f"{steps} must > 0."

    # init latents
    if visual_cond is not None:
        bsz = visual_cond.shape[0]
        device = visual_cond.device
        dtype = visual_cond.dtype
    if caption_cond is not None:
        bsz = caption_cond.shape[0]
        device = caption_cond.device
        dtype = caption_cond.dtype
    if label_cond is not None:
        bsz = label_cond.shape[0]
        device = label_cond.device
        dtype = label_cond.dtype

    if do_classifier_free_guidance:
        bsz = bsz // 2
    latents = torch.randn(
        (bsz, *shape),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    try:
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma
    except AttributeError:
        pass

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    extra_step_kwargs = {"generator": generator}

    # set timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        steps,
        device,
    )
    if eta > 0:
        assert 0 <= eta <= 1, f"eta must be between [0, 1]. Got {eta}."
        assert (
            scheduler.__class__.__name__ == "DDIMScheduler"
        ), f"eta is only used with the DDIMScheduler."
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs["eta"] = eta

    # reverse
    for i, t in enumerate(
        tqdm(timesteps, disable=disable_prog, desc="DDIM Sampling:", leave=False)
    ):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )

        # predict the noise residual
        timestep_tensor = torch.tensor([t], dtype=torch.long, device=device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        noise_pred = diffusion_model.forward(
            latent_model_input, timestep_tensor, visual_cond, caption_cond, label_cond
        ).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # compute the previous noisy sample x_t -> x_t-1
        latents = ddim_scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs
        ).prev_sample

        yield latents, t


@torch.no_grad()
def flow_sample(
    scheduler: DDIMScheduler,
    diffusion_model: torch.nn.Module,
    shape: Union[List[int], Tuple[int]],
    visual_cond: torch.FloatTensor,
    caption_cond: torch.FloatTensor,
    label_cond: torch.FloatTensor,
    steps: int,
    eta: float = 0.0,
    guidance_scale: float = 3.0,
    do_classifier_free_guidance: bool = True,
    generator: Optional[torch.Generator] = None,
    device: torch.device = "cuda:0",
    disable_prog: bool = True,
):

    assert steps > 0, f"{steps} must > 0."

    # init latents
    if visual_cond is not None:
        bsz = visual_cond.shape[0]
        device = visual_cond.device
        dtype = visual_cond.dtype
    if caption_cond is not None:
        bsz = caption_cond.shape[0]
        device = caption_cond.device
        dtype = caption_cond.dtype
    if label_cond is not None:
        bsz = label_cond.shape[0]
        device = label_cond.device
        dtype = label_cond.dtype

    if do_classifier_free_guidance:
        bsz = bsz // 2
    latents = torch.randn(
        (bsz, *shape),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    try:
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma
    except AttributeError:
        pass

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    extra_step_kwargs = {"generator": generator}

    # set timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        steps + 1,
        device,
    )
    if eta > 0:
        assert 0 <= eta <= 1, f"eta must be between [0, 1]. Got {eta}."
        assert (
            scheduler.__class__.__name__ == "DDIMScheduler"
        ), f"eta is only used with the DDIMScheduler."
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs["eta"] = eta

    # reverse
    distance = (timesteps[:-1] - timesteps[1:]) / scheduler.config.num_train_timesteps
    for i, t in enumerate(
        tqdm(timesteps[:-1], disable=disable_prog, desc="Flow Sampling:", leave=False)
    ):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        # predict the noise residual
        timestep_tensor = torch.tensor([t], dtype=latents.dtype, device=device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        noise_pred = diffusion_model.forward(
            latent_model_input, timestep_tensor, visual_cond, caption_cond, label_cond
        ).sample
        if isinstance(noise_pred, tuple):
            noise_pred, layer_idx_list, ones_list, pred_c_list = noise_pred

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # compute the previous noisy sample x_t -> x_t-1
        latents = latents - distance[i] * noise_pred

        yield latents, t


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
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def read_image(img, img_size=224):
    transform = transforms.Compose(
        [
            transforms.Resize(
                img_size, transforms.InterpolationMode.BICUBIC, antialias=True
            ),
            transforms.CenterCrop(img_size),  # crop a (224, 224) square
            transforms.ToTensor(),
        ]
    )
    rgb = Image.open(img)
    rgb = transform(rgb)[:3, ...].permute(1, 2, 0)
    return rgb


def preprocess_image(
    images_pil: List[Image.Image],
    force: bool = False,
    background_color: List[int] = [255, 255, 255],
    foreground_ratio: float = 0.95,
):
    r"""
    Crop and remote the background of the input image
    Args:
        image_pil (`List[PIL.Image.Image]`):
            List of `PIL.Image.Image` objects representing the input image.
        force (`bool`, *optional*, defaults to `False`):
            Whether to force remove the background even if the image has an alpha channel.
    Returns:
        `List[PIL.Image.Image]`: List of `PIL.Image.Image` objects representing the preprocessed image.
    """
    preprocessed_images = []
    for i in range(len(images_pil)):
        image = images_pil[i]
        width, height, size = image.width, image.height, image.size
        do_remove = True
        if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
            # explain why current do not rm bg
            print(
                "alhpa channl not empty, skip remove background, using alpha channel as mask"
            )
            do_remove = False
        do_remove = do_remove or force
        if do_remove:
            image = rembg.remove(image)

        # calculate the min bbox of the image
        alpha = image.split()[-1]
        bboxs = alpha.getbbox()
        x1, y1, x2, y2 = bboxs
        dy, dx = y2 - y1, x2 - x1
        s = min(height * foreground_ratio / dy, width * foreground_ratio / dx)
        Ht, Wt = int(dy * s), int(dx * s)

        background = Image.new("RGBA", image.size, (*background_color, 255))
        image = Image.alpha_composite(background, image)
        image = image.crop(alpha.getbbox())
        alpha = alpha.crop(alpha.getbbox())

        # Calculate the new size after rescaling
        new_size = tuple(int(dim * foreground_ratio) for dim in size)
        # Resize the image while maintaining the aspect ratio
        resized_image = image.resize((Wt, Ht))
        resized_alpha = alpha.resize((Wt, Ht))
        # Create a new image with the original size and white background
        padded_image = Image.new("RGB", size, tuple(background_color))
        padded_alpha = Image.new("L", size, (0))
        paste_position = (
            (width - resized_image.width) // 2,
            (height - resized_image.height) // 2,
        )
        padded_image.paste(resized_image, paste_position)
        padded_alpha.paste(resized_alpha, paste_position)

        # expand image to 1:1
        width, height = padded_image.size
        if width == height:
            padded_image.putalpha(padded_alpha)
            preprocessed_images.append(padded_image)
            continue
        new_size = (max(width, height), max(width, height))
        new_image = Image.new("RGB", new_size, tuple(background_color))
        new_alpha = Image.new("L", new_size, (0))
        paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
        new_image.paste(padded_image, paste_position)
        new_alpha.paste(padded_alpha, paste_position)
        new_image.putalpha(new_alpha)
        preprocessed_images.append(new_image)

    return preprocessed_images
