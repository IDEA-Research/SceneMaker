import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image
from PIL import ImageOps
import sys
import os

import pdb

def crop_square_with_padding(image, mask, bbox_ratio=0.9, pad_color=(255,255,255,255), crop_mode="padding"): 
    # obtain mask coordinates
    coordinates = mask.getbbox()
    if coordinates is None:
        return image

    left, upper, right, lower = coordinates
    width = right - left
    height = lower - upper
    side = max(width, height)
    crop_side = int(side / bbox_ratio)
    center_x = (left + right) // 2
    center_y = (upper + lower) // 2

    new_left = center_x - crop_side // 2
    new_upper = center_y - crop_side // 2
    new_right = new_left + crop_side
    new_lower = new_upper + crop_side

    pad_left = max(0, -new_left)
    pad_upper = max(0, -new_upper)
    pad_right = max(0, new_right - image.width)
    pad_lower = max(0, new_lower - image.height)

    crop_box = (
        max(new_left, 0),
        max(new_upper, 0),
        min(new_right, image.width),
        min(new_lower, image.height)
    )
    cropped = image.crop(crop_box)
    cropped_mask = mask.crop(crop_box)

    if any([pad_left, pad_upper, pad_right, pad_lower]):
        new_img = Image.new(image.mode, (crop_side, crop_side), pad_color)
        new_img.paste(cropped, (pad_left, pad_upper))
        new_mask = Image.new(mask.mode, (crop_side, crop_side), 0)
        new_mask.paste(cropped_mask, (pad_left, pad_upper))
        return new_img, new_mask
    else:
        return cropped, cropped_mask
    

def flux_deocc_crop_and_inpaint(pipe, image_path, mask_path, save_path, data_mode="instpifu", deocc_mode="flux", bbox_ratio=0.9, pad_color=(255,255,255,255), prompt="", guidance_scale=3.5):
    # load image and mask
    image = Image.open(image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    if min(image.size) < 512 or min(mask.size) < 512:
        scale = 512 / min(image.size)
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.BICUBIC)
        mask = mask.resize(new_size, Image.BICUBIC)
    
    # crop and padding to square
    cropped_img, cropped_mask = crop_square_with_padding(image, mask, bbox_ratio, pad_color)
    cropped_img_rgb = cropped_img.convert("RGB")
    # apply cropped mask on cropped image
    cropped_mask_inv = ImageOps.invert(cropped_mask.convert("L"))
    cropped_img_rgb.paste((255,255,255), mask=cropped_mask_inv)
    # save cropped image
    cropped_img_path = save_path.replace("deocclusion_images", "cropped_images")
    os.makedirs(os.path.dirname(cropped_img_path), exist_ok=True)
    cropped_img_rgb.save(cropped_img_path)

    # deocc with flux - use official recommended calling style
    if deocc_mode == "flux_inpaint":
        result = pipe(
            prompt=prompt, 
            image=cropped_img_rgb, 
            mask_image=cropped_mask_inv, 
            strength=1.0,
            generator=torch.Generator().manual_seed(42)
        ).images[0]
    else:
        # Official FLUX.1-Kontext calling style (from app.py)
        result = pipe(
            image=cropped_img_rgb, 
            prompt=prompt, 
            guidance_scale=guidance_scale, 
            width=cropped_img_rgb.size[0],
            height=cropped_img_rgb.size[1],
            num_inference_steps=28,
            generator=torch.Generator().manual_seed(42)
        ).images[0]
        
    # save
    result.save(save_path)
    print(f"Deoccluded image saved to {save_path}")
    
    return save_path
