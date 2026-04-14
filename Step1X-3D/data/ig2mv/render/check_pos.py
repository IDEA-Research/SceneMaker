
import cv2 
import os
import torch
import json
from typing import List, Optional, Union
from PIL import Image
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def load_depth(path, height, width):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    depth = torch.from_numpy(depth[..., 0:1]).float()
    mask = torch.ones_like(depth)
    mask[depth > 1000.0] = 0.0  # depth = 65535 is the invalid value
    depth[~(mask > 0.5)] = 0.0
    return depth, mask


def tensor_to_image(
    data: Union[Image.Image, torch.Tensor, np.ndarray],
    batched: bool = False,
    format: str = "HWC",
) -> Union[Image.Image, List[Image.Image]]:
    if isinstance(data, Image.Image):
        return data
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if data.dtype == np.float32 or data.dtype == np.float16:
        data = (data * 255).astype(np.uint8)
    elif data.dtype == np.bool_:
        data = data.astype(np.uint8) * 255
    assert data.dtype == np.uint8
    if format == "CHW":
        if batched and data.ndim == 4:
            data = data.transpose((0, 2, 3, 1))
        elif not batched and data.ndim == 3:
            data = data.transpose((1, 2, 0))

    if batched:
        return [Image.fromarray(d) for d in data]
    return Image.fromarray(data)

def get_position_map_from_depth_ortho(
    depth, mask, extrinsics, ortho_scale, image_wh=None
):
    """Compute the position map from the depth map and the camera parameters for a batch of views
    using orthographic projection with a given ortho_scale.

    Args:
        depth (torch.Tensor): The depth maps with the shape (B, H, W, 1).
        mask (torch.Tensor): The masks with the shape (B, H, W, 1).
        extrinsics (torch.Tensor): The camera extrinsics matrices with the shape (B, 4, 4).
        ortho_scale (torch.Tensor): The scaling factor for the orthographic projection with the shape (B, 1, 1, 1).
        image_wh (Tuple[int, int]): Optional. The image width and height.

    Returns:
        torch.Tensor: The position maps with the shape (B, H, W, 3).
    """
    if image_wh is None:
        image_wh = depth.shape[2], depth.shape[1]

    B, H, W, _ = depth.shape
    depth = depth.squeeze(-1)

    # Generating grid of coordinates in the image space
    u_coord, v_coord = torch.meshgrid(
        torch.arange(0, image_wh[0]), torch.arange(0, image_wh[1]), indexing="xy"
    )
    u_coord = u_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)
    v_coord = v_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)

    # Compute the position map using orthographic projection with ortho_scale
    x = (u_coord - image_wh[0] / 2) * ortho_scale / image_wh[0]
    y = (v_coord - image_wh[1] / 2) * ortho_scale / image_wh[1]
    z = depth

    # Concatenate to form the 3D coordinates in the camera frame
    camera_coords = torch.stack([x, y, z], dim=-1)

    # Apply the extrinsic matrix to get coordinates in the world frame
    coords_homogeneous = torch.nn.functional.pad(
        camera_coords, (0, 1), "constant", 1.0
    )  # Add a homogeneous coordinate
    world_coords = torch.matmul(
        coords_homogeneous.view(B, -1, 4), extrinsics.transpose(1, 2)
    ).view(B, H, W, 4)

    # Apply the mask to the position map
    position_map = world_coords[..., :3] * mask

    return position_map


def make_image_grid(
    images: List[Image.Image],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    resize: Optional[int] = None,
) -> Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    if rows is None and cols is not None:
        assert len(images) % cols == 0
        rows = len(images) // cols
    elif cols is None and rows is not None:
        assert len(images) % rows == 0
        cols = len(images) // rows
    elif rows is None and cols is None:
        rows = largest_factor_near_sqrt(len(images))
        cols = len(images) // rows

    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


scene_dir = "/data/projects/Step1X-3D/data/ig2mv/render/save_renders/6b0285934db445998b49efafc8c57de7"
depth_masks = [
                    load_depth(
                        os.path.join(scene_dir, f"depth_000{f}.exr"),
                        768,
                        768,
                    )
                    for f in [0,1,2,3,4,5]
                ]

depths = torch.stack([d for d, _ in depth_masks])
masks = torch.stack([m for _, m in depth_masks])

# camera
with open(os.path.join(scene_dir, "meta.json")) as f:
            meta = json.load(f)
name2loc = {loc["index"]: loc for loc in meta["locations"]}
c2w = [
    torch.as_tensor(name2loc[f"000{name}"]["transform_matrix"])
    for name in [0,1,2,3,4,5]
]
c2w = torch.stack(c2w, dim=0)

c2w_ = c2w.clone()
c2w_[:, :, 1:3] *= -1
ortho_scale = 1.1
position_maps = get_position_map_from_depth_ortho(
        depths,
        masks,
        c2w_,
        ortho_scale,
        image_wh=(768, 768),
    )

position_offset = 0.5
position_scale = 1.0
position_maps = (
    (position_maps + position_offset) / position_scale
).clamp(0.0, 1.0)

pos_images = tensor_to_image(position_maps,  batched=True)

make_image_grid(pos_images, rows=1).save("./pos.png")
