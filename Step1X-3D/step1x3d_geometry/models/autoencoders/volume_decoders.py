# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from typing import Union, Tuple, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from tqdm import tqdm

cube_corners = torch.tensor(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=torch.int,
)


def extract_near_surface_volume_fn(input_tensor: torch.Tensor, alpha: float):
    device = input_tensor.device
    D = input_tensor.shape[0]
    signed_val = 0.0

    # 添加偏移并处理无效值
    val = input_tensor + alpha
    valid_mask = val > -9000  # 假设-9000是无效值

    # 改进的邻居获取函数（保持维度一致）
    def get_neighbor(t, shift, axis):
        """根据指定轴进行位移并保持维度一致"""
        if shift == 0:
            return t.clone()

        # 确定填充轴（输入为[D, D, D]对应z,y,x轴）
        pad_dims = [0, 0, 0, 0, 0, 0]  # 格式：[x前，x后，y前，y后，z前，z后]

        # 根据轴类型设置填充
        if axis == 0:  # x轴（最后一个维度）
            pad_idx = 0 if shift > 0 else 1
            pad_dims[pad_idx] = abs(shift)
        elif axis == 1:  # y轴（中间维度）
            pad_idx = 2 if shift > 0 else 3
            pad_dims[pad_idx] = abs(shift)
        elif axis == 2:  # z轴（第一个维度）
            pad_idx = 4 if shift > 0 else 5
            pad_dims[pad_idx] = abs(shift)

        # 执行填充（添加batch和channel维度适配F.pad）
        padded = F.pad(
            t.unsqueeze(0).unsqueeze(0), pad_dims[::-1], mode="replicate"
        )  # 反转顺序适配F.pad

        # 构建动态切片索引
        slice_dims = [slice(None)] * 3  # 初始化为全切片
        if axis == 0:  # x轴（dim=2）
            if shift > 0:
                slice_dims[0] = slice(shift, None)
            else:
                slice_dims[0] = slice(None, shift)
        elif axis == 1:  # y轴（dim=1）
            if shift > 0:
                slice_dims[1] = slice(shift, None)
            else:
                slice_dims[1] = slice(None, shift)
        elif axis == 2:  # z轴（dim=0）
            if shift > 0:
                slice_dims[2] = slice(shift, None)
            else:
                slice_dims[2] = slice(None, shift)

        # 应用切片并恢复维度
        padded = padded.squeeze(0).squeeze(0)
        sliced = padded[slice_dims]
        return sliced

    # 获取各方向邻居（确保维度一致）
    left = get_neighbor(val, 1, axis=0)  # x方向
    right = get_neighbor(val, -1, axis=0)
    back = get_neighbor(val, 1, axis=1)  # y方向
    front = get_neighbor(val, -1, axis=1)
    down = get_neighbor(val, 1, axis=2)  # z方向
    up = get_neighbor(val, -1, axis=2)

    # 处理边界无效值（使用where保持维度一致）
    def safe_where(neighbor):
        return torch.where(neighbor > -9000, neighbor, val)

    left = safe_where(left)
    right = safe_where(right)
    back = safe_where(back)
    front = safe_where(front)
    down = safe_where(down)
    up = safe_where(up)

    # 计算符号一致性（转换为float32确保精度）
    sign = torch.sign(val.to(torch.float32))
    neighbors_sign = torch.stack(
        [
            torch.sign(left.to(torch.float32)),
            torch.sign(right.to(torch.float32)),
            torch.sign(back.to(torch.float32)),
            torch.sign(front.to(torch.float32)),
            torch.sign(down.to(torch.float32)),
            torch.sign(up.to(torch.float32)),
        ],
        dim=0,
    )

    # 检查所有符号是否一致
    same_sign = torch.all(neighbors_sign == sign, dim=0)

    # 生成最终掩码
    mask = (~same_sign).to(torch.int32)
    return mask * valid_mask.to(torch.int32)


def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    octree_resolution: int,
    indexing: str = "ij",
):
    length = bbox_max - bbox_min
    num_cells = octree_resolution

    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


class VanillaVolumeDecoder:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        octree_resolution: int = 384,
        enable_pbar: bool = True,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=octree_resolution,
            indexing="ij",
        )
        xyz_samples = (
            torch.from_numpy(xyz_samples)
            .to(device, dtype=dtype)
            .contiguous()
            .reshape(-1, 3)
        )

        # 2. latents to 3d volume
        batch_features = []
        for start in tqdm(
            range(0, xyz_samples.shape[0], num_chunks),
            desc=f"Volume Decoding",
            disable=not enable_pbar,
        ):
            chunk_queries = xyz_samples[start : start + num_chunks, :]
            chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
            features = geo_decoder(queries=chunk_queries, latents=latents)
            batch_features.append(features)

        grid_features = torch.cat(batch_features, dim=1)
        grid_logits, grid_features = grid_features[..., 0:1], grid_features[..., 1:]
        grid_logits = grid_logits.view((batch_size, *grid_size)).float()

        return grid_logits, xyz_samples, grid_features, None


class HierarchicalVolumeDecoder:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 65536,
        mc_level: float = 0.0,
        octree_resolution: int = 384,
        min_resolution: int = 63,
        enable_pbar: bool = True,
        empty_value: float = float("nan"),
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype

        resolutions = []
        if octree_resolution < min_resolution:
            resolutions.append(octree_resolution)
        while octree_resolution >= min_resolution:
            resolutions.append(octree_resolution)
            octree_resolution = octree_resolution // 2
        resolutions.reverse()

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=resolutions[0],
            indexing="ij",
        )

        dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=dtype)
        dilate.weight = torch.nn.Parameter(
            torch.ones(dilate.weight.shape, dtype=dtype, device=device)
        )

        grid_size = np.array(grid_size)
        xyz_samples = (
            torch.from_numpy(xyz_samples)
            .to(device, dtype=dtype)
            .contiguous()
            .reshape(-1, 3)
        )

        # 2. latents to 3d volume
        batch_features = []
        batch_size = latents.shape[0]
        for start in tqdm(
            range(0, xyz_samples.shape[0], num_chunks),
            desc=f"Hierarchical Volume Decoding [r{resolutions[0] + 1}]",
            disable=not enable_pbar,
        ):
            queries = xyz_samples[start : start + num_chunks, :]
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
            features = geo_decoder(queries=batch_queries, latents=latents)
            batch_features.append(features)

        grid_features = torch.cat(batch_features, dim=1).view(
            (batch_size, grid_size[0], grid_size[1], grid_size[2], -1)
        )
        grid_logits = grid_features[..., 0]  # assume the first element is the logits

        for octree_depth_now in resolutions[1:]:
            grid_size = np.array([octree_depth_now + 1] * 3)
            resolution = bbox_size / octree_depth_now
            next_index = torch.zeros(tuple(grid_size), dtype=dtype, device=device)
            next_logits = torch.full(
                next_index.shape, -10000.0, dtype=dtype, device=device
            )
            curr_points = extract_near_surface_volume_fn(
                grid_logits.squeeze(0), mc_level
            )
            curr_points += grid_logits.squeeze(0).abs() < 0.95

            if octree_depth_now == resolutions[-1]:
                expand_num = 0
            else:
                expand_num = 1
            for i in range(expand_num):
                curr_points = dilate(curr_points.unsqueeze(0).to(dtype)).squeeze(0)
            (cidx_x, cidx_y, cidx_z) = torch.where(curr_points > 0)
            next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
            for i in range(2 - expand_num):
                next_index = dilate(next_index.unsqueeze(0)).squeeze(0)
            nidx = torch.where(next_index > 0)

            next_points = torch.stack(nidx, dim=1)
            next_points = next_points * torch.tensor(
                resolution, dtype=latents.dtype, device=device
            ) + torch.tensor(bbox_min, dtype=latents.dtype, device=device)

            batch_features = []
            for start in tqdm(
                range(0, next_points.shape[0], num_chunks),
                desc=f"Hierarchical Volume Decoding [r{octree_depth_now + 1}]",
                disable=not enable_pbar,
            ):
                queries = next_points[start : start + num_chunks, :]
                batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
                features = geo_decoder(
                    queries=batch_queries.to(latents.dtype), latents=latents
                )
                batch_features.append(features)
            grid_features = torch.cat(batch_features, dim=1)
            grid_logits = grid_features[..., 0:1]
            next_logits[nidx] = grid_logits[0, ..., 0]
            grid_logits = next_logits.unsqueeze(0)
        grid_logits[grid_logits == -10000.0] = empty_value

        return grid_logits
