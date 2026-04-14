import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from torch import BoolTensor, FloatTensor

from . import logging
from .camera import Camera
import xatlas

logger = logging.get_logger(__name__)


def dot(x: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
    return torch.sum(x * y, -1, keepdim=True)


@dataclass
class TexturedMesh:
    v_pos: torch.FloatTensor
    t_pos_idx: torch.LongTensor

    # texture coordinates
    v_tex: Optional[torch.FloatTensor] = None
    t_tex_idx: Optional[torch.LongTensor] = None

    # texture map
    texture: Optional[torch.FloatTensor] = None

    # vertices, faces after vertex merging
    _stitched_v_pos: Optional[torch.FloatTensor] = None
    _stitched_t_pos_idx: Optional[torch.LongTensor] = None

    _v_nrm: Optional[torch.FloatTensor] = None

    @property
    def v_nrm(self) -> torch.FloatTensor:
        if self._v_nrm is None:
            self._v_nrm = self._compute_vertex_normal()
        return self._v_nrm

    def set_stitched_mesh(
        self, v_pos: torch.FloatTensor, t_pos_idx: torch.LongTensor
    ) -> None:
        self._stitched_v_pos = v_pos
        self._stitched_t_pos_idx = t_pos_idx

    @property
    def stitched_v_pos(self) -> torch.FloatTensor:
        if self._stitched_v_pos is None:
            logger.warning("Stitched vertices not available, using original vertices!")
            return self.v_pos
        return self._stitched_v_pos

    @property
    def stitched_t_pos_idx(self) -> torch.LongTensor:
        if self._stitched_t_pos_idx is None:
            logger.warning("Stitched faces not available, using original faces!")
            return self.t_pos_idx
        return self._stitched_t_pos_idx

    def _compute_vertex_normal(self) -> torch.FloatTensor:
        if self._stitched_v_pos is None or self._stitched_t_pos_idx is None:
            logger.warning(
                "Stitched vertices and faces not available, computing vertex normals on original mesh, which can be erroneous!"
            )
            v_pos, t_pos_idx = self.v_pos, self.t_pos_idx
        else:
            v_pos, t_pos_idx = self._stitched_v_pos, self._stitched_t_pos_idx

        i0 = t_pos_idx[:, 0]
        i1 = t_pos_idx[:, 1]
        i2 = t_pos_idx[:, 2]

        v0 = v_pos[i0, :]
        v1 = v_pos[i1, :]
        v2 = v_pos[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(v_pos)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = torch.where(
            dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
        )
        v_nrm = F.normalize(v_nrm, dim=1)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_nrm))

        return v_nrm

    def to(self, device: Optional[str] = None):
        self.v_pos = self.v_pos.to(device)
        self.t_pos_idx = self.t_pos_idx.to(device)
        if self.v_tex is not None:
            self.v_tex = self.v_tex.to(device)
        if self.t_tex_idx is not None:
            self.t_tex_idx = self.t_tex_idx.to(device)
        if self.texture is not None:
            self.texture = self.texture.to(device)
        if self._stitched_v_pos is not None:
            self._stitched_v_pos = self._stitched_v_pos.to(device)
        if self._stitched_t_pos_idx is not None:
            self._stitched_t_pos_idx = self._stitched_t_pos_idx.to(device)
        if self._v_nrm is not None:
            self._v_nrm = self._v_nrm.to(device)


def mesh_uv_wrap(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if len(mesh.faces) > 500000000:
        raise ValueError(
            "The mesh has more than 500,000,000 faces, which is not supported."
        )

    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)

    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs

    return mesh


def load_mesh(
    mesh: str,
    rescale: bool = False,
    move_to_center: bool = False,
    scale: float = 0.5,
    flip_uv: bool = True,
    merge_vertices: bool = True,
    default_uv_size: int = 2048,
    shape_init_mesh_up: str = "+y",
    shape_init_mesh_front: str = "+x",
    front_x_to_y: bool = False,
    device: Optional[str] = None,
    return_transform: bool = False,
) -> TexturedMesh:
    scene = mesh
    # scene = trimesh.load(mesh, force="mesh", process=False)
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    elif isinstance(scene, trimesh.scene.Scene):
        mesh = trimesh.Trimesh()
        for obj in scene.geometry.values():
            mesh = trimesh.util.concatenate([mesh, obj])
    else:
        raise ValueError(f"Unknown mesh type at {mesh_path}.")

    # move to center
    if move_to_center:
        centroid = mesh.vertices.mean(0)
        mesh.vertices = mesh.vertices - centroid
    mesh = mesh_uv_wrap(mesh)
    # rescale
    if rescale:
        max_scale = np.abs(mesh.vertices).max()
        mesh.vertices = mesh.vertices / max_scale * scale

    mesh_bp = trimesh.base.Trimesh.copy(mesh)
    dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
    dir2vec = {
        "+x": np.array([1, 0, 0]),
        "+y": np.array([0, 1, 0]),
        "+z": np.array([0, 0, 1]),
        "-x": np.array([-1, 0, 0]),
        "-y": np.array([0, -1, 0]),
        "-z": np.array([0, 0, -1]),
    }
    if shape_init_mesh_up not in dirs or shape_init_mesh_front not in dirs:
        raise ValueError(
            f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}."
        )
    if shape_init_mesh_up[1] == shape_init_mesh_front[1]:
        raise ValueError(
            "shape_init_mesh_up and shape_init_mesh_front must be orthogonal."
        )
    z_, x_ = (
        dir2vec[shape_init_mesh_up],
        dir2vec[shape_init_mesh_front],
    )
    y_ = np.cross(z_, x_)
    std2mesh = np.stack([x_, y_, z_], axis=0).T
    mesh2std = np.linalg.inv(std2mesh)
    mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T
    if front_x_to_y:
        x = mesh.vertices[:, 1].copy()
        y = -mesh.vertices[:, 0].copy()
        mesh.vertices[:, 0] = x
        mesh.vertices[:, 1] = y

    v_pos = torch.tensor(mesh.vertices, dtype=torch.float32)
    t_pos_idx = torch.tensor(mesh.faces, dtype=torch.int64)

    if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv"):
        v_tex = torch.tensor(mesh.visual.uv, dtype=torch.float32)
        if flip_uv:
            v_tex[:, 1] = 1.0 - v_tex[:, 1]
        t_tex_idx = t_pos_idx.clone()
        # if (
        #     hasattr(mesh.visual.material, "baseColorTexture")
        #     and mesh.visual.material.baseColorTexture
        # ):
        #     texture = torch.tensor(
        #         np.array(mesh.visual.material.baseColorTexture) / 255.0,
        #         dtype=torch.float32,
        #     )[..., :3]
        # else:
        texture = torch.zeros(
            (default_uv_size, default_uv_size, 3), dtype=torch.float32
        )
    else:
        v_tex = None
        t_tex_idx = None
        texture = None

    textured_mesh = TexturedMesh(
        v_pos=v_pos,
        t_pos_idx=t_pos_idx,
        v_tex=v_tex,
        t_tex_idx=t_tex_idx,
        texture=texture,
    )

    if merge_vertices:
        mesh.merge_vertices(merge_tex=True)
        textured_mesh.set_stitched_mesh(
            torch.tensor(mesh.vertices, dtype=torch.float32),
            torch.tensor(mesh.faces, dtype=torch.int64),
        )

    textured_mesh.to(device)

    if return_transform:
        return textured_mesh, np.array(centroid), max_scale / scale

    return textured_mesh, mesh_bp


@dataclass
class RenderOutput:
    attr: Optional[torch.FloatTensor] = None
    mask: Optional[torch.BoolTensor] = None
    depth: Optional[torch.FloatTensor] = None
    normal: Optional[torch.FloatTensor] = None
    pos: Optional[torch.FloatTensor] = None


class NVDiffRastContextWrapper:
    def __init__(self, device: str, context_type: str = "gl"):
        if context_type == "gl":
            self.ctx = dr.RasterizeGLContext(device=device)
        elif context_type == "cuda":
            self.ctx = dr.RasterizeCudaContext(device=device)
        else:
            raise NotImplementedError

    def rasterize(self, pos, tri, resolution, ranges=None, grad_db=True):
        """
        Rasterize triangles.

        All input tensors must be contiguous and reside in GPU memory except for the ranges tensor that, if specified, has to reside in CPU memory. The output tensors will be contiguous and reside in GPU memory.

        Arguments:
        glctx	Rasterizer context of type RasterizeGLContext or RasterizeCudaContext.
        pos	Vertex position tensor with dtype torch.float32. To enable range mode, this tensor should have a 2D shape [num_vertices, 4]. To enable instanced mode, use a 3D shape [minibatch_size, num_vertices, 4].
        tri	Triangle tensor with shape [num_triangles, 3] and dtype torch.int32.
        resolution	Output resolution as integer tuple (height, width).
        ranges	In range mode, tensor with shape [minibatch_size, 2] and dtype torch.int32, specifying start indices and counts into tri. Ignored in instanced mode.
        grad_db	Propagate gradients of image-space derivatives of barycentrics into pos in backward pass. Ignored if using an OpenGL context that was not configured to output image-space derivatives.
        Returns:
        A tuple of two tensors. The first output tensor has shape [minibatch_size, height, width, 4] and contains the main rasterizer output in order (u, v, z/w, triangle_id). If the OpenGL context was configured to output image-space derivatives of barycentrics, the second output tensor will also have shape [minibatch_size, height, width, 4] and contain said derivatives in order (du/dX, du/dY, dv/dX, dv/dY). Otherwise it will be an empty tensor with shape [minibatch_size, height, width, 0].
        """
        return dr.rasterize(
            self.ctx, pos.float(), tri.int(), resolution, ranges, grad_db
        )

    def interpolate(self, attr, rast, tri, rast_db=None, diff_attrs=None):
        """
        Interpolate vertex attributes.

        All input tensors must be contiguous and reside in GPU memory. The output tensors will be contiguous and reside in GPU memory.

        Arguments:
        attr	Attribute tensor with dtype torch.float32. Shape is [num_vertices, num_attributes] in range mode, or [minibatch_size, num_vertices, num_attributes] in instanced mode. Broadcasting is supported along the minibatch axis.
        rast	Main output tensor from rasterize().
        tri	Triangle tensor with shape [num_triangles, 3] and dtype torch.int32.
        rast_db	(Optional) Tensor containing image-space derivatives of barycentrics, i.e., the second output tensor from rasterize(). Enables computing image-space derivatives of attributes.
        diff_attrs	(Optional) List of attribute indices for which image-space derivatives are to be computed. Special value 'all' is equivalent to list [0, 1, ..., num_attributes - 1].
        Returns:
        A tuple of two tensors. The first output tensor contains interpolated attributes and has shape [minibatch_size, height, width, num_attributes]. If rast_db and diff_attrs were specified, the second output tensor contains the image-space derivatives of the selected attributes and has shape [minibatch_size, height, width, 2 * len(diff_attrs)]. The derivatives of the first selected attribute A will be on channels 0 and 1 as (dA/dX, dA/dY), etc. Otherwise, the second output tensor will be an empty tensor with shape [minibatch_size, height, width, 0].
        """
        return dr.interpolate(attr.float(), rast, tri.int(), rast_db, diff_attrs)

    def texture(
        self,
        tex,
        uv,
        uv_da=None,
        mip_level_bias=None,
        mip=None,
        filter_mode="auto",
        boundary_mode="wrap",
        max_mip_level=None,
    ):
        """
        Perform texture sampling.

        All input tensors must be contiguous and reside in GPU memory. The output tensor will be contiguous and reside in GPU memory.

        Arguments:
        tex	Texture tensor with dtype torch.float32. For 2D textures, must have shape [minibatch_size, tex_height, tex_width, tex_channels]. For cube map textures, must have shape [minibatch_size, 6, tex_height, tex_width, tex_channels] where tex_width and tex_height are equal. Note that boundary_mode must also be set to 'cube' to enable cube map mode. Broadcasting is supported along the minibatch axis.
        uv	Tensor containing per-pixel texture coordinates. When sampling a 2D texture, must have shape [minibatch_size, height, width, 2]. When sampling a cube map texture, must have shape [minibatch_size, height, width, 3].
        uv_da	(Optional) Tensor containing image-space derivatives of texture coordinates. Must have same shape as uv except for the last dimension that is to be twice as long.
        mip_level_bias	(Optional) Per-pixel bias for mip level selection. If uv_da is omitted, determines mip level directly. Must have shape [minibatch_size, height, width].
        mip	(Optional) Preconstructed mipmap stack from a texture_construct_mip() call, or a list of tensors specifying a custom mipmap stack. When specifying a custom mipmap stack, the tensors in the list must follow the same format as tex except for width and height that must follow the usual rules for mipmap sizes. The base level texture is still supplied in tex and must not be included in the list. Gradients of a custom mipmap stack are not automatically propagated to base texture but the mipmap tensors will receive gradients of their own. If a mipmap stack is not specified but the chosen filter mode requires it, the mipmap stack is constructed internally and discarded afterwards.
        filter_mode	Texture filtering mode to be used. Valid values are 'auto', 'nearest', 'linear', 'linear-mipmap-nearest', and 'linear-mipmap-linear'. Mode 'auto' selects 'linear' if neither uv_da or mip_level_bias is specified, and 'linear-mipmap-linear' when at least one of them is specified, these being the highest-quality modes possible depending on the availability of the image-space derivatives of the texture coordinates or direct mip level information.
        boundary_mode	Valid values are 'wrap', 'clamp', 'zero', and 'cube'. If tex defines a cube map, this must be set to 'cube'. The default mode 'wrap' takes fractional part of texture coordinates. Mode 'clamp' clamps texture coordinates to the centers of the boundary texels. Mode 'zero' virtually extends the texture with all-zero values in all directions.
        max_mip_level	If specified, limits the number of mipmaps constructed and used in mipmap-based filter modes.
        Returns:
        A tensor containing the results of the texture sampling with shape [minibatch_size, height, width, tex_channels]. Cube map fetches with invalid uv coordinates (e.g., zero vectors) output all zeros and do not propagate gradients.
        """
        return dr.texture(
            tex.float(),
            uv.float(),
            uv_da,
            mip_level_bias,
            mip,
            filter_mode,
            boundary_mode,
            max_mip_level,
        )

    def antialias(
        self, color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0
    ):
        """
        Perform antialiasing.

        All input tensors must be contiguous and reside in GPU memory. The output tensor will be contiguous and reside in GPU memory.

        Note that silhouette edge determination is based on vertex indices in the triangle tensor. For it to work properly, a vertex belonging to multiple triangles must be referred to using the same vertex index in each triangle. Otherwise, nvdiffrast will always classify the adjacent edges as silhouette edges, which leads to bad performance and potentially incorrect gradients. If you are unsure whether your data is good, check which pixels are modified by the antialias operation and compare to the example in the documentation.

        Arguments:
        color	Input image to antialias with shape [minibatch_size, height, width, num_channels].
        rast	Main output tensor from rasterize().
        pos	Vertex position tensor used in the rasterization operation.
        tri	Triangle tensor used in the rasterization operation.
        topology_hash	(Optional) Preconstructed topology hash for the triangle tensor. If not specified, the topology hash is constructed internally and discarded afterwards.
        pos_gradient_boost	(Optional) Multiplier for gradients propagated to pos.
        Returns:
        A tensor containing the antialiased image with the same shape as color input tensor.
        """
        return dr.antialias(
            color.float(),
            rast,
            pos.float(),
            tri.int(),
            topology_hash,
            pos_gradient_boost,
        )


def get_clip_space_position(pos: torch.FloatTensor, mvp_mtx: torch.FloatTensor):
    pos_homo = torch.cat([pos, torch.ones([pos.shape[0], 1]).to(pos)], dim=-1)
    return torch.matmul(pos_homo, mvp_mtx.permute(0, 2, 1))


def transform_points_homo(pos: torch.FloatTensor, mtx: torch.FloatTensor):
    batch_size = pos.shape[0]
    pos_shape = pos.shape[1:-1]
    pos = pos.reshape(batch_size, -1, 3)
    pos_homo = torch.cat([pos, torch.ones_like(pos[..., 0:1])], dim=-1)
    pos = (pos_homo.unsqueeze(2) * mtx.unsqueeze(1)).sum(-1)[..., :3]
    pos = pos.reshape(batch_size, *pos_shape, 3)
    return pos


class DepthNormalizationStrategy(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(
        self, depth: torch.FloatTensor, mask: torch.BoolTensor
    ) -> torch.FloatTensor:
        pass


class DepthControlNetNormalization(DepthNormalizationStrategy):
    def __init__(
        self, far_clip: float = 0.25, near_clip: float = 1.0, bg_value: float = 0.0
    ):
        self.far_clip = far_clip
        self.near_clip = near_clip
        self.bg_value = bg_value

    def __call__(
        self, depth: torch.FloatTensor, mask: torch.BoolTensor
    ) -> torch.FloatTensor:
        batch_size = depth.shape[0]
        min_depth = depth.view(batch_size, -1).min(dim=-1)[0][:, None, None]
        max_depth = depth.view(batch_size, -1).max(dim=-1)[0][:, None, None]
        depth = 1.0 - ((depth - min_depth) / (max_depth - min_depth + 1e-5)).clamp(
            0.0, 1.0
        )
        depth = depth * (self.near_clip - self.far_clip) + self.far_clip
        depth[~mask] = self.bg_value
        return depth


class Zero123PlusPlusNormalization(DepthNormalizationStrategy):
    def __init__(self, bg_value: float = 0.8):
        self.bg_value = bg_value

    def __call__(self, depth: FloatTensor, mask: BoolTensor) -> FloatTensor:
        batch_size = depth.shape[0]
        min_depth = depth.view(batch_size, -1).min(dim=-1)[0][:, None, None]
        max_depth = depth.view(batch_size, -1).max(dim=-1)[0][:, None, None]
        depth = ((depth - min_depth) / (max_depth - min_depth + 1e-5)).clamp(0.0, 1.0)
        depth[~mask] = self.bg_value
        return depth


class SimpleNormalization(DepthNormalizationStrategy):
    def __init__(
        self,
        scale: float = 1.0,
        offset: float = -1.0,
        clamp: bool = True,
        bg_value: float = 1.0,
    ):
        self.scale = scale
        self.offset = offset
        self.clamp = clamp
        self.bg_value = bg_value

    def __call__(self, depth: FloatTensor, mask: BoolTensor) -> FloatTensor:
        depth = depth * self.scale + self.offset
        if self.clamp:
            depth = depth.clamp(0.0, 1.0)
        depth[~mask] = self.bg_value
        return depth


def render(
    ctx: NVDiffRastContextWrapper,
    mesh: TexturedMesh,
    cam: Camera,
    height: int,
    width: int,
    render_attr: bool = True,
    render_depth: bool = True,
    render_normal: bool = True,
    depth_normalization_strategy: DepthNormalizationStrategy = DepthControlNetNormalization(),
    attr_background: Union[float, torch.FloatTensor] = 0.5,
    antialias_attr=False,
    normal_background: Union[float, torch.FloatTensor] = 0.5,
    texture_override=None,
    texture_filter_mode: str = "linear",
) -> RenderOutput:
    output_dict = {}

    v_pos_clip = get_clip_space_position(mesh.v_pos, cam.mvp_mtx)
    rast, _ = ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width), grad_db=True)
    mask = rast[..., 3] > 0

    gb_pos, _ = ctx.interpolate(mesh.v_pos[None], rast, mesh.t_pos_idx)
    output_dict.update({"mask": mask, "pos": gb_pos})

    if render_depth:
        gb_pos_vs = transform_points_homo(gb_pos, cam.w2c)
        gb_depth = -gb_pos_vs[..., 2]
        # set background pixels to min depth value for correct min/max calculation
        gb_depth = torch.where(
            mask,
            gb_depth,
            gb_depth.view(gb_depth.shape[0], -1).min(dim=-1)[0][:, None, None],
        )
        gb_depth = depth_normalization_strategy(gb_depth, mask)
        output_dict["depth"] = gb_depth

    if render_attr:
        tex_c, _ = ctx.interpolate(mesh.v_tex[None], rast, mesh.t_tex_idx)
        texture = (
            texture_override[None]
            if texture_override is not None
            else mesh.texture[None]
        )
        gb_rgb_fg = ctx.texture(texture, tex_c, filter_mode=texture_filter_mode)
        gb_rgb_bg = torch.ones_like(gb_rgb_fg) * attr_background
        gb_rgb = torch.where(mask[..., None], gb_rgb_fg, gb_rgb_bg)
        if antialias_attr:
            gb_rgb = ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)
        output_dict["attr"] = gb_rgb

    if render_normal:
        gb_nrm, _ = ctx.interpolate(mesh.v_nrm[None], rast, mesh.stitched_t_pos_idx)
        gb_nrm = F.normalize(gb_nrm, dim=-1, p=2)
        gb_nrm[~mask] = normal_background
        output_dict["normal"] = gb_nrm

    return RenderOutput(**output_dict)
