from dataclasses import dataclass
import math

import torch
import numpy as np
import random
import time
import trimesh
import fpsample
import torch.nn as nn
from einops import repeat, rearrange
from tqdm import trange
from itertools import product

import craftsman
from craftsman.models.transformers.perceiver_1d import Perceiver
from craftsman.models.transformers.attention import ResidualCrossAttentionBlock
from craftsman.utils.checkpoint import checkpoint
from craftsman.utils.base import BaseModule
from craftsman.utils.typing import *
from craftsman.utils.misc import get_world_size, get_device
from craftsman.utils.ops import generate_dense_grid_points
from craftsman.models.geometry.utils import FlexiCubes


def _batch_gather_by_idx(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather batched points/features with per-batch indices."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _kdline_fps_indices(points: torch.Tensor, n_samples: int, h: int = 5) -> torch.Tensor:
    """Run fpsample KD-line FPS per batch and return indices on the original device."""
    bs, n_points, _ = points.shape
    n_samples = min(int(n_samples), n_points)
    if n_samples >= n_points:
        return torch.arange(n_points, device=points.device).unsqueeze(0).repeat(bs, 1)

    idx_list = []
    for b in range(bs):
        pts_np = points[b, :, :3].detach().float().cpu().numpy()
        sample_idx = fpsample.bucket_fps_kdline_sampling(pts_np, n_samples, h=h)
        sample_idx = np.asarray(sample_idx, dtype=np.int64)
        idx_list.append(torch.from_numpy(sample_idx))

    return torch.stack(idx_list, dim=0).to(points.device)

###################### Utils
class FlexiCubesPlus(FlexiCubes):
    @torch.no_grad()
    def _get_case_id(self, occ_fx8, surf_cubes, res):
        return (occ_fx8[surf_cubes] * self.cube_corners_idx.to(self.device).unsqueeze(0)).sum(-1)

def generate_surface(mesh, expand_coarse, expand_fine, mesh_sample=200000, octree_depth_coarse=6, octree_depth=8, device=None):
    print("mesh_sample:", mesh_sample)
    print("octree_depth_coarse:", octree_depth_coarse)
    print("octree_depth:", octree_depth)
    print("expand_coarse:", expand_coarse)
    print("expand_fine:", expand_fine)
        
    ## sample surface points
    start = time.time()
    surface_points, _ = trimesh.sample.sample_surface(mesh=mesh, count=mesh_sample, face_weight=None, seed=666)
    surface_points = torch.as_tensor(surface_points, dtype=torch.float32, device=device)
    print(f"generate surface points time: {time.time()-start}")
    
    start = time.time()
    # Build indices on coarse
    octree_depth_delta = octree_depth - octree_depth_coarse
    print("octree_depth_delta:", octree_depth_delta)
    assert octree_depth_delta > 0
    indices = torch.floor((torch.clamp(surface_points, -1.0, 1.0) * 0.5 + 0.5) * (2 ** (octree_depth - octree_depth_delta))).to(dtype=torch.int64)
    b = (2 ** (octree_depth - octree_depth_delta))
    indices = torch.unique(indices, dim=0)
    bx, by, bz = b ** 2, b, 1
    indices_spatial = indices[:, 0] * bx + indices[:, 1] * by + indices[:, 2] * bz
    bool_grid = torch.zeros((2 ** octree_depth, 2 ** octree_depth, 2 ** octree_depth), dtype=torch.bool, device=device)
    print(f"generate voxel index and bool grid time: {time.time()-start}")
    del surface_points, indices
    
    ## expand leaf cubes
    start = time.time()
    expand_basis = list(product(*([range(-expand_coarse, expand_coarse+1)] * 3)))
    bool_grid = bool_grid.reshape(b, 2 ** octree_depth_delta, b, 2 ** octree_depth_delta, b, 2 ** octree_depth_delta).permute(0, 2, 4, 1, 3, 5).reshape(-1, (2 ** octree_depth_delta) ** 3)
    for i, basis in enumerate(expand_basis):
        bool_grid.scatter_(0, torch.clamp(indices_spatial.unsqueeze(-1) + basis[0] * bx + basis[1] * by + basis[2] * bz, 0, b ** 3 - 1), True)
    bool_grid = bool_grid.reshape(b, b, b, 2 ** octree_depth_delta, 2 ** octree_depth_delta, 2 ** octree_depth_delta).permute(0, 3, 1, 4, 2, 5).reshape((2 ** octree_depth_delta) * b, (2 ** octree_depth_delta) * b, (2 ** octree_depth_delta) * b)
    indices = torch.stack(torch.where(bool_grid), dim=-1)
    b = (2 ** octree_depth)
    bx, by, bz = b ** 2, b, 1
    indices_spatial = indices[:, 0] * bx + indices[:, 1] * by + indices[:, 2] * bz
    expand_basis = list(product(*([range(-expand_fine, expand_fine+1)] * 3)))
    for i, basis in enumerate(expand_basis):
        bool_grid.reshape(-1).scatter_(0, torch.clamp(indices_spatial + basis[0] * bx + basis[1] * by + basis[2] * bz, 0, b ** 3 - 1), True)
    print(f"expand leaf cubes time: {time.time()-start}")
    del indices_spatial
    
    ## remove reduplicated cubes and vertices
    start = time.time()
    indices = torch.stack(torch.where(bool_grid), dim=-1)
    b = (2 ** octree_depth + 1)
    bx, by, bz = b ** 2, b, 1
    indices_spatial = indices[:, 0] * bx + indices[:, 1] * by + indices[:, 2] * bz
    expand_basis = list(product(*([[0,1]] * 3)))
    indices_spatial = torch.tile(indices_spatial[:, None], (1, len(expand_basis)))
    for i, basis in enumerate(expand_basis):
        indices_spatial[:, i] = indices_spatial[:, i] + basis[0] * bx + basis[1] * by + basis[2] * bz
    indices_spatial_unique, indices_spatial_inverse = torch.unique(indices_spatial.reshape(-1), dim=0, return_inverse=True)
    indices_unique = torch.stack([
        indices_spatial_unique // bx % b,
        indices_spatial_unique // by % b,
        indices_spatial_unique // bz % b,
    ], dim=-1)
    points_octree = None
    points_octree_unique = (indices_unique.to(dtype=torch.float32) / (2 ** octree_depth)) * 2.0 - 1.0
    points_octree_unique_inverse = indices_spatial_inverse.reshape(-1, 8)
    print(f"remove reduplicated cubes and vertices time: {time.time()-start}")
    del bool_grid, indices, indices_spatial, indices_spatial_unique, indices_spatial_inverse, indices_unique
    
    return points_octree_unique, points_octree_unique_inverse

VALID_EMBED_TYPES = ["identity", "fourier", "learned_fourier", "siren"]

class FourierEmbedder(nn.Module):
    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x

class LearnedFourierEmbedder(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        per_channel_dim = half_dim // input_dim
        self.weights = nn.Parameter(torch.randn(per_channel_dim))

        self.out_dim = self.get_dims(input_dim)

    def forward(self, x):
        # [b, t, c, 1] * [1, d] = [b, t, c, d] -> [b, t, c * d]
        freqs = (x[..., None] * self.weights[None] * 2 * np.pi).view(*x.shape[:-1], -1)
        fouriered = torch.cat((x, freqs.sin(), freqs.cos()), dim=-1)
        return fouriered
    
    def get_dims(self, input_dim):
        return input_dim * (self.weights.shape[0] * 2 + 1)

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
    
class Siren(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        w0 = 1.,
        c = 6.,
        is_first = False,
        use_bias = True,
        activation = None,
        dropout = 0.
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_first = is_first

        weight = torch.zeros(out_dim, in_dim)
        bias = torch.zeros(out_dim) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)
    
    def init_(self, weight, bias, c, w0):
        dim = self.in_dim

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out
    
def get_embedder(embed_type="fourier", num_freqs=-1, input_dim=3, include_pi=True):
    if embed_type == "identity" or (embed_type == "fourier" and num_freqs == -1):
        return nn.Identity(), input_dim

    elif embed_type == "fourier":
        embedder_obj = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

    elif embed_type == "learned_fourier":
        embedder_obj = LearnedFourierEmbedder(in_channels=input_dim, dim=num_freqs)
    
    elif embed_type == "siren":
        embedder_obj = Siren(in_dim=input_dim, out_dim=num_freqs * input_dim * 2 + input_dim)

    else:
        raise ValueError(f"{embed_type} is not valid. Currently only supprts {VALID_EMBED_TYPES}")
    return embedder_obj


###################### AutoEncoder
class AutoEncoder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        num_latents: int = 256
        embed_dim: int = 64
        width: int = 768
        
    cfg: Config

    def configure(self) -> None:
        super().configure()

    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        raise NotImplementedError

    def decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def encode_kl_embed(self, latents: torch.FloatTensor, sample_posterior: bool = True):
        posterior = None
        if self.cfg.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
            if sample_posterior:
                kl_embed = posterior.sample()
            else:
                kl_embed = posterior.mode()
        else:
            kl_embed = latents
        return kl_embed, posterior
    
    def forward(self,
                surface: torch.FloatTensor,
                queries: torch.FloatTensor,
                sample_posterior: bool = True,
                sharp_surface: torch.FloatTensor = None):
        shape_latents, kl_embed, posterior = self.encode(surface, sample_posterior=sample_posterior, sharp_surface=sharp_surface)

        latents = self.decode(kl_embed) # [B, num_latents, width]

        logits = self.query(queries, latents) # [B,]

        return shape_latents, latents, posterior, logits
    
    def query(self, queries: torch.FloatTensor, latents: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError
    
    @torch.no_grad()
    def extract_geometry(self,
                         latents: torch.FloatTensor,
                         extract_mesh_func: str = "mc",
                         bounds: Union[Tuple[float], List[float], float] = (-1.05, -1.05, -1.05, 1.05, 1.05, 1.05),
                         octree_depth: int = 8,
                         num_chunks: int = 5000,
                         ):
        
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_depth=octree_depth,
            indexing="ij"
        )
        xyz_samples = torch.FloatTensor(xyz_samples)
        batch_size = latents.shape[0]

        batch_logits = []
        for start in range(0, xyz_samples.shape[0], num_chunks):
            queries = xyz_samples[start: start + num_chunks, :].to(latents)
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)

            logits = self.query(batch_queries, latents)
            batch_logits.append(logits.cpu())

        grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, grid_size[0], grid_size[1], grid_size[2])).float().numpy()

        mesh_v_f = []
        has_surface = np.zeros((batch_size,), dtype=np.bool_)
        for i in range(batch_size):
            try:
                if extract_mesh_func == "mc":
                    from skimage import measure
                    vertices, faces, normals, _ = measure.marching_cubes(grid_logits[i], 0, method="lewiner")
                    # vertices, faces = mcubes.marching_cubes(grid_logits[i], 0)
                    vertices = vertices / grid_size * bbox_size + bbox_min
                    faces = faces[:, [2, 1, 0]]
                elif extract_mesh_func == "diffmc":
                    from diso import DiffMC
                    diffmc = DiffMC(dtype=torch.float32).to(latents.device)
                    vertices, faces = diffmc(-torch.tensor(grid_logits[i]).float().to(latents.device), isovalue=0)
                    vertices = vertices * 2 - 1
                    vertices = vertices.cpu().numpy()
                    faces = faces.cpu().numpy()
                elif extract_mesh_func == "diffdmc":
                    from diso import DiffDMC
                    diffmc = DiffDMC(dtype=torch.float32).to(latents.device)
                    vertices, faces = diffmc(-torch.tensor(grid_logits[i]).float().to(latents.device), isovalue=0)
                    vertices = vertices * 2 - 1
                    vertices = vertices.cpu().numpy()
                    faces = faces.cpu().numpy()
                else:
                    raise NotImplementedError(f"{extract_mesh_func} not implement")
                mesh_v_f.append((vertices.astype(np.float32), np.ascontiguousarray(faces.astype(np.int64))))
                has_surface[i] = True
            except:
                mesh_v_f.append((None, None))
                has_surface[i] = False

        return mesh_v_f, has_surface


    @torch.no_grad()
    def extract_geometry_multi_step(self,
                         latents: torch.FloatTensor,
                         extract_mesh_func: str = "diffmc",
                         bounds: Union[Tuple[float], List[float], float] = (-1.05, -1.05, -1.05, 1.05, 1.05, 1.05),
                         dense_grid_octree_depth: int = 7,
                         expand_coarse: int = 4,
                         expand_fine: int = 12,
                         octree_depth_coarse: int = 7,
                         octree_depth: int = 9,
                         num_chunks: int = 100000,
                         ):
        
        # Stage 1: Direct Coarse Surface Extract from SDF by DISO
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        start_time_generate_dense_grid_points = time.time()
        print("dense_grid_octree_depth:", dense_grid_octree_depth)
        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_depth=dense_grid_octree_depth,
            indexing="ij"
        )
        print(f"generate_dense_grid_points time: {time.time()-start_time_generate_dense_grid_points}")
        
        xyz_samples = torch.FloatTensor(xyz_samples)
        batch_size = latents.shape[0]

        start_time_query_sdf_stage_1 = time.time()
        batch_logits = []
        for start in trange(0, xyz_samples.shape[0], num_chunks):
            queries = xyz_samples[start: start + num_chunks, :].to(latents)
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)

            logits = self.query(batch_queries, latents)
            batch_logits.append(logits.cpu())

        grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, grid_size[0], grid_size[1], grid_size[2])).float().numpy()
        print(f"query_sdf_stage_1 time: {time.time()-start_time_query_sdf_stage_1}")
        
        start_time_diso_diffmc = time.time()
        mesh_v_f = []
        has_surface = np.zeros((batch_size,), dtype=np.bool_)
        for i in range(batch_size):
            try:
                if extract_mesh_func == "diffmc":
                    from diso import DiffMC
                    diffmc = DiffMC(dtype=torch.float32).to(latents.device)
                    vertices, faces = diffmc(-torch.tensor(grid_logits[i]).float().to(latents.device), isovalue=0)
                    vertices = vertices * 2 - 1
                    vertices = vertices.cpu().numpy()
                    faces = faces.cpu().numpy()
                elif extract_mesh_func == "diffdmc":
                    from diso import DiffDMC
                    diffmc = DiffDMC(dtype=torch.float32).to(latents.device)
                    vertices, faces = diffmc(-torch.tensor(grid_logits[i]).float().to(latents.device), isovalue=0)
                    vertices = vertices * 2 - 1
                    vertices = vertices.cpu().numpy()
                    faces = faces.cpu().numpy()
                else:
                    raise NotImplementedError(f"{extract_mesh_func} not implement")
                mesh_v_f.append((vertices.astype(np.float32), np.ascontiguousarray(faces.astype(np.int64))))
                has_surface[i] = True
            except:
                mesh_v_f.append((None, None))
                has_surface[i] = False
        print(f"diso_diffmc time: {time.time()-start_time_diso_diffmc}")
        
        start_time_stage_2 = time.time()
        # Stage 2
        mesh_v_f_fine = []
        has_surface_fine = np.zeros((batch_size,), dtype=np.bool_)
        for i in range(batch_size):
            try:
                start_time_build_surface_grid = time.time()
                mesh = trimesh.Trimesh(vertices=mesh_v_f[i][0], faces=mesh_v_f[i][1], process=False)
                scale = 0.8
                mesh = mesh.apply_scale([scale, scale, scale])
                points_octree_unique, points_octree_unique_inverse = generate_surface(mesh=mesh, expand_coarse=expand_coarse, expand_fine=expand_fine, mesh_sample=1000000, octree_depth_coarse=octree_depth_coarse, octree_depth=octree_depth, device=latents.device)
                points_octree_unique = points_octree_unique / scale
                print("points_octree_unique.shape:", points_octree_unique.shape)
                print("points_octree_unique_inverse.shape:", points_octree_unique_inverse.shape)
                print(f"build_surface_grid time: {time.time()-start_time_build_surface_grid}")
                
                start_time_query_sdf_stage_2 = time.time()
                batch_logits = []
                for start in trange(0, points_octree_unique.shape[0], num_chunks):
                    queries = points_octree_unique[start: start + num_chunks, :].to(latents)
                    batch_queries = repeat(queries, "p c -> b p c", b=batch_size)

                    logits = self.query(batch_queries, latents)
                    batch_logits.append(logits)

                grid_logits = torch.cat(batch_logits, dim=1).view((-1, 1))
                print("grid_logits.shape:", grid_logits.shape)
                print(f"query_sdf_stage_2 time: {time.time()-start_time_query_sdf_stage_2}")
                
                start_time_flexcubeplus = time.time()
                points_octree_unique = points_octree_unique * scale
                fc = FlexiCubesPlus(device=latents.device)
                vertices, faces, _ = fc(points_octree_unique, grid_logits, points_octree_unique_inverse, 2 ** octree_depth)
                
                del points_octree_unique, grid_logits, points_octree_unique_inverse
                
                vertices = vertices * 2 - 1
                vertices = vertices.cpu().numpy()
                faces = faces.cpu().numpy()
                print("vertices.shape:", vertices.shape)
                print("faces.shape:", faces.shape)
                mesh_v_f_fine.append((vertices.astype(np.float32), np.ascontiguousarray(faces.astype(np.int64))))
                has_surface_fine[i] = True
                print(f"flexcubeplus time: {time.time()-start_time_flexcubeplus}")
            except:
                mesh_v_f_fine.append((None, None))
                has_surface_fine[i] = False
                
        return mesh_v_f_fine, has_surface_fine
    
    def extract_logits(self,
                         latents: torch.FloatTensor,
                         bounds: Union[Tuple[float], List[float], float] = (-1.05, -1.05, -1.05, 1.05, 1.05, 1.05),
                         octree_depth: int = 4,
                         num_chunks: int = 5000,
                         ):
        
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_depth=octree_depth,
            indexing="ij"
        )
        xyz_samples = torch.FloatTensor(xyz_samples).to(latents)
        batch_size = latents.shape[0]

        batch_logits = []
        for start in range(0, xyz_samples.shape[0], num_chunks):
            queries = xyz_samples[start: start + num_chunks, :]
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)

            logits = self.query(batch_queries, latents)
            batch_logits.append(logits)

        grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, grid_size[0], grid_size[1], grid_size[2])).float()

        return grid_logits, xyz_samples

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: Union[torch.Tensor, List[torch.Tensor]], deterministic=False, feat_dim=1):
        self.feat_dim = feat_dim
        self.parameters = parameters

        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                        + self.var - 1.0 - self.logvar,
                                        dim=dims)
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=dims)

    def nll(self, sample, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class PerceiverCrossAttentionEncoder(nn.Module):
    def __init__(self,
                 use_downsample: bool,
                 num_latents: int,
                 embedder: FourierEmbedder,
                 point_feats: int,
                 embed_point_feats: bool,
                 width: int,
                 heads: int,
                 layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 qk_norm: bool = True,
                 use_ln_post: bool = False,
                 use_flash: bool = False,
                 use_checkpoint: bool = False,
                 use_multi_reso: bool = False,
                 resolutions: list = [],
                 sampling_prob: list = [],
                 with_sharp_data: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.use_downsample = use_downsample
        self.embed_point_feats = embed_point_feats
        self.use_multi_reso = use_multi_reso
        self.resolutions = resolutions
        self.sampling_prob = sampling_prob

        if not self.use_downsample:
            self.query = nn.Parameter(torch.randn((num_latents, width)) * 0.02)

        self.embedder = embedder
        if self.embed_point_feats:
            self.input_proj = nn.Linear(self.embedder.out_dim * 2, width)
        else:
            self.input_proj = nn.Linear(self.embedder.out_dim + point_feats, width)

        self.cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            use_flash=use_flash,
        )

        self.with_sharp_data = with_sharp_data
        if with_sharp_data:
            self.downsmaple_num_latents = num_latents // 2
            self.input_proj_sharp = nn.Linear(self.embedder.out_dim + point_feats, width)
            self.cross_attn_sharp = ResidualCrossAttentionBlock(  #给sharp 数据的cross attn
                width=width,
                heads=heads,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                use_flash=use_flash
            )
        else:
            self.downsmaple_num_latents = num_latents

        self.self_attn = Perceiver(
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            use_flash=use_flash,
            use_checkpoint=False
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width)
        else:
            self.ln_post = None

    def _forward(self, pc, feats, sharp_pc = None, sharp_feat = None):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:

        """

        bs, N, D = pc.shape
        
        data = self.embedder(pc)
        if feats is not None:
            if self.embed_point_feats:
                feats = self.embedder(feats)
            data = torch.cat([data, feats], dim=-1)
        data = self.input_proj(data)

        if self.with_sharp_data and sharp_pc is not None:
            sharp_data = self.embedder(sharp_pc)
            if sharp_feat is not None:
                if self.embed_point_feats:
                    sharp_feat = self.embedder(sharp_feat)
                sharp_data = torch.cat([sharp_data, sharp_feat], dim=-1)
            sharp_data = self.input_proj_sharp(sharp_data)

        if self.use_multi_reso:
            resolution = random.choice(self.resolutions, size=1, p=self.sampling_prob)[0]

            if resolution != N:
                idx = _kdline_fps_indices(pc, resolution, h=5)
                pc = _batch_gather_by_idx(pc, idx)
                if feats is not None:
                    feats = _batch_gather_by_idx(feats, idx)
                bs, N, D = pc.shape

        if self.use_downsample:
            idx = _kdline_fps_indices(pc, self.downsmaple_num_latents, h=5)
            query = _batch_gather_by_idx(data, idx)

            if self.with_sharp_data and sharp_pc is not None:
                idx = _kdline_fps_indices(sharp_pc, self.downsmaple_num_latents, h=5)
                sharp_query = _batch_gather_by_idx(sharp_data, idx)

                query = torch.cat([query, sharp_query], dim=1)
        else:
            query = self.query
            query = repeat(query, "m c -> b m c", b=bs)

        latents = self.cross_attn(query, data)
        if self.with_sharp_data and sharp_pc is not None:
            latents = latents + self.cross_attn_sharp(query, sharp_data)
        latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents

    def forward(self, pc: torch.FloatTensor, 
                feats: Optional[torch.FloatTensor] = None, 
                sharp_pc: Optional[torch.FloatTensor] = None,
                sharp_feats: Optional[torch.FloatTensor] = None):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:
            dict
        """

        return checkpoint(self._forward, (pc, feats, sharp_pc, sharp_feats), self.parameters(), self.use_checkpoint)


class PerceiverCrossAttentionDecoder(nn.Module):

    def __init__(self,
                 num_latents: int,
                 out_dim: int,
                 embedder: FourierEmbedder,
                 width: int,
                 heads: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 qk_norm: bool = True,
                 use_flash: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.embedder = embedder

        self.query_proj = nn.Linear(self.embedder.out_dim, width)

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            use_flash=use_flash
        )

        self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_dim)

    def _forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        queries = self.query_proj(self.embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        return checkpoint(self._forward, (queries, latents), self.parameters(), self.use_checkpoint)


@craftsman.register("michelangelo-autoencoder")
class MichelangeloAutoencoder(AutoEncoder):
    r"""
    A VAE model for encoding shapes into latents and decoding latent representations into shapes.
    """

    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        n_samples: int = 4096
        n_samples_pcd: int = 4096
        use_downsample: bool = False
        downsample_ratio: float = 0.0625
        num_latents: int = 256
        point_feats: int = 0
        embed_point_feats: bool = False
        out_dim: int = 1
        embed_dim: int = 64
        embed_type: str = "fourier"
        num_freqs: int = 8
        include_pi: bool = True
        width: int = 768
        heads: int = 12
        num_encoder_layers: int = 8
        num_decoder_layers: int = 16
        init_scale: float = 0.25
        qkv_bias: bool = True
        qk_norm: bool = False
        use_ln_post: bool = False
        use_flash: bool = False
        use_checkpoint: bool = True
        use_multi_reso: Optional[bool] = False
        resolutions: Optional[List[int]] = None
        sampling_prob: Optional[List[float]] = None
        with_sharp_data: Optional[bool] = True

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.embedder = get_embedder(embed_type=self.cfg.embed_type, num_freqs=self.cfg.num_freqs, include_pi=self.cfg.include_pi)

        # encoder
        self.cfg.init_scale = self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width)
        self.encoder = PerceiverCrossAttentionEncoder(
            use_downsample=self.cfg.use_downsample,
            embedder=self.embedder,
            num_latents=self.cfg.num_latents,
            point_feats=self.cfg.point_feats,
            embed_point_feats=self.cfg.embed_point_feats,
            width=self.cfg.width,
            heads=self.cfg.heads,
            layers=self.cfg.num_encoder_layers,
            init_scale=self.cfg.init_scale,
            qkv_bias=self.cfg.qkv_bias,
            qk_norm=self.cfg.qk_norm,
            use_ln_post=self.cfg.use_ln_post,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint,
            use_multi_reso=self.cfg.use_multi_reso,
            resolutions=self.cfg.resolutions,
            sampling_prob=self.cfg.sampling_prob,
            with_sharp_data=self.cfg.with_sharp_data
        )

        if self.cfg.embed_dim > 0:
            # VAE embed
            self.pre_kl = nn.Linear(self.cfg.width, self.cfg.embed_dim * 2)
            self.post_kl = nn.Linear(self.cfg.embed_dim, self.cfg.width)
            self.latent_shape = (self.cfg.num_latents, self.cfg.embed_dim)
        else:
            self.latent_shape = (self.cfg.num_latents, self.cfg.width)

        self.transformer = Perceiver(
            n_ctx=self.cfg.num_latents,
            width=self.cfg.width,
            layers=self.cfg.num_decoder_layers,
            heads=self.cfg.heads,
            init_scale=self.cfg.init_scale,
            qkv_bias=self.cfg.qkv_bias,
            qk_norm=self.cfg.qk_norm,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint
        )

        # decoder
        self.decoder = PerceiverCrossAttentionDecoder(
            embedder=self.embedder,
            out_dim=self.cfg.out_dim,
            num_latents=self.cfg.num_latents,
            width=self.cfg.width,
            heads=self.cfg.heads,
            init_scale=self.cfg.init_scale,
            qkv_bias=self.cfg.qkv_bias,
            qk_norm=self.cfg.qk_norm,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint
        )

        if self.cfg.pretrained_model_name_or_path != "":
            print(f"Loading pretrained VAE model from {self.cfg.pretrained_model_name_or_path}")
            pretrained_ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu")
            if 'state_dict' in pretrained_ckpt:
                _pretrained_ckpt = {}
                for k, v in pretrained_ckpt['state_dict'].items():
                    if k.startswith('shape_model.'):
                        _pretrained_ckpt[k.replace('shape_model.', '')] = v
                pretrained_ckpt = _pretrained_ckpt
                self.load_state_dict(pretrained_ckpt, strict=True)
            else:
                if "hunyuan" in self.cfg.pretrained_model_name_or_path:
                    _pretrained_ckpt = {}
                    for k, v in pretrained_ckpt['vae'].items():
                        if k.startswith('geo_decoder.'):
                            _pretrained_ckpt[k.replace('geo_decoder.', 'decoder.')] = v
                        else:
                            _pretrained_ckpt[k] = v
                    pretrained_ckpt = _pretrained_ckpt
                    self.load_state_dict(pretrained_ckpt, strict=False)
                else:
                    _pretrained_ckpt = {}
                    for k, v in pretrained_ckpt.items():
                        if k.startswith('shape_model.'):
                            _pretrained_ckpt[k.replace('shape_model.', '')] = v
                    pretrained_ckpt = _pretrained_ckpt
                    self.load_state_dict(pretrained_ckpt, strict=True)
            
    
    def encode(self,
               surface: torch.FloatTensor,
               sample_posterior: bool = True,
               sharp_surface: torch.FloatTensor = None):
        """
        Args:
            surface (torch.FloatTensor): [B, N, 3+C]
            sample_posterior (bool):

        Returns:
            shape_latents (torch.FloatTensor): [B, num_latents, width]
            kl_embed (torch.FloatTensor): [B, num_latents, embed_dim]
            posterior (DiagonalGaussianDistribution or None):
        """
        assert surface.shape[-1] == 3 + self.cfg.point_feats, f"\
            Expected {3 + self.cfg.point_feats} channels, got {surface.shape[-1]}"
        
        pc, feats = surface[..., :3], surface[..., 3:] # B, n_samples, 3
        bs, N, D = pc.shape
        if N > self.cfg.n_samples:
            # idx = furthest_point_sample(pc, self.cfg.n_samples) # (B, 3, npoint)
            # pc = gather_operation(pc, idx).transpose(2, 1).contiguous()
            # feats = gather_operation(feats, idx).transpose(2, 1).contiguous()
            idx = _kdline_fps_indices(pc, self.cfg.n_samples, h=5)
            pc = _batch_gather_by_idx(pc, idx)
            feats = _batch_gather_by_idx(feats, idx)

        if sharp_surface is not None:
            sharp_pc, sharp_feats = sharp_surface[..., :3], sharp_surface[..., 3:] # B, n_samples, 3
            bs, N, D = sharp_pc.shape
            if N > self.cfg.n_samples:
                idx = _kdline_fps_indices(sharp_pc, self.cfg.n_samples, h=5)
                sharp_pc = _batch_gather_by_idx(sharp_pc, idx)
                sharp_feats = _batch_gather_by_idx(sharp_feats, idx)
        else:
            sharp_pc, sharp_feats = None, None

        shape_latents = self.encoder(pc, feats, sharp_pc, sharp_feats) # B, num_latents, width
        kl_embed, posterior = self.encode_kl_embed(shape_latents, sample_posterior)  # B, num_latents, embed_dim

        return shape_latents, kl_embed, posterior


    def decode(self, 
               latents: torch.FloatTensor):
        """
        Args:
            latents (torch.FloatTensor): [B, embed_dim]

        Returns:
            latents (torch.FloatTensor): [B, embed_dim]
        """
        latents = self.post_kl(latents) # [B, num_latents, embed_dim] -> [B, num_latents, width]

        return self.transformer(latents)


    def query(self, 
              queries: torch.FloatTensor, 
              latents: torch.FloatTensor):
        """
        Args:
            queries (torch.FloatTensor): [B, N, 3]
            latents (torch.FloatTensor): [B, embed_dim]

        Returns:
            logits (torch.FloatTensor): [B, N], occupancy logits
        """

        logits = self.decoder(queries, latents).squeeze(-1)

        return logits
    
    def encode_pcd(self,
               surface: torch.FloatTensor,
               sample_posterior: bool = True,
               sharp_surface: torch.FloatTensor = None):
        """
        Args:
            surface (torch.FloatTensor): [B, N, 3]
            sample_posterior (bool):

        Returns:
            shape_latents (torch.FloatTensor): [B, num_latents, width]
            kl_embed (torch.FloatTensor): [B, num_latents, embed_dim]
            posterior (DiagonalGaussianDistribution or None):
        """
        
        assert surface.shape[-1] == 3 + self.cfg.point_feats, f"\
            Expected {3 + self.cfg.point_feats} channels, got {surface.shape[-1]}"
        
        pc, feats = surface[..., :3], surface[..., 3:] # B, n_samples, 3
        bs, N, D = pc.shape
        if N > self.cfg.n_samples_pcd:
            # idx = furthest_point_sample(pc, self.cfg.n_samples_pcd) # (B, 3, npoint)
            # pc = gather_operation(pc, idx).transpose(2, 1).contiguous()
            # feats = gather_operation(feats, idx).transpose(2, 1).contiguous()
            idx = _kdline_fps_indices(pc, self.cfg.n_samples_pcd, h=5)
            pc = _batch_gather_by_idx(pc, idx)
            feats = _batch_gather_by_idx(feats, idx)

        if sharp_surface is not None:
            sharp_pc, sharp_feats = sharp_surface[..., :3], sharp_surface[..., 3:] # B, n_samples_pcd, 3
            bs, N, D = sharp_pc.shape
            if N > self.cfg.n_samples_pcd:
                idx = _kdline_fps_indices(sharp_pc, self.cfg.n_samples_pcd, h=5)
                sharp_pc = _batch_gather_by_idx(sharp_pc, idx)
                sharp_feats = _batch_gather_by_idx(sharp_feats, idx)
        else:
            sharp_pc, sharp_feats = None, None

        shape_latents = self.encoder(pc, feats, sharp_pc, sharp_feats) # B, num_latents, width
        kl_embed, posterior = self.encode_kl_embed(shape_latents, sample_posterior)  # B, num_latents, embed_dim

        return shape_latents, kl_embed, posterior
