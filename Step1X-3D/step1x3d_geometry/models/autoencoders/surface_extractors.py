from typing import Union, Tuple, List

import numpy as np
import torch
from skimage import measure


class MeshExtractResult:
    def __init__(self, verts, faces, vertex_attrs=None, res=64):
        self.verts = verts
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.face_normal = self.comput_face_normals()
        self.vert_normal = self.comput_v_normals()
        self.res = res
        self.success = verts.shape[0] != 0 and faces.shape[0] != 0

        # training only
        self.tsdf_v = None
        self.tsdf_s = None
        self.reg_loss = None

    def comput_face_normals(self):
        i0 = self.faces[..., 0].long()
        i1 = self.faces[..., 1].long()
        i2 = self.faces[..., 2].long()

        v0 = self.verts[i0, :]
        v1 = self.verts[i1, :]
        v2 = self.verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        return face_normals[:, None, :].repeat(1, 3, 1)

    def comput_v_normals(self):
        i0 = self.faces[..., 0].long()
        i1 = self.faces[..., 1].long()
        i2 = self.faces[..., 2].long()

        v0 = self.verts[i0, :]
        v1 = self.verts[i1, :]
        v2 = self.verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(self.verts)
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        return v_normals


def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(dim=0)[0]
    vert_max = vertices.max(dim=0)[0]
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center


class SurfaceExtractor:
    def _compute_box_stat(
        self, bounds: Union[Tuple[float], List[float], float], octree_resolution: int
    ):
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        grid_size = [
            int(octree_resolution) + 1,
            int(octree_resolution) + 1,
            int(octree_resolution) + 1,
        ]
        return grid_size, bbox_min, bbox_size

    def run(self, *args, **kwargs):
        return NotImplementedError

    def __call__(self, grid_logits, **kwargs):
        outputs = []
        for i in range(grid_logits.shape[0]):
            try:
                verts, faces = self.run(grid_logits[i], **kwargs)
                outputs.append(
                    MeshExtractResult(
                        verts=verts.float(),
                        faces=faces,
                        res=kwargs["octree_resolution"],
                    )
                )

            except Exception:
                import traceback

                traceback.print_exc()
                outputs.append(None)

        return outputs


class MCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        verts, faces, normals, _ = measure.marching_cubes(
            grid_logit.float().cpu().numpy(), mc_level, method="lewiner"
        )
        grid_size, bbox_min, bbox_size = self._compute_box_stat(
            bounds, octree_resolution
        )
        verts = verts / grid_size * bbox_size + bbox_min
        verts = torch.tensor(verts, device=grid_logit.device, dtype=torch.float32)
        faces = torch.tensor(
            np.ascontiguousarray(faces), device=grid_logit.device, dtype=torch.long
        )
        faces = faces[:, [2, 1, 0]]
        return verts, faces


class DMCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, octree_resolution, **kwargs):
        device = grid_logit.device
        if not hasattr(self, "dmc"):
            try:
                from diso import DiffDMC
            except:
                raise ImportError(
                    "Please install diso via `pip install diso`, or set mc_algo to 'mc'"
                )
            self.dmc = DiffDMC(dtype=torch.float32).to(device)
        sdf = -grid_logit / octree_resolution
        sdf = sdf.to(torch.float32).contiguous()
        verts, faces = self.dmc(sdf, deform=None, return_quads=False, normalize=True)
        grid_size, bbox_min, bbox_size = self._compute_box_stat(
            kwargs["bounds"], octree_resolution
        )
        verts = verts * kwargs["bounds"] * 2 - kwargs["bounds"]
        return verts, faces
