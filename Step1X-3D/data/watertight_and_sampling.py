import os
import math
import torch
import torch.nn.functional as F
import igl # pip install libigl==2.5.1
import trimesh
import mcubes # pip install mcubes
import numpy as np
import nvdiffrast.torch as dr

from pysdf import SDF
from matplotlib import image
from argparse import ArgumentParser

def sample_from_sphere(num_views, radius, upper=False):
    """sample x,y,z location from the sphere
    reference: https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
    """
    num_views = num_views * 2 if upper else num_views
    phi = (np.sqrt(5) - 1.0) / 2.0
    pos_list = []
    for n in range(1, num_views + 1):
        y = (2.0 * n - 1) / num_views - 1.0
        x = np.cos(2 * np.pi * n * phi) * np.sqrt(1 - y * y)
        z = np.sin(2 * np.pi * n * phi) * np.sqrt(1 - y * y)
        if upper and y < 0:
            continue
        pos_list.append((x * radius, y * radius, z * radius))
    return np.array(pos_list)


class MeshRenderer:
    def __init__(
            self,
            resolution=(1024, 1024), # resolution of the rendered image
            near=0.1, # near plane for the camera
            far=10.0, # far plane for the camera
            device='cuda' # device to run the renderer on
            ):
        """Initialize the mesh renderer."""
        self.resolution = resolution
        self.near = near
        self.far = far
        self.device = torch.device(device)
        # check if the device is cuda
        if torch.cuda.is_available() and device == 'cuda':
            self._ctx = dr.RasterizeCudaContext(device=self.device)
        elif device == 'cpu':
            self._ctx = dr.RasterizeGLContext(device=self.device)
        else:
            raise ValueError("Device must be 'cuda' or 'cpu'.")

        # warm up the renderer
        self._warmup()

    def _warmup(self):
        """Warm up the renderer to avoid the first frame being slow."""
        #windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
        def tensor(*args, **kwargs):
            return torch.tensor(*args, device='cuda', **kwargs)
        pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
        tri = tensor([[0, 1, 2]], dtype=torch.int32)
        dr.rasterize(self._ctx, pos, tri, resolution=[256, 256])


    def rasterize(
        self,
        pos: torch.FloatTensor,
        tri: torch.IntTensor,
        resolution = (1024, 1024), # resolution of the rendered image
        grad_db: bool = True,
    ):
        """
        Rasterize the given vertices and triangles.
        Args:
            pos (Float[Tensor, "B Nv 4"]): Vertex positions
            tri (Integer[Tensor, "Nf 3"]): Triangle indices
            resolution (Union[int, Tuple[int, int]]): Output resolution
            grad_db (Bool): Enable gradient backpropagation
        Returns:
            Rasterized outputs
        """
        # rasterize in instance mode (single topology)
        return dr.rasterize(
            self._ctx, pos.float(), tri.int(), resolution, grad_db=grad_db
        )

    def interpolate(
        self,
        attr: torch.FloatTensor,
        rast: torch.FloatTensor,
        tri: torch.IntTensor,
        rast_db=None,
        diff_attrs=None,
    ):
        """
        Interpolate attributes using the given rasterization outputs.
        Args:
            attr (Float[Tensor, "B Nv C"]): Attributes to interpolate
            rast (Float[Tensor, "B H W 4"]): Rasterization outputs
            tri (Integer[Tensor, "Nf 3"]): Triangle indices
            rast_db (Float[Tensor, "B H W 4"], optional): Differentiable rasterization outputs
            diff_attrs (Float[Tensor, "B Nv C"], optional): Differentiable attributes
        Returns:
            Interpolated attribute values
        """
        return dr.interpolate(
            attr.float(), rast, tri.int(), rast_db=rast_db, diff_attrs=diff_attrs
        )

    def render(
        self,
        mesh: trimesh.Trimesh, # trimesh object
        cam2world_matrixs: torch.Tensor, #N,4,4
        mvp_matrixs: torch.Tensor, #N,4,4
        render_vert_depth: bool = True, # whether to render vertex depth
        render_face_normals: bool = True, # whether to render face normals
    ):
        """
        Render the mesh using the given camera and model view projection matrices.
        Args:
            mesh (trimesh.Trimesh): The mesh to render
            cam2world_matrixs (torch.Tensor): Camera to world matrix (N, 4, 4)
            mvp_matrixs (torch.Tensor): Model view projection matrix (N, 4, 4)
            render_vert_depth (bool): Whether to render vertex depth
            render_face_normals (bool): Whether to render face normals
        Returns:
            results (dict): Dictionary containing rendered outputs
        """
        results = {}

        v_pos = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device) # (num_vertices, 3)
        t_pos_idx = torch.tensor(mesh.faces, dtype=torch.int32, device=self.device) # (num_faces, 3)

        verts_homo = torch.cat([v_pos, torch.ones([v_pos.shape[0], 1]).to(v_pos)], dim=-1)
        v_pos_clip: Float[Tensor, "B Nv 4"] = torch.matmul(verts_homo, mvp_matrixs.permute(0, 2, 1))

        rast, _ = self.rasterize(v_pos_clip, t_pos_idx, self.resolution)
        mask = rast[..., 3:] > 0

        if render_vert_depth:
            verts_homo = torch.cat(
                [
                    v_pos, torch.ones([v_pos.shape[0], 1]).to(v_pos),
                ],
                dim=-1,
            )
            v_pos_cam = verts_homo @ cam2world_matrixs.inverse().transpose(-1, -2)
            v_depth = v_pos_cam[..., 2:3] * -1  # (B,n_v,1)
            gb_depth, _ = self.interpolate(
                v_depth.contiguous(), rast, t_pos_idx
            )
            gb_depth[~mask] = self.far
            results.update({"vert_depth": gb_depth})
        

        if render_face_normals:
            flat_face_index = torch.arange(
                len(t_pos_idx) * 3, device=self.device, dtype=torch.int
            ).reshape(-1, 3)
            
            i0 = t_pos_idx[:, 0]
            i1 = t_pos_idx[:, 1]
            i2 = t_pos_idx[:, 2]

            v0 = v_pos[i0, :]
            v1 = v_pos[i1, :]
            v2 = v_pos[i2, :]

            face_normals = torch.linalg.cross(v1 - v0, v2 - v0)
            f_nrm = face_normals[:, None, :].repeat(1, 3, 1).reshape(-1, 3)

            gb_normal, _ = self.interpolate(f_nrm, rast, flat_face_index)

            gb_normal = gb_normal.view(-1, self.resolution[0] * self.resolution[1], 3)
            gb_normal = torch.matmul(
                torch.linalg.inv(cam2world_matrixs[:, :3, :3]),
                gb_normal.transpose(1, 2),
            ).transpose(1, 2)
            gb_normal = gb_normal.view(-1, self.resolution[0], self.resolution[0], 3)
            gb_normal = F.normalize(gb_normal, dim=-1).contiguous()
            gb_normal = torch.lerp(
                torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
            )
            gb_normal = torch.cat([gb_normal, mask], dim=-1)
            results.update({"face_normal": gb_normal})

        return results


@torch.no_grad()
def visibility_check(points, depths, cam2world_matrixs, mvp_matrixs):
    '''
    Visibility check for points in 3D space

    Args:
    - points: (n_points, 3), 3D points in world space
    - depths: (n_view, H, W, 1), depth maps
    - cam2world_matrixs: (n_views, 4, 4), camera to world matrix
    - mvp_matrixs: (n_views, 4, 4), model view projection matrix

    Returns:
    - mask: (n_points, ), visibility mask
    - dist: (n_points, ), distance to the visible surface
    '''
    dist = torch.ones(points.shape[0]).to(points) # defult as one
    mask = torch.zeros(points.shape[0], dtype=torch.bool).to(points.device)  # visibility

    points_homo = torch.cat(
        [points, torch.ones([points.shape[0], 1]).to(points)], dim=-1
    )
    for i, cam2world_matrix in enumerate(cam2world_matrixs):
        points_clip_i = points_homo @ mvp_matrixs[i].permute(1,0)
        valid_region = (torch.abs(points_clip_i[...,0]) < 0.999) & \
            (torch.abs(points_clip_i[...,1]) < 0.999) 
        points_valid = points_clip_i[valid_region].float()

        v_pos_cam = points_homo @ cam2world_matrix.inverse().transpose(-1, -2)
        v_depth = v_pos_cam[..., 2:3] * -1

        # query using (u, v)
        sample_z = torch.nn.functional.grid_sample(depths[i].view(1, 1, depths.shape[1], depths.shape[2]).float(),
            points_valid[:, :2].reshape(1, 1, points_valid.shape[0], 2), align_corners=True, mode='bilinear').reshape(-1) 

        visible_points = v_depth[valid_region].squeeze() < sample_z  # visible if z smaller than render depth
        mask[torch.where(valid_region)[0][torch.where(visible_points)[0]]] = True

        # dist to hitting point along camera ray
        dist[valid_region] = torch.minimum(dist[valid_region], torch.abs(sample_z - v_depth[valid_region].squeeze())) 

    return mask, dist


@torch.no_grad()
def watertight(
        mesh, 
        grid_resolution=256, 
        device='cuda', 
        num_views=50, 
        sample_size=2.1, 
        winding_number_thres=0.5,
    ):
    """
    Convert a mesh to a watertight mesh using trimesh.
    
    Args:
        mesh: Input mesh as a trimesh object
        grid_resolution: Resolution of the grid for sampling
        device: Device to run the script on (cpu or cuda)
        num_views: Number of views for visibility check, default is 50
        sample_size: Size of the sample space, default is 2.1
        winding_number_thres: Threshold for winding number, default is 0.5
    
    Returns:
        watertight_mesh: A watertight mesh as a trimesh object
    """
    # setup grid points
    x,y,z = np.meshgrid(
        np.arange(grid_resolution, dtype=np.float32), 
        np.arange(grid_resolution, dtype=np.float32), 
        np.arange(grid_resolution, dtype=np.float32),  
        indexing='ij')
    grid_points = np.stack(
            (x.reshape(-1) + 0.5, y.reshape(-1) + 0.5, z.reshape(-1) + 0.5
        ),axis=-1) / grid_resolution * sample_size - sample_size / 2.0
    grid_points = torch.tensor(grid_points).to(device)
    print(f"number of grid_points: {grid_points.shape} with resolution {grid_resolution}, range {grid_points.min()} to {grid_points.max()}")
    
    # setup for rendering depth maps
    cam_poses = sample_from_sphere(num_views, 4.0, upper=False) # (num_views, 3)
    scale = 1.0 # scale for the orthogonal camera projection matrix
    resolution = 1024 # resolution of the rendered images
    aspect_ratio = 1.0 # aspect ratio of the rendered images
    near, far = 0.1, 10.0 # near and far plane for the camera
    cam2world_matrixs, mvp_matrixs = [], []
    for position in cam_poses:
        # extrinsic matrix
        backward = np.array([0, 0, 0]) - position
        backward = backward / np.linalg.norm(backward)
        right = np.cross(backward, np.array([0, 1, 0]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, backward)

        R = np.stack([right, up, -backward], axis=0)
        t = -R @ position
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        cam2world_matrixs.append(np.linalg.inv(extrinsic)) # (4, 4)

        # projection matrix
        proj_mtx = np.zeros([4, 4])
        proj_mtx[0, 0] = scale
        proj_mtx[1, 1] = scale * -1
        proj_mtx[2, 2] = -2 / (far - near)
        proj_mtx[2, 3] = -(far + near) / (far - near)
        proj_mtx[3, 3] = 1
        mvp_matrix = proj_mtx @ extrinsic
        mvp_matrixs.append(mvp_matrix) # (4, 4)
    cam2world_matrixs = torch.tensor(np.array(cam2world_matrixs), dtype=torch.float32).to(device) # (num_views, 4, 4)
    mvp_matrixs = torch.tensor(np.array(mvp_matrixs), dtype=torch.float32).to(device) # (num_views, 4, 4)

    rendererd_imgs = MeshRenderer((resolution, resolution), near, far, device).render(
        mesh,
        cam2world_matrixs=cam2world_matrixs,
        mvp_matrixs=mvp_matrixs,
    )

    # STEP A. do visibility_check for each grid point
    visibility, dist = visibility_check(grid_points, rendererd_imgs['vert_depth'], cam2world_matrixs, mvp_matrixs)
    winding_numbers = igl.fast_winding_number_for_meshes(
        np.array(mesh.vertices, dtype=np.float32), 
        np.array(mesh.faces, dtype=np.int32), 
        grid_points.detach().cpu().numpy()
    )
    winding_numbers = torch.from_numpy(winding_numbers).to(device)
    visibility[visibility & (winding_numbers > winding_number_thres)]= False # combine visibility with winding number mask

    ## STEP B. refine sdf close to the surface
    near_surface_idx = torch.where(dist < 1.0)[0]
    squared_distances, closest_points, face_indices = \
            igl.point_mesh_squared_distance(grid_points[near_surface_idx].detach().cpu().numpy(),
                                            mesh.vertices,
                                            mesh.faces)
    squared_distances = torch.from_numpy(squared_distances).to(grid_points)
    dist[near_surface_idx] = torch.sqrt(squared_distances)

    ## STEP C. convert udf to sdf
    dist[visibility==False] *= -1

    ## STEP D. generate the mesh using Marching Cube
    sdf = dist.view(grid_resolution, grid_resolution, grid_resolution)
    # not the 0-level surface, we use the surface with a small offset
    mesh = mcubes.marching_cubes(sdf.cpu().numpy(), sample_size / grid_resolution)
    mesh = trimesh.Trimesh(
        vertices=mesh[0] / grid_resolution * sample_size - sample_size / 2.0,
        faces=mesh[1],
        process=False
    )

    return mesh


def sharp_edge_sampling(mesh_path, num_views=100000, sharpness_threshold=math.radians(30)):
    """
    Sample points on sharp edges of the mesh.
    Code borrowed from the Dora github repository: https://github.com/Seed3D/Dora/blob/main/sharp_edge_sampling/sharp_sample.py#L37
    Please consider citing the Dora paper if you use this code in your work.

    Args:
        mesh_path: Path to the OBJ file
        sharpness_threshold: Threshold for sharp edge detection
        num_views: Target number of points to generate
    
    Returns:
        sharp_surface: Array of sharp surface points with positions and normals
    """
    import bpy # bpy==4.0.0
    import bmesh
    # Import OBJ file
    bpy.ops.wm.obj_import(filepath=mesh_path)
    obj = bpy.context.selected_objects[0]

    # Enter Edit mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Ensure edge selection mode
    bpy.ops.mesh.select_mode(type="EDGE")

    # Select sharp edges
    bpy.ops.mesh.edges_select_sharp(sharpness=sharpness_threshold)

    # Switch back to Object mode to access selection state
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create bmesh instance
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # Get selected sharp edges
    sharp_edges = [edge for edge in bm.edges if edge.select]

    # Collect sharp edge vertex pairs
    sharp_edges_vertices = []
    link_normal1 = []
    link_normal2 = []
    sharp_edges_angle = []
    # Unique vertices set
    vertices_set = set()

    for edge in sharp_edges:
        vertices_set.update(edge.verts[:])  # Add to unique vertices set
        
        # Collect sharp edge vertex pair indices
        sharp_edges_vertices.append([edge.verts[0].index, edge.verts[1].index])

        # Get normals of linked faces
        normal1 = edge.link_faces[0].normal
        normal2 = edge.link_faces[1].normal

        link_normal1.append(normal1)
        link_normal2.append(normal2)

        if normal1.length == 0.0 or normal2.length == 0.0:
            sharp_edges_angle.append(0.0)
        # Compute the angle between the two normals
        else:
            sharp_edges_angle.append(math.degrees(normal1.angle(normal2)))

    # Extract vertex data
    vertices = []
    vertices_index = []
    vertices_normal = []

    for vertex in vertices_set:
        vertices.append(vertex.co)
        vertices_index.append(vertex.index)
        vertices_normal.append(vertex.normal)

    # Convert to numpy arrays
    vertices = np.array(vertices)
    vertices_index = np.array(vertices_index)
    vertices_normal = np.array(vertices_normal)

    sharp_edges_count = np.array(len(sharp_edges))
    sharp_edges_angle_array = np.array(sharp_edges_angle)

    if sharp_edges_count > 0:
        sharp_edge_link_normal = np.array(np.concatenate([link_normal1, link_normal2], axis=1))
        nan_mask = np.isnan(sharp_edge_link_normal)
        # Replace NaN values with 0 using boolean indexing
        sharp_edge_link_normal = np.where(nan_mask, 0, sharp_edge_link_normal)
        
        nan_mask = np.isnan(vertices_normal)
        # Replace NaN values with 0 using boolean indexing
        vertices_normal = np.where(nan_mask, 0, vertices_normal)

    # Convert to numpy array
    sharp_edges_vertices_array = np.array(sharp_edges_vertices)

    if sharp_edges_count > 0:
        mesh = trimesh.load(mesh_path, process=False, force='mesh')
        num_target_sharp_vertices = num_views // 2
        sharp_edge_length = sharp_edges_count
        sharp_edges_vertices_pair = sharp_edges_vertices_array
        sharp_vertices_pair = mesh.vertices[sharp_edges_vertices_pair]  # Vertex pair coordinates (1225, 2, 3)
        epsilon = 1e-4  # Small numerical value
        
        # Calculate edge normals
        edge_normal = 0.5 * sharp_edge_link_normal[:, :3] + 0.5 * sharp_edge_link_normal[:, 3:]
        norms = np.linalg.norm(edge_normal, axis=1, keepdims=True)
        norms = np.where(norms > epsilon, norms, epsilon)
        edge_normal = edge_normal / norms  # Normalize edge normals
        
        known_vertices = vertices  # Unique sharp vertices
        known_vertices_normal = vertices_normal
        known_vertices = np.concatenate([known_vertices, known_vertices_normal], axis=1)

        num_known_vertices = known_vertices.shape[0]  # Number of unique sharp vertices
        
        if num_known_vertices < num_target_sharp_vertices:  # If known vertices < target vertices
            num_new_vertices = num_target_sharp_vertices - num_known_vertices
            
            if num_new_vertices >= sharp_edge_length:  # If new vertices needed >= sharp edges count
                # Each sharp edge needs at least one interpolated vertex
                num_new_vertices_per_pair = num_new_vertices // sharp_edge_length  # Vertices per edge
                new_vertices = np.zeros((sharp_edge_length, num_new_vertices_per_pair, 6))  # Initialize new vertices array

                start_vertex = sharp_vertices_pair[:, 0]
                end_vertex = sharp_vertices_pair[:, 1]
                
                for j in range(1, num_new_vertices_per_pair + 1):
                    t = j / float(num_new_vertices_per_pair + 1)
                    new_vertices[:, j - 1, :3] = (1 - t) * start_vertex + t * end_vertex
                    new_vertices[:, j - 1, 3:] = edge_normal  # Same normal within each edge
                
                new_vertices = new_vertices.reshape(-1, 6)

                remaining_vertices = num_new_vertices % sharp_edge_length  # Calculate remaining vertices
                if remaining_vertices > 0:
                    rng = np.random.default_rng()
                    ind = rng.choice(sharp_edge_length, remaining_vertices, replace=False)
                    new_vertices_remain = np.zeros((remaining_vertices, 6))  # Initialize remaining vertices array
                    
                    start_vertex = sharp_vertices_pair[ind, 0]
                    end_vertex = sharp_vertices_pair[ind, 1]
                    t = np.random.rand(remaining_vertices).reshape(-1, 1)
                    new_vertices_remain[:, :3] = (1 - t) * start_vertex + t * end_vertex

                    edge_normal = 0.5 * sharp_edge_link_normal[ind, :3] + 0.5 * sharp_edge_link_normal[ind, 3:]
                    edge_normal = edge_normal / np.linalg.norm(edge_normal, axis=1, keepdims=True)
                    new_vertices_remain[:, 3:] = edge_normal

                    new_vertices = np.concatenate([new_vertices, new_vertices_remain], axis=0)
            else:
                remaining_vertices = num_new_vertices % sharp_edge_length  # Calculate remaining vertices to allocate
                if remaining_vertices > 0:
                    rng = np.random.default_rng()
                    ind = rng.choice(sharp_edge_length, remaining_vertices, replace=False)
                    new_vertices_remain = np.zeros((remaining_vertices, 6))  # Initialize new vertices array
                    
                    start_vertex = sharp_vertices_pair[ind, 0]
                    end_vertex = sharp_vertices_pair[ind, 1]
                    t = np.random.rand(remaining_vertices).reshape(-1, 1)
                    new_vertices_remain[:, :3] = (1 - t) * start_vertex + t * end_vertex

                    edge_normal = 0.5 * sharp_edge_link_normal[ind, :3] + 0.5 * sharp_edge_link_normal[ind, 3:]
                    edge_normal = edge_normal / np.linalg.norm(edge_normal, axis=1, keepdims=True)
                    new_vertices_remain[:, 3:] = edge_normal

                    new_vertices = new_vertices_remain

            sharp_surface = np.concatenate([new_vertices, known_vertices], axis=0)
        else:
            sharp_surface = known_vertices
        # Make sure the sharp surface has the correct number of samples
        sharp_surface = sharp_surface[np.random.choice(sharp_surface.shape[0], num_views, replace=True), :]
        print(f"Sampled {sharp_surface.shape[0]} points on sharp edges of the mesh.")
        # manually remove the bpy object and free memory
        bm.free()
        bpy.data.objects.remove(obj, do_unlink=True)

        return sharp_surface
    else:
        print("No sharp edges found in the mesh.")
        return None


if __name__ == "__main__":
    parser = ArgumentParser(description="Watertight mesh and sampling points")
    parser.add_argument("--input_mesh", type=str, required=True, help="Path to the input mesh file")
    parser.add_argument("--output_path", type=str, default="./output", help="Path to save the watertight mesh and sampled points")
    parser.add_argument("--skip_watertight", action='store_true', help="Skip the watertight check and directly sample points from the mesh")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the script on (cpu or cuda)")
    # Add command-line arguments for watertight conversion
    parser.add_argument("--grid_resolution", type=int, default=256, help="Resolution of the grid for sampling")
    # Add command-line arguments for sampling
    parser.add_argument("--sample_sharp_edge", type=bool, default=True, help="Sample points on sharp edges of the mesh in Dora paper")
    parser.add_argument("--angle_threshold", type=float, default=15.0, help="Angle threshold for sharp edge detection in degrees")
    parser.add_argument("--num_surface_points", type=int, default=100000, help="Number of points to sample from the mesh")
    parser.add_argument("--num_sharp_surface_points", type=int, default=100000, help="Number of points to sample from the sharp edges of the mesh")
    parser.add_argument("--num_near_surface_points", type=int, default=100000, help="Number of points to sample near the mesh surface")
    parser.add_argument("--num_vlume_points", type=int, default=100000, help="Number of points to sample inside the mesh volume")
    parser.add_argument("--bounds", type=float, default=1.05, help="Bounds for sampling points in the mesh, a little larger than the mesh size")
    args = parser.parse_args()

    # Load the mesh
    mesh = trimesh.load(args.input_mesh, force='mesh')
    # Normalize the mesh into a unit cube
    mesh.apply_translation(-np.mean(mesh.vertices, axis=0))
    mesh.apply_scale(1.0 / np.max(np.abs(mesh.vertices)))

    # Check if the mesh is watertight
    if mesh.is_watertight and args.skip_watertight:
        print("Mesh is already watertight. Proceeding to sample points.")
    else:
        print("Attempting to convert the mesh to a watertight mesh.")
        mesh = watertight(
            mesh,
            grid_resolution=args.grid_resolution,
            device=args.device,
        )

    # Save the watertight mesh
    output_path = f"{args.output_path}/{args.input_mesh.split('/')[-1].split('.')[0]}"
    os.makedirs(output_path, exist_ok=True)
    mesh.export(f"{output_path}/watertight_mesh.obj")
    print(f"Watertight mesh saved to {output_path}/watertight_mesh.obj")

    # sample points near the surface and in the space within bounds
    surface_points, faces = mesh.sample(args.num_surface_points, return_index=True)
    near_points = [
        surface_points + np.random.normal(scale=0.001, size=(args.num_near_surface_points, 3)),
        surface_points + np.random.normal(scale=0.01, size=(args.num_near_surface_points, 3)),
    ]
    near_surface_points = np.concatenate(near_points)
    volume_rand_points = np.random.uniform(-args.bounds, args.bounds, size=(args.num_vlume_points, 3))
    f = SDF(mesh.vertices, mesh.faces); # (num_vertices, 3) and (num_faces, 3)
    # compute SDF values for the sampled points
    near_surface_points_with_sdf = np.concatenate([near_surface_points, f(near_surface_points)[:, np.newaxis]], axis=1) # (num_near_surface_points, 4)
    volume_rand_points_with_sdf = np.concatenate([volume_rand_points, f(volume_rand_points)[:, np.newaxis]], axis=1) # (num_vlume_points, 4)

    # Sample points with normals on the surface
    surface_points, faces = mesh.sample(args.num_surface_points, return_index=True)
    normals = mesh.face_normals[faces]
    surface = np.concatenate([surface_points, normals], axis=1)
    if args.sample_sharp_edge:
        # Sample points on sharp edges
        print("Sampling points on sharp edges of the mesh.")
        sharp_surface = sharp_edge_sampling(
            f"{output_path}/watertight_mesh.obj", # fix bug
            num_views=args.num_sharp_surface_points,
            sharpness_threshold=math.radians(args.angle_threshold)
        )
        # Save the samples
        np.savez(
            f'{output_path}/samples.npz',
            surface=surface, # (num_surface_points, 6), surface points with normals
            sharp_surface=sharp_surface, # (num_sharp_surface_points, 6), sharp surface points with normals
            near_surface_points=near_surface_points_with_sdf, # (num_near_surface_points, 4), sampled points near the surface with SDF values
            volume_rand_points=volume_rand_points_with_sdf, # (num_vlume_points, 4), sampled points in the volume within bounds with SDF values
            bounds=np.array([-args.bounds, args.bounds])
        )
    else:
        # Save the samples
        np.savez(
            f'{output_path}/samples.npz',
            surface=surface, # (num_surface_points, 6), surface points with normals
            near_surface_points=near_surface_points_with_sdf, # (num_near_surface_points, 4), sampled points near the surface with SDF values
            volume_rand_points=volume_rand_points_with_sdf, # (num_vlume_points, 4), sampled points in the volume within bounds with SDF values
            bounds=np.array([-args.bounds, args.bounds])
        )
