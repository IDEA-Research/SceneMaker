import bpy
from mathutils import Euler, Matrix, Vector
from typing import List, Tuple, Literal, Optional
import numpy as np
import math
import os
import imageio
from PIL import Image

def add_camera(
    cam2world_matrix: Matrix,
    camera_type: Literal["PERSP", "ORTHO"] = "PERSP",
    camera_sensor_width: int = 32,
    camera_lens: int = 35,
    ortho_scale: int = 1.1,
    add_frame: bool = False,
):
    if not isinstance(cam2world_matrix, Matrix):
        cam2world_matrix = Matrix(cam2world_matrix)
    if bpy.context.scene.camera is None:
        bpy.ops.object.camera_add(location=(0, 0, 0))
        for obj in bpy.data.objects:
            if obj.type == "CAMERA":
                bpy.context.scene.camera = obj

    cam_ob = bpy.context.scene.camera
    cam_ob.data.type = camera_type
    cam_ob.data.sensor_width = camera_sensor_width
    if camera_type == "PERSP":
        cam_ob.data.lens = camera_lens
    elif camera_type == "ORTHO":
        cam_ob.data.ortho_scale = ortho_scale
    cam_ob.matrix_world = cam2world_matrix

    frame = bpy.context.scene.frame_end
    cam_ob.keyframe_insert(data_path="location", frame=frame)
    cam_ob.keyframe_insert(data_path="rotation_euler", frame=frame)
    cam_ob.data.keyframe_insert(data_path="type", frame=frame)
    cam_ob.data.keyframe_insert(data_path="sensor_width", frame=frame)

    if camera_type == "ORTHO":
        cam_ob.data.keyframe_insert(data_path="ortho_scale", frame=frame)
    elif camera_type == "PERSP":
        cam_ob.data.keyframe_insert(data_path="lens", frame=frame)

    if add_frame:
        bpy.context.scene.frame_end += 1

    return cam_ob


def init_render_engine(
    engine: Literal["CYCLES", "BLENDER_EEVEE"], render_samples: int = 64
):
    """Initialize the rendering engine.

    Args:
        engine (Literal[&quot;CYCLES&quot;, &quot;BLENDER_EEVEE&quot;]):
            The rendering engine to use. Either CYCLES or BLENDER_EEVEE.
        render_samples (int, optional):
            Number of samples to render. Defaults to 64.

    Raises:
        ValueError: If the engine is not CYCLES or BLENDER_EEVEE.
    """
    if engine == "CYCLES":
        cycles_init(render_samples)
    elif engine == "BLENDER_EEVEE":
        eevee_init(render_samples)
    else:
        raise ValueError(f"Unknown engine: {engine}")


def eevee_init(render_samples: int):
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.context.scene.eevee.taa_render_samples = render_samples
    bpy.context.scene.eevee.use_gtao = True
    bpy.context.scene.eevee.use_ssr = True
    bpy.context.scene.eevee.use_bloom = True
    bpy.context.scene.render.use_high_quality_normals = True


def cycles_init(render_samples: int):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = render_samples
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3
    bpy.context.scene.cycles.transmission_bounces = 3
    bpy.context.scene.cycles.filter_width = 0.01
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.render.film_transparent = True


def set_env_map(env_path: str):
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    else:
        world.use_nodes = True
        world.node_tree.nodes.clear()

    env_texture_node = world.node_tree.nodes.new(type="ShaderNodeTexEnvironment")
    env_texture_node.location = (0, 0)

    bg_node = world.node_tree.nodes.new(type="ShaderNodeBackground")
    bg_node.location = (400, 0)

    output_node = world.node_tree.nodes.new(type="ShaderNodeOutputWorld")
    output_node.location = (800, 0)

    links = world.node_tree.links
    links.new(env_texture_node.outputs["Color"], bg_node.inputs["Color"])
    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])

    bpy.ops.image.open(filepath=env_path)
    env_texture_node.image = bpy.data.images.get(os.path.basename(env_path))


def set_background_color(rgba: List = [1.0, 1.0, 1.0, 1.0]):
    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes["Background"]
    back_node.inputs["Color"].default_value = Vector(rgba)
    back_node.inputs["Strength"].default_value = 1.0


def load_file(path):
    """A naive function"""
    if path.endswith(".vrm"):
        bpy.ops.import_scene.vrm(filepath=path)
    elif path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=path)
    elif path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=path)
    elif path.endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=path)
    else:
        raise RuntimeError(f"Invalid input file: {path}")


def load_armature(path, ignore_components: List = []):
    currentObjects = set(bpy.data.objects)
    load_file(path)
    toRemove = [
        x
        for x in bpy.data.objects
        if x not in currentObjects
        and any([component in x.name for component in ignore_components])
    ]
    for obj in toRemove:
        bpy.data.objects.remove(obj, do_unlink=True)
    objects = [
        x for x in bpy.data.objects if x not in currentObjects and x.type == "ARMATURE"
    ]
    armature = objects[0]

    return armature


def enable_color_output(
    width: int,
    height: int,
    output_dir: Optional[str] = "",
    file_prefix: str = "color_",
    file_format: Literal["WEBP", "PNG"] = "WEBP",
    mode: Literal["IMAGE", "VIDEO"] = "IMAGE",
    **kwargs,
):
    film_transparent = kwargs.get("film_transparent", True)
    fps = kwargs.get("fps", 24)

    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = film_transparent
    scene.render.image_settings.quality = 100

    if mode == "IMAGE":
        scene.render.image_settings.file_format = file_format
        scene.render.image_settings.color_mode = "RGBA"
        # scene.render.image_settings.color_depth = "16"
    elif mode == "VIDEO":
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.fps = fps
    scene.render.filepath = os.path.join(output_dir, file_prefix)


def enable_normals_output(output_dir: Optional[str] = "", file_prefix: str = "normal_"):
    bpy.context.view_layer.use_pass_normal = True
    bpy.context.view_layer.use_pass_z = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # # Create input render layer node.
    render_layers = nodes.new('CompositorNodeRLayers')

    scale_normal = nodes.new(type="CompositorNodeMixRGB")
    scale_normal.blend_type = 'MULTIPLY'
    scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])
    bias_normal = nodes.new(type="CompositorNodeMixRGB")
    bias_normal.blend_type = 'ADD'
    bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_normal.outputs[0], bias_normal.inputs[1])
    normal_file_output = nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    links.new(bias_normal.outputs[0], normal_file_output.inputs[0])
    normal_file_output.base_path = output_dir
    normal_file_output.file_slots.values()[0].path = file_prefix
    normal_file_output.format.file_format = "OPEN_EXR" # default is "PNG"
    normal_file_output.format.color_mode = "RGB"  # default is "BW"


def enable_depth_output(output_dir: Optional[str] = "", file_prefix: str = "depth_"):
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    links = tree.links

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]
    bpy.context.view_layer.use_pass_z = True

    depth_output = tree.nodes.new("CompositorNodeOutputFile")
    depth_output.base_path = output_dir
    depth_output.name = "DepthOutput"
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.format.color_depth = "32"
    depth_output.file_slots.values()[0].path = file_prefix

    links.new(rl.outputs["Depth"], depth_output.inputs["Image"])


def enable_albedo_output(output_dir: Optional[str] = "", file_prefix: str = "albedo_"):
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree

    if "Render Layers" not in tree.nodes:
        rl = tree.nodes.new("CompositorNodeRLayers")
    else:
        rl = tree.nodes["Render Layers"]
    bpy.context.view_layer.use_pass_diffuse_color = True

    alpha_albedo = tree.nodes.new(type="CompositorNodeSetAlpha")
    tree.links.new(rl.outputs["DiffCol"], alpha_albedo.inputs["Image"])
    tree.links.new(rl.outputs["Alpha"], alpha_albedo.inputs["Alpha"])

    albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.base_path = output_dir
    albedo_file_output.file_slots[0].use_node_format = True
    albedo_file_output.format.file_format = "PNG"
    albedo_file_output.format.color_mode = "RGBA"
    albedo_file_output.format.color_depth = "16"
    albedo_file_output.file_slots.values()[0].path = file_prefix

    tree.links.new(alpha_albedo.outputs["Image"], albedo_file_output.inputs[0])

def get_local2world_mat(blender_obj) -> np.ndarray:
    """Returns the pose of the object in the form of a local2world matrix.
    :return: The 4x4 local2world matrix.
    """
    obj = blender_obj
    # Start with local2parent matrix (if obj has no parent, that equals local2world)
    matrix_world = obj.matrix_basis

    # Go up the scene graph along all parents
    while obj.parent is not None:
        # Add transformation to parent frame
        matrix_world = (
            obj.parent.matrix_basis @ obj.matrix_parent_inverse @ matrix_world
        )
        obj = obj.parent

    return np.array(matrix_world)

def build_transformation_mat(translation, rotation) -> np.ndarray:
    """Build a transformation matrix from translation and rotation parts.

    :param translation: A (3,) vector representing the translation part.
    :param rotation: A 3x3 rotation matrix or Euler angles of shape (3,).
    :return: The 4x4 transformation matrix.
    """
    translation = np.array(translation)
    rotation = np.array(rotation)

    mat = np.eye(4)
    if translation.shape[0] == 3:
        mat[:3, 3] = translation
    else:
        raise RuntimeError(
            f"Translation has invalid shape: {translation.shape}. Must be (3,) or (3,1) vector."
        )
    if rotation.shape == (3, 3):
        mat[:3, :3] = rotation
    elif rotation.shape[0] == 3:
        mat[:3, :3] = np.array(Euler(rotation).to_matrix())
    else:
        raise RuntimeError(
            f"Rotation has invalid shape: {rotation.shape}. Must be rotation matrix of shape "
            f"(3,3) or Euler angles of shape (3,) or (3,1)."
        )

    return mat


def get_camera_positions_on_sphere(
    center: Tuple[float, float, float],
    radius: float,
    elevations: List[float],
    num_camera_per_layer: Optional[int] = None,
    azimuth_offset: Optional[float] = 0.0,
    azimuths: Optional[List[float]] = None,
) -> Tuple[List, List, List, List]:
    """
    Get camera positions on a sphere around a center point.

    Places cameras at specified elevation angles around a sphere, with a given number of cameras
    per elevation layer. The cameras are positioned to look at the center point.

    Args:
        center: (x,y,z) coordinates of the sphere center
        radius: Radius of the sphere
        elevations: List of elevation angles in degrees
        num_camera_per_layer: Number of cameras to place at each elevation angle

    Returns:
        Tuple containing:
        - points: List of camera position vectors
        - mats: List of camera transformation matrices
        - elevation_t: List of elevation angles for each camera
        - azimuth_t: List of azimuth angles for each camera
    """
    points, mats, elevation_t, azimuth_t = [], [], [], []

    elevation_deg = elevations
    elevation = np.deg2rad(elevation_deg)

    if num_camera_per_layer is not None and azimuths is None:
        azimuth_deg = np.linspace(0, 360, num_camera_per_layer + 1)[:-1]
        azimuth_deg = azimuth_deg % 360
        if azimuth_offset is not None:
            azimuth_deg += azimuth_offset
    else:
        azimuth_deg = azimuths
    azimuth = np.deg2rad(azimuth_deg)
    for _phi, theta in zip(elevation, azimuth):
        phi = 0.5 * math.pi - _phi
        elevation_t.append(_phi)
        azimuth_t.append(theta)

        r = radius
        x = center[0] + r * math.sin(phi) * math.cos(theta)
        y = center[1] + r * math.sin(phi) * math.sin(theta)
        z = center[2] + r * math.cos(phi)
        cam_pos = Vector((x, y, z))
        points.append(cam_pos)

        center = Vector(center)
        rotation_euler = (center - cam_pos).to_track_quat("-Z", "Y").to_euler()
        cam_matrix = build_transformation_mat(cam_pos, rotation_euler)
        mats.append(cam_matrix)

    return points, mats, elevation_t, azimuth_t

def convert_normal_to_webp(src: str, dst: str, src_render: str):
    data = load_image(src, 4)
    normal_map = data[:, :, :3] * 256
    try:
        alpha_channel = load_image(src_render, 4)[:, :, 3]
        for i in range(alpha_channel.shape[0]):
            for j in range(alpha_channel.shape[1]):
                alpha_channel[i][j] = 256 if alpha_channel[i][j] > 0 else 0
        normal_map = np.concatenate(
            (normal_map, alpha_channel[:, :, np.newaxis]), axis=2
        )
    except:
        pass

    save_type = dst.split(".")[-1]
    Image.fromarray(normal_map.astype(np.uint8)).save(dst, save_type, quality=100)
    # normal_unit16 = normal_map.astype(np.uint16)
    # root = os.path.dirname(dst)
    # basename = os.path.basename(dst).split('.')[0]
    # save_path = os.path.join(root, basename+'.'+save_type)
    # cv2.imwrite(save_path, normal_unit16)

def load_image(file_path: str, num_channels: int = 3) -> np.ndarray:
    """Load the image at the given path returns its pixels as a numpy array.

    The alpha channel is neglected.

    :param file_path: The path to the image.
    :param num_channels: Number of channels to return.
    :return: The numpy array
    """
    file_ending = file_path[file_path.rfind(".") + 1 :].lower()
    if file_ending in ["exr", "png", "webp"]:
        return imageio.imread(file_path)[:, :, :num_channels]
    elif file_ending in ["jpg"]:
        import cv2

        img = cv2.imread(file_path)  # reads an image in the BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        raise NotImplementedError(
            "File with ending " + file_ending + " cannot be loaded."
        )

class SceneManager:
    @property
    def objects(self):
        return bpy.context.scene.objects

    @property
    def scene_meshes(self):
        return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    @property
    def scene_armatures(self):
        return [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]

    @property
    def data_meshes(self):
        return [obj for obj in bpy.data.objects if obj.type == "MESH"]

    @property
    def root_objects(self):
        for obj in bpy.context.scene.objects.values():
            if not obj.parent:
                yield obj

    @property
    def num_frames(self):
        return bpy.context.scene.frame_end + 1

    def get_scene_bbox(self, single_obj=None, ignore_matrix=False):
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3

        meshes = self.scene_meshes if single_obj is None else [single_obj]
        if len(meshes) == 0:
            raise RuntimeError("No objects in scene to compute bounding box for")

        for obj in meshes:
            for coord in obj.bound_box:
                coord = Vector(coord)
                if not ignore_matrix:
                    coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

        return Vector(bbox_min), Vector(bbox_max)

    def get_scene_bbox_all_frames(self):
        """Get bounding box that contains the entire animation sequence"""
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3

        # Store current frame
        current_frame = bpy.context.scene.frame_current

        # Iterate through all frames
        for frame in range(
            bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1
        ):
            bpy.context.scene.frame_set(frame)
            frame_min, frame_max = self.get_scene_bbox()
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, frame_min))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, frame_max))

        # Restore original frame
        bpy.context.scene.frame_set(current_frame)

        return Vector(bbox_min), Vector(bbox_max)

    def normalize_scene(
        self,
        normalize_range: float = 1.0,
        range_type: Literal["CUBE", "SPHERE"] = "CUBE",
        process_frames: bool = False,
        use_parent_node: bool = False,
    ):
        # Recompute bounding box, offset and scale
        if process_frames:
            bbox_min, bbox_max = self.get_scene_bbox_all_frames()
        else:
            bbox_min, bbox_max = self.get_scene_bbox()

        if range_type == "CUBE":
            scale = normalize_range / max(bbox_max - bbox_min)
        elif range_type == "SPHERE":
            scale = normalize_range / (bbox_max - bbox_min).length
        else:
            raise ValueError(
                f"Invalid range_type: {range_type}. Must be either 'CUBE' or 'SPHERE'"
            )

        # Calculate offset to center
        offset = -(bbox_min + bbox_max) / 2

        if use_parent_node:
            # Create a new empty object as parent
            parent = bpy.data.objects.new("NormalizationNode", None)
            bpy.context.scene.collection.objects.link(parent)

            # Parent all root objects to the new node
            for obj in self.root_objects:
                if obj is not parent:
                    obj.parent = parent
                    # Keep the object's local transform
                    obj.matrix_parent_inverse = parent.matrix_world.inverted()

            # Set parent's location and scale
            parent.scale = (scale, scale, scale)
            parent.location = offset * scale  # !!!important for use_parent_node!!!
        else:
            # Original behavior: modify each object directly
            for obj in self.root_objects:
                obj.matrix_world.translation += offset
                # Scale relative to world center by adjusting translation and scale
                original_translation = obj.matrix_world.translation.copy()
                obj.matrix_world.translation = original_translation * scale
                obj.scale = obj.scale * scale
                bpy.context.view_layer.update()

        # Restore original frame
        bpy.ops.object.select_all(action="DESELECT")

    def rotate_model(self, object, rotateQuaternion):
        object.select_set(True)
        bpy.context.view_layer.objects.active = object
        object.rotation_mode = "QUATERNION"
        object.rotation_quaternion = mathutils.Quaternion(rotateQuaternion)
        bpy.ops.object.transform_apply()

    def render(self):
        bpy.context.scene.render.use_compositing = True
        bpy.context.scene.use_nodes = True

        tree = bpy.context.scene.node_tree
        if "Render Layers" not in tree.nodes:
            tree.nodes.new("CompositorNodeRLayers")
        else:
            tree.nodes["Render Layers"]

        bpy.ops.render.render(animation=True, write_still=True)

    def smooth(self):
        for obj in self.scene_meshes:
            obj.data.use_auto_smooth = True
            obj.data.auto_smooth_angle = np.deg2rad(30)

    def clear_normal_map(self):
        for material in bpy.data.materials:
            material.use_nodes = True
            node_tree = material.node_tree
            try:
                bsdf = node_tree.nodes["Principled BSDF"]
                if bsdf.inputs["Normal"].is_linked:
                    for link in bsdf.inputs["Normal"].links:
                        node_tree.links.remove(link)
            except:
                pass

    def set_material_transparency(self, show_transparent_back: bool) -> None:
        """Set transparency settings for materials with blend mode 'BLEND'.

        Args:
            show_transparent_back: Whether to show the back face of transparent materials.
        """
        for material in bpy.data.materials:
            if not material.use_nodes:
                continue

            if material.blend_method == "BLEND":
                material.show_transparent_back = show_transparent_back

    def set_materials_opaque(self) -> None:
        """Set all materials to opaque blend mode.

        This is useful for rendering passes like normal maps that require
        fully opaque materials for correct results.
        """
        for material in bpy.data.materials:
            if not material.use_nodes:
                continue

            material.blend_method = "OPAQUE"

    def update_scene_frames(
        self, mode: Literal["auto", "manual"] = "auto", num_frames: Optional[int] = None
    ):
        if mode == "auto":
            armatures = self.scene_armatures
            keyframes = get_keyframes(armatures)
            bpy.context.scene.frame_end = (
                int(max(keyframes)) if len(keyframes) > 0 else 0
            )
        elif mode == "manual":
            if num_frames is None:
                raise ValueError(f"num_frames must be provided if the mode is 'manual'")
            bpy.context.scene.frame_end = num_frames - 1

    def clear(
        self,
        clear_objects: Optional[bool] = True,
        clear_nodes: Optional[bool] = True,
        reset_keyframes: Optional[bool] = True,
    ):
        if clear_objects:
            objects = [x for x in bpy.data.objects]
            for obj in objects:
                bpy.data.objects.remove(obj, do_unlink=True)

        # Clear all nodes
        if clear_nodes:
            bpy.context.scene.use_nodes = True
            node_tree = bpy.context.scene.node_tree
            for node in node_tree.nodes:
                node_tree.nodes.remove(node)

        # Reset keyframes
        if reset_keyframes:
            bpy.context.scene.frame_start = 0
            bpy.context.scene.frame_end = 0
            for a in bpy.data.actions:
                bpy.data.actions.remove(a)

    def gc(self):
        for _ in range(10):
            bpy.ops.outliner.orphans_purge()


