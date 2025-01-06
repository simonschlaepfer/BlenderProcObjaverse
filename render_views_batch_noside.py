import blenderproc as bproc
import numpy as np
import cv2
import bpy
from blenderproc.python.renderer.RendererUtility import set_world_background
from blenderproc.python.renderer import RendererUtility
from blenderproc.python.types.MaterialUtility import Material
from blenderproc.python.material import MaterialLoaderUtility
from mathutils import Euler, Matrix
import os
from tqdm import tqdm
import argparse


def calculate_fov(focal_lenghts, resolution):
    fx, fy = focal_lenghts
    image_width, image_height = resolution
    fov_x = 2 * np.arctan(image_width / (2 * fx))
    fov_y = 2 * np.arctan(image_height / (2 * fy))
    return fov_x, fov_y

def camera_poses(obj, nocs, focal_lenghts, resolution):
    poi = bproc.object.compute_poi(obj)
    bbox = obj[0].get_bound_box()
    poi = bbox.mean(axis=0)
    radius_bbox = np.linalg.norm(bbox, axis=1).max()  # Maximum bounding box radius
    fov_x, fov_y = calculate_fov(focal_lenghts, resolution)
    fixed_radius = radius_bbox / np.tan(min(fov_x, fov_y) / 2)

    # Define camera locations on a sphere around the object
    top_z_multiplier = 1.0  # Adjust this multiplier for a steeper top-down view
    camera_positions = {
        "top_0": [fixed_radius / np.sqrt(2), 0, (fixed_radius * top_z_multiplier)-poi[2]],
        "top_1": [0, fixed_radius / np.sqrt(2), (fixed_radius * top_z_multiplier)-poi[2]],
        "top_2": [- fixed_radius / np.sqrt(2), 0, (fixed_radius * top_z_multiplier)-poi[2]], 
        "top_3": [0, -fixed_radius / np.sqrt(2), (fixed_radius * top_z_multiplier)-poi[2]],   
        "bottom_0": [fixed_radius / np.sqrt(3), fixed_radius / np.sqrt(3), (- fixed_radius * top_z_multiplier)+2*poi[2]],
        "bottom_1": [-fixed_radius / np.sqrt(3), fixed_radius / np.sqrt(3), (- fixed_radius * top_z_multiplier)+2*poi[2]],
        "bottom_2": [fixed_radius / np.sqrt(3), -fixed_radius / np.sqrt(3), (- fixed_radius * top_z_multiplier)+2*poi[2]], 
        "bottom_3": [-fixed_radius / np.sqrt(3), -fixed_radius / np.sqrt(3), (- fixed_radius * top_z_multiplier)+2*poi[2]],      
    }
    # camera_positions = {
    #     "top_0": (bbox[2]+bbox[3])/2,
    #     "top_1": (bbox[3]+bbox[7])/2,
    #     "top_2": (bbox[7]+bbox[6])/2,
    #     "top_3": (bbox[6]+bbox[2])/2,
    #     "bottom_0": bbox[0],
    #     "bottom_1": bbox[1],
    #     "bottom_2": bbox[4],
    #     "bottom_3": bbox[5],  
    # }

    if not nocs:
        light_point = bproc.types.Light()
        light_point.set_type("POINT")
        light_point.set_energy(200)
        light_point.set_radius(1.0)
        light_point.set_color(np.array([1, 1, 1]))

        # Add sunlight for global directional lighting
        # sunlight = bproc.types.Light()
        # sunlight.set_type("SUN")
        # sunlight.set_energy(2)  # Energy for the sun light
        # sunlight.set_color(np.array([1, 1, 1]))

    locations = ["top_0", "top_1", "top_2", "top_3", "bottom_0", "bottom_1", "bottom_2", "bottom_3"]

    for idx, location in enumerate(locations):
        cam_pos = camera_positions[location]
        # Compute rotation and camera matrix
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - cam_pos)
        cam2world_matrix = bproc.math.build_transformation_mat(cam_pos, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix, frame=idx)

        # Fit the camera to the object to ensure proper framing
        cam = bpy.data.objects["Camera"]
        bpy.ops.object.select_all(action='DESELECT')
        obj[0].blender_obj.select_set(True)
        bpy.ops.view3d.camera_to_view_selected()
        cam.keyframe_insert(data_path='location', frame=idx)
        cam.keyframe_insert(data_path='rotation_euler', frame=idx)

        # Add light at the same position as the camera for consistent lighting
        if not nocs:
            light_point.set_location(cam_pos, frame=idx)

            # rotation_matrix_blender = Matrix(rotation_matrix)
            # euler_angles = rotation_matrix_blender.to_euler()  # Convert to Euler angles
            # sunlight.set_rotation_euler(np.array([euler_angles.x, euler_angles.y, euler_angles.z]), frame=idx)


def create_emission_material_from_vertex_colors() -> Material:
    """Creates a material that uses vertex colors directly and emits them without lighting effects."""
    emission_material = MaterialLoaderUtility.create("emission_material")

    # Create nodes for the emission shader
    vertex_color_node = emission_material.new_node("ShaderNodeVertexColor")
    emission_node = emission_material.new_node("ShaderNodeEmission")
    
    # Link vertex colors to the emission shader
    emission_material.link(vertex_color_node.outputs["Color"], emission_node.inputs["Color"])

    # Link the emission shader to the output node
    output_node = emission_material.get_the_one_node_with_type('OutputMaterial')
    emission_material.link(emission_node.outputs["Emission"], output_node.inputs['Surface'])

    return emission_material

def apply_emission_material_to_objects(obj_list):
    """Apply the emission material to all objects in the scene."""
    emission_material = create_emission_material_from_vertex_colors()

    # Get all mesh objects and apply the emission material
    for idx, obj in enumerate(obj_list):
        # Apply the emission material using BlenderProc's set_material method
        obj.set_material(idx, emission_material)

def renderering(obj_source_path, target_path):
    """Main rendering function where the material is not affected by light."""
    nocs = "nocs" in obj_source_path

    # Set output resolution
    image_resolution = (480, 480)
    bproc.camera.set_resolution(image_resolution[0], image_resolution[1])

    # Load the target object (your .ply file)
    obj = bproc.loader.load_obj(obj_source_path)
    if not nocs:
        obj[0].set_shading_mode('auto')
    else:
        obj[0].blender_obj.rotation_euler.rotate(Euler((np.pi/2, 0, 0)))
    obj[0].move_origin_to_bottom_mean_point()

    # Define camera intrinsics (as in your original code)
    fx, fy = 800, 800
    cx, cy = image_resolution[0] / 2, image_resolution[1] / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    bproc.camera.set_intrinsics_from_K_matrix(K, image_resolution[0], image_resolution[1])

    camera_poses(obj, nocs, (fx, fy), image_resolution)

    # Apply the emission material to all objects
    if nocs:
        apply_emission_material_to_objects(obj)

    RendererUtility.render_init()
    if nocs:
        RendererUtility.set_max_amount_of_samples(1)
        RendererUtility.set_noise_threshold(0)
        RendererUtility.set_denoiser(None)
        RendererUtility.set_light_bounces(1, 0, 0, 1, 0, 8, 0)
        bpy.context.scene.cycles.filter_width = 0.0
    else:
        RendererUtility.set_max_amount_of_samples(32)
        RendererUtility.set_noise_threshold(0.01)
        RendererUtility.set_denoiser("INTEL")
        RendererUtility.set_light_bounces(3, 0, 3, 3, 0, 8, 0)
        bpy.context.scene.cycles.filter_width = 0.0

    # Render settings
    bproc.renderer.set_output_format('PNG', enable_transparency=True, view_transform="Standard")

    # bproc.renderer.set_world_background([0, 0, 0])
    data = bproc.renderer.render()

    # Save rendered images
    colors = data["colors"]

    os.makedirs(target_path, exist_ok=True)
    
    for idx, color_rgba in enumerate(colors):
        # Extract the alpha channel
        alpha_channel = color_rgba[..., 3]

        # Create a binary mask: all alpha > 0 set to 1, alpha = 0 set to 0
        mask = np.where(alpha_channel > 0, 1, 0).astype(np.uint8)
        mask = np.expand_dims(mask, axis=-1)

        color_rgb = color_rgba[..., :3] * mask
        
        # Convert RGB to BGR for OpenCV
        color_bgr = color_rgb[..., ::-1]
        
        # Save the image
        save_path = os.path.join(target_path, f"{idx}.png")
        cv2.imwrite(save_path, color_bgr)
    
    # Clear the scene
    bproc.clean_up()

def render_loop(base_path, github_id):
    bproc.init()

    source_base_path_nocs = os.path.join(base_path, "objaverse", "nocs")
    target_base_path_nocs = os.path.join(base_path, "objaverse_views_noside", "nocs")

    source_base_path_rgb = os.path.join(base_path, "objaverse", "objs")
    target_base_path_rgb = os.path.join(base_path, "objaverse_views_noside", "rgb")

    # create a log file txt, if already exists append to it
    log_path = os.path.join(base_path, "logs", "rendering_log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    for objaverse_id in tqdm(sorted(os.listdir(os.path.join(source_base_path_nocs, github_id))), desc="objaverse_ids"):
        # obj_source_path_nocs = os.path.join(source_base_path_nocs, github_id, objaverse_id, f"model_scaled_nocs.ply")
        # target_path_nocs = os.path.join(target_base_path_nocs, github_id, objaverse_id)
        # if os.path.exists(os.path.join(target_path_nocs, "7.png")):
        #    continue
        # renderering(obj_source_path_nocs, target_path_nocs)

        obj_source_path_rgb = os.path.join(source_base_path_rgb, github_id, objaverse_id, f"model_scaled.obj")
        target_path_rgb = os.path.join(target_base_path_rgb, github_id, objaverse_id)
        if os.path.exists(os.path.join(target_path_rgb, "7.png")):
            continue
        renderering(obj_source_path_rgb, target_path_rgb)

        with open(log_path, "a") as f:
            f.write(f"{github_id}_{objaverse_id}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--github_id', type=int, default=85)
    args = parser.parse_args()

    base_path = "/cluster/work/riner/users/simschla/datasets/objapose_base"

    github_number = str(args.github_id).zfill(3)

    github_id = f"000-{github_number}"

    print("Rendering for", github_id)

    render_loop(base_path, github_id)
    print("Done")

