import blenderproc as bproc
import numpy as np
import cv2
import bpy
from blenderproc.python.renderer.RendererUtility import set_world_background
from blenderproc.python.renderer import RendererUtility
from blenderproc.python.types.MaterialUtility import Material
from blenderproc.python.material import MaterialLoaderUtility
from mathutils import Euler
import os
from tqdm import tqdm
import argparse


def calculate_fov(focal_lenghts, resolution):
    fx, fy = focal_lenghts
    image_width, image_height = resolution
    fov_x = 2 * np.arctan(image_width / (2 * fx))
    fov_y = 2 * np.arctan(image_height / (2 * fy))
    return fov_x, fov_y

def camera_poses(obj, focal_lenghts, resolution):
    poi = bproc.object.compute_poi(obj)
    bbox = obj[0].get_bound_box()
    poi = bbox.mean(axis=0)
    radius_bbox = np.linalg.norm(bbox, axis=1).max()  # Maximum bounding box radius
    fixed_radius = radius_bbox * 2  # Scale the radius if needed for consistent framing

    # Define camera locations on a sphere around the object
    top_z_multiplier = 1.25  # Adjust this multiplier for a steeper top-down view
    camera_positions = {
        "front": [fixed_radius, 0, poi[2]],
        "right": [0, fixed_radius, poi[2]],
        "back": [-fixed_radius, 0, poi[2]],
        "left": [0, -fixed_radius, poi[2]],
        "bottom": [0, 0, -fixed_radius],
        "top_0": [fixed_radius / np.sqrt(2), 0, fixed_radius * top_z_multiplier],
        "top_1": [-fixed_radius / 2, fixed_radius * np.sqrt(3) / 2, fixed_radius * top_z_multiplier],
        "top_2": [-fixed_radius / 2, -fixed_radius * np.sqrt(3) / 2, fixed_radius * top_z_multiplier],
    }

    locations = ["left", "front", "right", "back", "bottom", "top_0", "top_1", "top_2"]

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

def create_emission_material_from_texture(texture_path) -> Material:
    """Creates a material that uses an image texture as an emission."""
    emission_material = MaterialLoaderUtility.create("emission_texture_material")

    # Add an image texture node
    texture_node = emission_material.new_node("ShaderNodeTexImage")
    texture_node.image = bpy.data.images.load(texture_path)

    # Create an emission shader node
    emission_node = emission_material.new_node("ShaderNodeEmission")
    emission_material.link(texture_node.outputs["Color"], emission_node.inputs["Color"])

    # Link emission shader to output
    output_node = emission_material.get_the_one_node_with_type("OutputMaterial")
    emission_material.link(emission_node.outputs["Emission"], output_node.inputs["Surface"])

    return emission_material

def apply_emission_material_to_objects(obj_list, texture_path):
    """Apply the emission material to all objects in the scene."""
    if texture_path is not None:
        emission_material = create_emission_material_from_texture(texture_path)
    else:
        emission_material = create_emission_material_from_vertex_colors()

    # Get all mesh objects and apply the emission material
    for idx, obj in enumerate(obj_list):
        # Apply the emission material using BlenderProc's set_material method
        obj.set_material(idx, emission_material)

def renderering(obj_source_path, target_path):
    """Main rendering function where the material is not affected by light."""
    texture =  "ycbv" in obj_source_path and "nocs" not in obj_source_path

    if texture:
        texture_path = obj_source_path.replace(".ply", ".png")
    else:
        texture_path = None

    # Set output resolution
    image_resolution = (480, 480)
    bproc.camera.set_resolution(image_resolution[0], image_resolution[1])

    # Load the target object (your .ply file)
    obj = bproc.loader.load_obj(obj_source_path)
    if "lmo" in obj_source_path:
        obj[0].blender_obj.rotation_euler.rotate(Euler((0, 0, np.pi/2))) 
    elif "tyol" in obj_source_path:
        obj[0].blender_obj.rotation_euler.rotate(Euler((0, 0, np.pi)))
    elif "tless" in obj_source_path:
        obj[0].blender_obj.rotation_euler.rotate(Euler((0, 0, np.pi))) 
    elif "tudl" in obj_source_path:
        obj[0].blender_obj.rotation_euler.rotate(Euler((0, 0, np.pi/2))) 
    elif "icbin" in obj_source_path:
        obj[0].blender_obj.rotation_euler.rotate(Euler((0, 0, -np.pi/2))) 
    elif "ycbv" in obj_source_path:
        obj[0].blender_obj.rotation_euler.rotate(Euler((0, 0, 0))) 
    else:
        raise ValueError("Chosen dataset is not supported or tested")
    obj[0].move_origin_to_bottom_mean_point()

    # Define camera intrinsics (as in your original code)
    fx, fy = 800, 800
    cx, cy = image_resolution[0] / 2, image_resolution[1] / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    bproc.camera.set_intrinsics_from_K_matrix(K, image_resolution[0], image_resolution[1])

    camera_poses(obj, (fx, fy), image_resolution)

    # Apply the emission material to all objects
    apply_emission_material_to_objects(obj, texture_path)

    RendererUtility.render_init()
    RendererUtility.set_max_amount_of_samples(1)
    RendererUtility.set_noise_threshold(0)
    RendererUtility.set_denoiser(None)
    RendererUtility.set_light_bounces(1, 0, 0, 1, 0, 8, 0)
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

def render_loop(dataset_path, dataset):
    bproc.init()

    source_base_path_nocs = os.path.join(dataset_path, f"{dataset}_models", "models_nocs")
    target_base_path_nocs = os.path.join(dataset_path, f"{dataset}_views", "nocs")

    source_base_path_rgb = os.path.join(dataset_path, f"{dataset}_models", "models")
    target_base_path_rgb = os.path.join(dataset_path, f"{dataset}_views", "rgb")

    # create a log file txt, if already exists append to it
    # log_path = os.path.join(base_path, "logs", "rendering_log.txt")
    # os.makedirs(os.path.dirname(log_path), exist_ok=True)

    for object_id in tqdm(sorted(os.listdir(os.path.join(source_base_path_nocs))), desc="object_ids"):
        obj_source_path_nocs = os.path.join(source_base_path_nocs, object_id, f"model_scaled_nocs.ply")
        target_path_nocs = os.path.join(target_base_path_nocs, object_id)
        # if os.path.exists(target_path_nocs):
        #    continue
        renderering(obj_source_path_nocs, target_path_nocs)

        obj_source_path_rgb = os.path.join(source_base_path_rgb, f"obj_{object_id}.ply")
        target_path_rgb = os.path.join(target_base_path_rgb, object_id)
        # if os.path.exists(target_path_rgb):
        #     continue
        renderering(obj_source_path_rgb, target_path_rgb)

        # with open(log_path, "a") as f:
        #     f.write(f"{github_id}_{objaverse_id}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tyol')
    args = parser.parse_args()

    base_path = "/cluster/work/riner/users/simschla/datasets/"

    dataset = args.dataset

    dataset_path = os.path.join(base_path, dataset)

    print("Rendering for", dataset)

    render_loop(dataset_path, dataset)
    print("Done")

