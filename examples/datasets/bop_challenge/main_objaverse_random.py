import blenderproc as bproc
import argparse
import os
import numpy as np
import random
import multiprocessing
import math
import json
from blenderproc.python.camera import CameraUtility

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', nargs='?', default='', help="Path to the bop datasets parent directory")
parser.add_argument('cc_textures_path', nargs='?', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', nargs='?', default='', help="Path to where the final files will be saved ")
parser.add_argument('--num_scenes', nargs='?', type=int, default=10, help="How many scenes with 25 images each to generate")
args = parser.parse_args()

args.bop_parent_path = '/cluster/work/riner/users/simschla/datasets'
args.cc_textures_path = '/cluster/work/riner/users/simschla/datasets/textures'
objaverse_base_path = '/cluster/work/riner/users/simschla/datasets/objapose_base/objaverse'
objaverse_model_json_path = os.path.join(objaverse_base_path, 'objaverse_models.json')
args.output_dir = os.path.join(objaverse_base_path, 'objaverse_scenes')
args.num_scenes = 2
sample_size = 4
total_num_objects = 50
dataset_name = 'objapose'
image_resolution = (720, 540)
focal_interval = (500, 3000)
diff_focal_interval = (1, 50)

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

bproc.init()

# bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, 'ycbv'))

bproc.loader.init_bop_toolkit(bop_dataset_path=os.path.join(args.bop_parent_path, dataset_name))

# load bop objects into the scene
import objaverse
processes = multiprocessing.cpu_count()
random.seed(42)
np.random.seed(42)
uids = objaverse.load_uids()
random_object_uids = random.sample(uids, total_num_objects)
objects = objaverse.load_objects(
    uids=random_object_uids,
    download_processes=1
)
objects_dict = objaverse.load_objects(uids=random_object_uids)

objaverse_dict = bproc.loader.update_json(objects_dict, objaverse_model_json_path)

target_bop_objs, objaverse_dict = bproc.loader.load_objaverse_objs(objaverse_dict, objaverse_base_path, object_model_unit='dm')

# save objaverse_dict at objaverse_model_json_path
with open(objaverse_model_json_path, 'w') as file:
    json.dump(objaverse_dict, file, indent=4)

# load distractor bop objects
tless_dist_bop_objs = [] # bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'tless'), model_type = 'cad', mm2m = True)
hb_dist_bop_objs = [] # bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'hb'), mm2m = True)
tyol_dist_bop_objs = [] # bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'tyol'), mm2m = True)

print("Objects loaded into Blender")

# set shading and hide objects
for obj in (target_bop_objs + tless_dist_bop_objs + hb_dist_bop_objs + tyol_dist_bop_objs):
    obj.set_shading_mode('auto')
    obj.hide(True)

print("Create Room")
# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

print("Create Light")
# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(200)

print("Load Textures")
# load cc_textures
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

print("Setup pose sampler and depth renderer")
# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())
    
# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False, convert_to_distance=False)
bproc.renderer.set_max_amount_of_samples(50)

print("Blender scene setup finished")

for i in range(args.num_scenes):
    # setup camera with random intrinsics
    # create random camera intrinsics as np array
    fx = np.random.uniform(*focal_interval)
    fy = fx + np.random.uniform(*diff_focal_interval)
    w, h = image_resolution
    cx, cy = w / 2, h / 2
    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ]
    )
    cam_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    CameraUtility.set_intrinsics_from_K_matrix(cam_K, image_resolution[0], image_resolution[1])

    # Sample bop objects for a scene
    sampled_target_bop_objs = list(np.random.choice(target_bop_objs, size=sample_size, replace=False))
    sampled_distractor_bop_objs = list(np.random.choice(tless_dist_bop_objs, size=0, replace=False))
    sampled_distractor_bop_objs += list(np.random.choice(hb_dist_bop_objs, size=0, replace=False))
    sampled_distractor_bop_objs += list(np.random.choice(tyol_dist_bop_objs, size=0, replace=False))


    # Randomize materials and set physics
    for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):
        if len(obj.get_materials()) > 0:        
            mat = obj.get_materials()[0]
            if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
                grey_col = np.random.uniform(0.1, 0.9)   
                mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])        
            mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
            mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0, 1.0))
            mat.set_principled_shader_value("Metallic", np.random.uniform(0, 0.5))
        obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        obj.hide(False)
    
    # Sample two light sources
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                    emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
    light_point.set_location(location)

    # sample CC Texture and assign to room planes
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)


    # Sample object poses and check collisions 
    bproc.object.sample_poses(objects_to_sample = sampled_target_bop_objs + sampled_distractor_bop_objs,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000)
            
    # Physics Positioning
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                    max_simulation_time=10,
                                                    check_object_interval=1,
                                                    substeps_per_frame = 20,
                                                    solver_iters=25)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs + sampled_distractor_bop_objs)

    cam_poses = 0
    while cam_poses < 25:
        # Sample location
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 0.61,
                                radius_max = 1.24,
                                elevation_min = 5,
                                elevation_max = 89)
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=int(math.ceil(0.7*sample_size)), replace=False))
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    # render the whole pipeline
    data = bproc.renderer.render()

    # data["depth"] = np.array(scaled_depth_list)


    # # convert from float64 to uint16
    # # float64_max = np.finfo(np.float64).max
    # uint16_max = np.iinfo(np.uint16).max
    # # print("simon max", float64_max, uint16_max)
    # data["depth"][0] = ((data["depth"][0] / 268.0) * uint16_max).astype(np.uint16)
    # print("simon depth_uint8", data["depth"][0].max(), data["depth"][0].min(), data["depth"][0].dtype, data["depth"][0])

    print("BOP writer started")
    # Write data in bop formatx
    bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                           target_objects = sampled_target_bop_objs,
                           dataset = dataset_name,
                           depth_scale = 0.1,
                           depths = data["depth"],
                           colors = data["colors"], 
                           color_file_format = "JPEG",
                           ignore_dist_thres = 10,
                           frames_per_chunk=25)
    
    for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):      
        obj.disable_rigidbody()
        obj.hide(True)

    print("Scene finished")
