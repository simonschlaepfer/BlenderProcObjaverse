import json
import os
from typing import Optional
import trimesh
import numpy as np
from tqdm import tqdm

def load_existing_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []

def save_to_json(file_path, data):
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def find_next_available_id(existing_data):
    used_ids = {item["obj_id"] for item in existing_data}
    new_id = 0
    while new_id in used_ids:
        new_id += 1
    return new_id

def convert_glb_to_obj(glb_path: str, obj_path: str):
    # Load the mesh or collection of meshes
    try:
        trimesh_mesh = trimesh.load(glb_path)
    except:
        return None
    # Check if the loaded object is a scene (which can contain multiple geometries)
    if isinstance(trimesh_mesh, trimesh.Scene):
        for name, geom in trimesh_mesh.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                print(f"Skipping {glb_path} as it is not a valid Trimesh")
                return None
        merged_mesh = trimesh.util.concatenate(trimesh_mesh.dump())
    else:
        # If trimesh_mesh is already a single mesh, no need to concatenate
        merged_mesh = trimesh_mesh

    # the mesh can be 50% smaller or bigger
    scale_randomizer = np.random.uniform(0.5, 2.0)

    norm_scale = max(trimesh_mesh.bounding_box.extents) * scale_randomizer

    merged_mesh.apply_scale(1 / norm_scale)
    merged_mesh.export(obj_path, include_texture=True, file_type='obj')

    return norm_scale

def update_json(objaverse_dict, objaverse_base_path, file_path='/meshes/objaverse_models.json'):
    # Load existing data if it exists
    existing_data = load_existing_json(file_path)
    
    # Create a map for quick lookup of existing objaverse_id
    existing_map = {item["objaverse_id"]: item["obj_id"] for item in existing_data}

    for idx, (objaverse_id, path) in enumerate(tqdm(objaverse_dict.items(), desc="Processing Objects")):
        if objaverse_id not in existing_map:
            github_id = path.split('/')[-2]
            # Find the next available obj_id
            glb_path = os.path.join(objaverse_base_path, "glbs", github_id, objaverse_id + '.glb')
            obj_path = os.path.join(objaverse_base_path, "objs", github_id, objaverse_id, 'model_scaled.obj')
            obj_path_dir = os.path.dirname(obj_path)
            os.makedirs(obj_path_dir, exist_ok=True)
            norm_scale = convert_glb_to_obj(glb_path, obj_path)
            if norm_scale is None:
                print(f"{glb_path} could not be converted to .obj")
                continue
            else:
                new_id = find_next_available_id(existing_data)
                existing_data.append({
                    "obj_id": new_id,
                    "objaverse_id": objaverse_id,
                    "github_id": github_id,
                    "norm_scale": norm_scale
                })
    
                # Save updated data back to the JSON file
                if idx % 250 == 0:
                    save_to_json(file_path, existing_data)

    save_to_json(file_path, existing_data)
    return
