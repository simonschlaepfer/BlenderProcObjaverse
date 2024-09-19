import json
import os

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

def update_json(objaverse_dict, file_path='/meshes/objaverse_models.json'):
    # Load existing data if it exists
    existing_data = load_existing_json(file_path)
    
    # Create a map for quick lookup of existing objaverse_id
    existing_map = {item["objaverse_id"]: item["obj_id"] for item in existing_data}

    for objaverse_id, path in objaverse_dict.items():
        if objaverse_id not in existing_map:
            # Find the next available obj_id
            new_id = find_next_available_id(existing_data)
            github_id = path.split('/')[-2]
            existing_data.append({
                "obj_id": new_id,
                "objaverse_id": objaverse_id,
                "github_id": github_id,
            })
    
    # Save updated data back to the JSON file
    save_to_json(file_path, existing_data)
    return existing_data
