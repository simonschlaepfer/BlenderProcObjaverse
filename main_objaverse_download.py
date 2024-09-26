import os
import numpy as np
import random
import multiprocessing
import math
import json
import objaverse_json

objaverse_base_path = '/cluster/work/riner/users/simschla/datasets/objapose_base/objaverse'
objaverse_model_json_path = os.path.join(objaverse_base_path, 'objaverse_models.json')
total_num_objects = 50000
seed = 42

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

print("Loading objects")

# load bop objects into the scene
import objaverse
processes = multiprocessing.cpu_count()
random.seed(seed)
np.random.seed(seed)
uids = objaverse.load_uids()
random_object_uids = random.sample(uids, total_num_objects)
objects = objaverse.load_objects(
    uids=random_object_uids,
    download_processes=8
)
objects_dict = objaverse.load_objects(uids=random_object_uids)

print("Objects loaded")

objaverse_json.update_json(objects_dict, objaverse_base_path, objaverse_model_json_path)

print("Done")