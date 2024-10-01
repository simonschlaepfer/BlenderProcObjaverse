#!/bin/bash

for i in {200..299}; do
    # Submit sbatch job
    sbatch --cpus-per-task=2 --mem-per-cpu=30G --time=24:00:00 --wrap="blenderproc run main_objaverse_random_cluster.py --seed $i --start_scene_id $i --num_scenes_per_node 25 --total_num_objects 1000"
done
