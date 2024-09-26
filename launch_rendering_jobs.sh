#!/bin/bash

for i in {0..99}; do
    # Submit sbatch job
    sbatch --cpus-per-task=4 --mem-per-cpu=40G --time=20:00:00 --wrap="blenderproc run main_objaverse_random_cluster.py --seed $i --start_scene_id $i --num_scenes_per_node 25 --total_num_objects 750"
done
