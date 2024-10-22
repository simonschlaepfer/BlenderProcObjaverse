#!/bin/bash

# Loop from 0 to 9
for i in {101..162}; do
    # Format github_id to have 3 digits with leading zeros
    shard_id=$(printf "%03d" $i)
    # Submit sbatch job
    sbatch --mem-per-cpu=50G --time=60:00:00 --wrap="blenderproc run render_views_batch.py --github_id $shard_id"
done