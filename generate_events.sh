#!/bin/bash

ROOT=/data/tim/ecddp/dataset

# Loop through each environment folder and run generate_event.py
for env in $(find $ROOT -mindepth 2 -maxdepth 2 -type d)
do
    python ./generate_data/generate_event.py --input $env --temp_folder ./temp
done

# Process events
python3 ./generate_data/split_event_chunks.py --file_template ${ROOT}/**/event_left.h5

# Get file paths
python3 ./generate_data/get_file_path.py --dataset_root ${ROOT}