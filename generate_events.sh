#!/bin/bash

# ROOT=/data/tim/ecddp/dataset
ROOT=/data/tim/datasets/E-TartanAir

# Check if current conda environment is "v2e"
if [ "$CONDA_DEFAULT_ENV" != "/data/conda_envs/v2e" ]; then
    echo "Please activate the 'v2e' conda environment before running this script."
    exit 1
fi

# Loop through each environment folder and run generate_event.py
for env in $(find $ROOT -mindepth 2 -maxdepth 2 -type d)
do
    else
        echo "############################################################################"
        echo -e "\n\n"
        echo "Generating events for $env"
        echo -e "\n\n"
        echo "############################################################################"
        python ./generate_data/generate_event.py --input $env --temp_folder ./temp
    fi
done

# Process events
python3 ./generate_data/split_events_chunks.py --file_template ${ROOT}/**/event_left.h5

# Precompute bins
python3 ./data/create_bins.py

# Get file paths
python3 ./generate_data/get_file_path.py --dataset_root ${ROOT}