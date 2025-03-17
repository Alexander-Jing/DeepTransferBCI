#!/bin/bash

# Define the workspace folder
workspace_folder="/home/jyt/workspace/DeepTransferBCI"

# Change the current directory to the workspace folder
cd $workspace_folder

# Set the PYTHONPATH environment variable
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Execute the Python script with the specified arguments
python3 "${workspace_folder}/download_data_1.py" \
  --dataset_name "Schirrmeister2017" \
  --data_save "True" \
  --data_path "./data/"