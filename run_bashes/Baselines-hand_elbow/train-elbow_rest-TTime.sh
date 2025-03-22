#!/bin/bash

# Define the workspace folder
workspace_folder="/home/jyt/workspace/DeepTransferBCI"

# Change the current directory to the workspace folder
cd $workspace_folder

# Define the program path
progarm_path="tl/ttime_debug_m_cls.py"

# Set the PYTHONPATH environment variable
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Execute the Python script with the specified arguments
python3 $progarm_path \
  --dataset_name "MI-elbow_rest" \
  --data_save "True" \
  --data_path "./data/" \
  --data_path_MI "/home/jyt/workspace/transfer_models/datasets_MI/hand_elbow/derivatives" \
  --log_path "./logs/ttime-debug-elbow_rest-momentum-cls-1/" \
  --use_pretrained_model "True" \
  --finetune "False" \
  --momentum "False" \
  --momentum_param "0.5"
