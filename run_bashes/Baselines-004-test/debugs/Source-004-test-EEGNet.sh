#!/bin/bash

# Define the workspace folder
workspace_folder="/home/jyt/workspace/DeepTransferBCI"
# Change the current directory to the workspace folder
cd $workspace_folder

# Define the program path
progarm_path="tl/source.py"

# Set the PYTHONPATH environment variable
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Execute the Python script with the specified arguments
python3 $progarm_path \
  --dataset_name "BNCI2014_004-test" \
  --data_save "True" \
  --data_path "./data/" \
  --data_path_MI "None" \
  --log_path "./logs/Source-BNCI2014_004-test-EEGNet/" \
  --use_pretrained_model "True" \
  --finetune "False" \
  --momentum "False" \
  --momentum_param "0.5" \
  --gpu_idx "0" \
  --align "True" \
  --batch_size "32" \
  --lr "0.001"  \
  --epoch "100"  \
  --backbone "EEGNet"