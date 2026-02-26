#!/bin/bash

# Define the workspace folder
workspace_folder="/home/jyt/workspace/DeepTransferBCI"
# Change the current directory to the workspace folder
cd $workspace_folder

# Define the program path
progarm_path="tl/rotta.py"

# Set the PYTHONPATH environment variable
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Execute the Python script with the specified arguments
python3 $progarm_path \
  --dataset_name "BNCI2014_004-test" \
  --data_save "True" \
  --data_path "./data/" \
  --data_path_MI "None" \
  --log_path "./logs/Baselines-004-test-e300-b64/rotta-BNCI2014_004-test-EEGNet-4,2-e300-b64-lr_online0.001-2/" \
  --use_pretrained_model "True" \
  --finetune "False" \
  --momentum "False" \
  --momentum_param "0.5" \
  --gpu_idx "1" \
  --align "True" \
  --batch_size "64" \
  --batch_size_online "1" \
  --lr "0.001"  \
  --lr_online "0.001" \
  --epoch "300"  \
  --backbone "EEGNet-4,2" \
  --param_runs "./runs_debug/"