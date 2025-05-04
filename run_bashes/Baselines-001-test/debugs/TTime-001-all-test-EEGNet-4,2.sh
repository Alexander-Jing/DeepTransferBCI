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
  --dataset_name "BNCI2014001-4-test" \
  --data_save "True" \
  --data_path "./data/" \
  --data_path_MI "None" \
  --log_path "./logs/ttime-BNCI2014001-test-EEGNet-4,2/" \
  --use_pretrained_model "False" \
  --finetune "False" \
  --momentum "False" \
  --momentum_param "0.5" \
  --gpu_idx "1" \
  --align "True" \
  --batch_size "32" \
  --lr "0.001"  \
  --epoch "100"  \
  --backbone "EEGNet-4,2"