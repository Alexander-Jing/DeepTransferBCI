#!/bin/bash

# Define the workspace folder
workspace_folder="/home/jyt/workspace/DeepTransferBCI"
# Change the current directory to the workspace folder
cd $workspace_folder

# Define the program path
progarm_path="tl/ttime_debug_m_cls_process.py"

# Set the PYTHONPATH environment variable
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Execute the Python script with the specified arguments
python3 $progarm_path \
  --dataset_name "BNCI2014001-4-all" \
  --data_save "True" \
  --data_path "./data/" \
  --data_path_MI "None" \
  --log_path "./logs/Baselines-001-all-e200-b64/ttime-BNCI2014001-all-EEGNet-4,2-e200-b64/" \
  --use_pretrained_model "False" \
  --finetune "False" \
  --momentum "False" \
  --momentum_param "0.5" \
  --gpu_idx "1" \
  --align "True" \
  --batch_size "64" \
  --lr "0.001"  \
  --epoch "200"  \
  --backbone "EEGNet-4,2" \
  --param_runs "./runs_debug/"