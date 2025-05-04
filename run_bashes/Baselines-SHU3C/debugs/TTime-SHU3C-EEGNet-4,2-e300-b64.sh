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
  --dataset_name "WBCIC-SHU-3C" \
  --data_save "True" \
  --data_path "./data/" \
  --data_path_MI "/data/datasets_Jyt/WBCIC_SHU_3C/processeddata/processeddata/" \
  --log_path "./logs/deubgs-WBCIC-SHU-3C-process/ttime-debug-WBCIC-SHU-3C-momentum-cls-1-test-EEGNet-4,2/" \
  --use_pretrained_model "False" \
  --finetune "False" \
  --momentum "False" \
  --momentum_param "0.5" \
  --gpu_idx "1" \
  --align "True" \
  --batch_size "64" \
  --batch_size_online "8" \
  --lr "0.001"  \
  --lr_online "0.0001" \
  --epoch "300"  \
  --backbone "EEGNet-4,2" \
  --param_runs "./runs_debug/"