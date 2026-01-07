#!/bin/bash

# Define the workspace folder
workspace_folder="/home/jyt/workspace/DeepTransferBCI"
# Change the current directory to the workspace folder
cd $workspace_folder

# Define the program path
progarm_path="tl/proposed_method/ours_debug_m_cls_process_18.py"

# Set the PYTHONPATH environment variable
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Execute the Python script with the specified arguments
python3 $progarm_path \
  --dataset_name "BNCI2014001-4-all" \
  --data_save "True" \
  --data_path "./data/" \
  --data_path_MI "None" \
  --log_path "./logs/Baselines-001-test-e300-b64-debugs/ours_debug_m_cls_process_1-BNCI2014001-4-all-EEGNet-4,2-e300-b64/proposed_50_BNoff_batch8stride8_CE_KL_lcs_ConsSamples_selection_twoStage_weighted_lr0.005-1-1/" \
  --use_pretrained_model "True" \
  --finetune "False" \
  --momentum "False" \
  --momentum_param "0.5" \
  --gpu_idx "1" \
  --align "True" \
  --batch_size "64" \
  --lr "0.001"  \
  --lr_online "0.005" \
  --epoch "300"  \
  --backbone "EEGNet-4,2" \
  --param_runs "./runs_debug/" \
  --use_BN "False" \
  --stride "8" \
  --batch_size_online "8" \
  --loss_func "CE_KL,ConsSamples_selection_two_stage_weighted" \
  --selection_ratio "1.0" \
  --mt "0.1" \
  --updating_type "entropy" \
  --loss_weights "1.0, 1.0, 1.0" \
  