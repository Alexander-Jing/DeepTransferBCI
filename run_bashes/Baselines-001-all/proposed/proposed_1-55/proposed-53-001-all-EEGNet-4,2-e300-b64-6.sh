#!/bin/bash

# Define the workspace folder
workspace_folder="/home/jyt/workspace/DeepTransferBCI"
# Change the current directory to the workspace folder
cd $workspace_folder

# Define the program path
progarm_path="tl/proposed_method/ours_debug_m_cls_process_25.py"

# Set the PYTHONPATH environment variable
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Execute the Python script with the specified arguments
python3 $progarm_path \
  --dataset_name "BNCI2014001-4-all" \
  --data_save "True" \
  --data_path "./data/" \
  --data_path_MI "None" \
  --log_path "./logs/Baselines-001-test-e300-b64-debugs/ours_debug_m_cls_process_1-BNCI2014001-4-all-EEGNet-4,2-e300-b64/proposed_53_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-6/" \
  --use_pretrained_model "True" \
  --finetune "False" \
  --momentum "False" \
  --momentum_param "0.5" \
  --gpu_idx "0" \
  --align "True" \
  --batch_size "64" \
  --lr "0.001"  \
  --lr_online "0.001" \
  --epoch "300"  \
  --backbone "EEGNet-4,2" \
  --param_runs "./runs_debug/" \
  --use_BN "False" \
  --stride "8" \
  --batch_size_online "8" \
  --loss_func "CE_KL_review_weighted_5,ConsSamples_selection_two_stage_weighted_4_1" \
  --selection_ratio "0.75" \
  --mt "0.9" \
  --updating_type "entropy_review" \
  --loss_weights "1.0, 1.0, 1.0" \
  --scale "10.0" \