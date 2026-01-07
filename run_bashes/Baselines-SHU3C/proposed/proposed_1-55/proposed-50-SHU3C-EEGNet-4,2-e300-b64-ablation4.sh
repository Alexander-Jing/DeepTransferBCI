#!/bin/bash

# Define the workspace folder
workspace_folder="/home/jyt/workspace/DeepTransferBCI"
# Change the current directory to the workspace folder
cd $workspace_folder

# Define the program path
progarm_path="tl/proposed_method/ours_debug_m_cls_process_20.py"

# Set the PYTHONPATH environment variable
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Execute the Python script with the specified arguments
python3 $progarm_path \
  --dataset_name "WBCIC-SHU-3C" \
  --data_save "True" \
  --data_path "./data/" \
  --data_path_MI "/data/datasets_Jyt/WBCIC_SHU_3C/processeddata/processeddata/" \
  --log_path "./logs/Baselines-WBCIC-SHU-3C-e300-b64/proposed/proposed_50_BNoff_batch8stride8_CE_KL_lcs_ConsSamples_selection_twoStage_weighted_lr0.001-ablation4/" \
  --use_pretrained_model "True" \
  --finetune "False" \
  --momentum "False" \
  --momentum_param "0.5" \
  --gpu_idx "0" \
  --align "True" \
  --batch_size "64" \
  --lr "0.001"  \
  --lr_online "0.0001" \
  --epoch "300"  \
  --backbone "EEGNet-4,2" \
  --param_runs "./runs_debug/" \
  --use_BN "False" \
  --stride "8" \
  --batch_size_online "8" \
  --loss_func "CE_KL" \
  --selection_ratio "1.0" \
  --mt "0.1" \
  --updating_type "entropy" \
  --loss_weights "1.0, 0.0, 0.0" \