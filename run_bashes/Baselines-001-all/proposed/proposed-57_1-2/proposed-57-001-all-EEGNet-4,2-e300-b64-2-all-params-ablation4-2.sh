#!/bin/bash

# Define the workspace folder
workspace_folder="/home/jyt/workspace/DeepTransferBCI"
# Change the current directory to the workspace folder
cd $workspace_folder

# Define the program path (注意: 原脚本中变量名有拼写错误 "progarm_path"，此处保持原样)
progarm_path="tl/proposed_method/ours_debug_m_cls_process_43.py"

# Set the PYTHONPATH environment variable
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Define the parameter values to iterate over
confidence_thresholds=(0.45)
thre_alpha=(1.0)
loss_weight_type=(20.0)

# Base log path (without the trailing index)
base_log_path="./logs/Baselines-001-test-e300-b64-debugs/ours_debug_m_cls_process_1-BNCI2014001-4-all-EEGNet-4,2-e300-b64/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-2-params-ablation4/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p"

# Initialize a counter for the parameter combinations
index=3

# Nested loops to iterate over all parameter combinations
for conf in "${confidence_thresholds[@]}"; do
    for ent in "${thre_alpha[@]}"; do
        for los in "${loss_weight_type[@]}"; do
          # Construct the log path with the current index
          log_path="${base_log_path}${index}"
          
          echo "Running combination $index: confidence_threshold=$conf, entropy_threshold=$ent"
          echo "Log path: $log_path"

          # Execute the Python script with the current parameters
          python3 $progarm_path \
            --dataset_name "BNCI2014001-4-all" \
            --data_save "True" \
            --data_path "./data/" \
            --data_path_MI "None" \
            --log_path "$log_path" \
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
            --loss_func "CE_KL_review_weighted_10,ConsSamples_selection_two_stage_weighted_4_1_review_4_4_4" \
            --selection_ratio "1.0" \
            --selection_ratio_review "1.0" \
            --mt "0.9" \
            --updating_type "entropy_review" \
            --loss_weights "1.0, 1.0, 1.0" \
            --scale "$los" \
            --confidence_threshold "$conf" \
            --entropy_threshold "0.50" \
            --memory_type "HUS" \
            --memory_review "get_memory" \
            --weight_type "entropy" \
            --memory_capacity "64" \
            --thre_alpha "$ent" \
            --loss_weight_type "buffer_sigmoid" \
            --gate_type "mean" \
            --buffer_selefction_type "dynamic_confidence" \
            --min_threshold "0.40" \
          # Increment the index for the next combination
          ((index++))
          
          echo "Completed combination $((index-1))"
          echo "----------------------------------------"
        done
    done
done

echo "All parameter combinations have been processed."