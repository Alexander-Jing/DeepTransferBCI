#!/bin/bash

# Define the workspace folder
workspace_folder="/home/jyt/workspace/DeepTransferBCI"
# Change the current directory to the workspace folder
cd $workspace_folder

# Define the program path (注意: 原脚本中变量名有拼写错误 "progarm_path"，此处保持原样)
progarm_path="tl/proposed_method/ours_debug_m_cls_process_28.py"

# Set the PYTHONPATH environment variable
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Define the parameter values to iterate over
confidence_thresholds=(0.60)
entropy_thresholds=(0.40)

# Base log path (without the trailing index)
base_log_path="./logs/Baselines-WBCIC-SHU-3C-e300-b64/proposed/proposed_53_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-12-4-params/proposed_53_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p"

# Initialize a counter for the parameter combinations
index=1

# Nested loops to iterate over all parameter combinations
for conf in "${confidence_thresholds[@]}"; do
    for ent in "${entropy_thresholds[@]}"; do
        # Construct the log path with the current index
        log_path="${base_log_path}${index}"
        
        echo "Running combination $index: confidence_threshold=$conf, entropy_threshold=$ent"
        echo "Log path: $log_path"

        # Execute the Python script with the current parameters
        python3 $progarm_path \
          --dataset_name "WBCIC-SHU-3C" \
          --data_save "True" \
          --data_path "./data/" \
          --data_path_MI "/data/datasets_Jyt/WBCIC_SHU_3C/processeddata/processeddata/" \
          --log_path "$log_path" \
          --use_pretrained_model "True" \
          --finetune "False" \
          --momentum "False" \
          --momentum_param "0.5" \
          --gpu_idx "0" \
          --align "True" \
          --batch_size "64" \
          --lr "0.001"  \
          --lr_online "0.0005" \
          --epoch "300"  \
          --backbone "EEGNet-4,2" \
          --param_runs "./runs_debug/" \
          --use_BN "False" \
          --stride "1" \
          --batch_size_online "8" \
          --loss_func "CE_KL_review_weighted_3,ConsSamples_selection_two_stage_weighted_4_1_review_4_1" \
          --selection_ratio "0.75" \
          --mt "0.9" \
          --updating_type "entropy_review" \
          --loss_weights "1.0, 1.0, 1.0" \
          --scale "10.0" \
          --confidence_threshold "$conf" \
          --entropy_threshold "$ent" \
          --memory_type "DropMemoryBank_review_1" \
          --memory_review "get_memory_review_1" \

        # Increment the index for the next combination
        ((index++))
        
        echo "Completed combination $((index-1))"
        echo "----------------------------------------"
    done
done

echo "All parameter combinations have been processed."