#!/bin/bash

workspace_folder="/home/jyt/workspace/DeepTransferBCI"
cd $workspace_folder

progarm_path="tl/proposed_method/ours_debug_m_cls_process_36.py"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

confidence_thresholds=(0.40)
entropy_thresholds=(0.45)
selection_ratios=(1.0)
selection_ratio_reviews=(1.0)

base_log_path="./logs/Baselines-001-test-e300-b64-debugs/ours_debug_m_cls_process_1-BNCI2014001-4-all-EEGNet-4,2-e300-b64/proposed_56_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-4-1-params/proposed_56_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p"

index=7

for conf in "${confidence_thresholds[@]}"; do
  for ent in "${entropy_thresholds[@]}"; do
    for sel in "${selection_ratios[@]}"; do
      for sel_review in "${selection_ratio_reviews[@]}"; do
        log_path="${base_log_path}${index}"
        echo "Running combination $index: confidence_threshold=$conf, entropy_threshold=$ent, selection_ratio=$sel, selection_ratio_review=$sel_review"
        echo "Log path: $log_path"

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
          --loss_func "CE_KL_review_weighted_3,ConsSamples_selection_two_stage_weighted_4_1_review_4_1" \
          --selection_ratio "$sel" \
          --selection_ratio_review "$sel_review" \
          --mt "0.9" \
          --updating_type "entropy_review" \
          --loss_weights "1.0, 1.0, 1.0" \
          --scale "10.0" \
          --confidence_threshold "$conf" \
          --entropy_threshold "$ent" \
          --memory_type "DropMemoryBank_review_7" \
          --memory_review "get_memory" \
          --weight_type "entropy" \
          --memory_capacity "64"

        ((index++))
        echo "Completed combination $((index-1))"
        echo "----------------------------------------"
      done
    done
  done
done

echo "All parameter combinations have been processed."