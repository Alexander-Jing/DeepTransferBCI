# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import csv
import re

from tl.utils.utils import str2bool
from tl.utils.utils import float_list
from tl.utils.network import backbone_net
from tl.utils.LogRecord import LogRecord
from tl.utils.dataloader import read_mi_combine_tar
from tl.utils.utils import fix_random_seed, cal_acc_comb, data_loader, cal_auc_comb, cal_score_online, makedir_if_not_exist, build_optimizer, \
    save_features_predictions, load_features_predictions
from tl.utils.alg_utils import EA, EA_online
from scipy.linalg import fractional_matrix_power
from tl.models.proposed_method_43 import proposed_TTA
from sklearn.metrics import roc_auc_score, accuracy_score

from box import Box
from collections import OrderedDict

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from scipy.stats import entropy
import seaborn as sns
from scipy.stats import linregress
from scipy.stats import wasserstein_distance
from scipy.stats import spearmanr
import glob

def grad_visual(result_path, seed, idx, capacity=64):
    dir_path = os.path.join(result_path, 'gradients', f'seed{seed}_sub_idx{idx}')
    file_list = glob.glob(os.path.join(dir_path, 'memory_buffer_instance_*.pt'))
    if not file_list:
        raise FileNotFoundError(f"No memory_buffer_instance_*.pt files found in {dir_path}")

    def extract_instance_num(filename):
        match = re.search(r'memory_buffer_instance_(\d+)\.pt', filename)
        return int(match.group(1)) if match else -1

    file_list = sorted(file_list, key=extract_instance_num)

    num_instances = []
    grads_1_norms = []
    grads_2_norms = []
    cosine_similarities = []

    for file_path in file_list:
        num_instance = extract_instance_num(file_path)
        data = torch.load(file_path, map_location='cpu')
        num_instances.append(num_instance)
        grads_1_norms.append(data['norm_1'])
        grads_2_norms.append(data['norm_2'])
        cosine_similarities.append(data['cosine_similarity'])

    plt.figure(figsize=(10, 6))
    plt.plot(num_instances, grads_1_norms, label='grads_1 norm')
    plt.plot(num_instances, grads_2_norms, label='grads_2 norm')
    plt.plot(num_instances, cosine_similarities, label='cosine_similarity')
    plt.xlabel('num_instance')
    plt.ylabel('Value')
    plt.title('Gradient Norms and Cosine Similarity vs num_instance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图片
    fig_dir = os.path.join(result_path, 'grad_figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f'seed{seed}_sub_idx{idx}.png')
    plt.savefig(fig_path)
    plt.close()

if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='BNCI2014001', help='the data set name, now support BNCI2014001, BNCI2014002, BNCI2015001 from moabb')
    parser.add_argument('--data_save', type=str2bool, default=True, help='whether save the data to file')
    parser.add_argument('--data_path', type=str, default='./data/', help='the path to save the data from mobba dataset')
    parser.add_argument('--data_path_MI', type=str, default='/home/jyt/workspace/transfer_models/datasets_MI/hand_elbow/derivatives', help='the path to save the data from other datasets')
    parser.add_argument('--log_path', type=str, default='./logs/', help='the path to save the logs')
    parser.add_argument('--gpu_idx', type=int, default=0, help='index of GPU')
    parser.add_argument('--use_pretrained_model', type=str2bool, default=False, help='whether to use the pretrained model parameters')
    parser.add_argument('--finetune', type=str2bool, default=False, help='whether to finetune the model with part of the target data')
    parser.add_argument('--ft_volume', type=int, default=7*40, help='the amount of data for finetuning in target domain')
    parser.add_argument('--momentum', type=str2bool, default=False, help='whether to use the momentum updating for model parameters')
    parser.add_argument('--momentum_param', type=float, default=0.5, help='the value for momentum updating')
    parser.add_argument('--align', type=str2bool, default=True, help='use EA alignment and IEA alignment')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in offline training')
    parser.add_argument('--batch_size_online', type=int, default=8, help='batch size in online adaptation')
    parser.add_argument('--stride', type=int, default=1, help='stride in online adaptation')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate in offline and online training')
    parser.add_argument('--lr_online', type=float, default=0.001, help='learning rate in online adaptation')
    parser.add_argument('--epoch', type=int, default=100, help='epoches in offline and online training')
    parser.add_argument('--backbone', type=str, default='EEGNet', help='backbone of the model')
    parser.add_argument('--param_runs', type=str, default='./runs/', help='folder for saving the run paramters')
    parser.add_argument('--use_BN', type=str2bool, default=True, help='whether to only use BN adaptation')
    parser.add_argument('--loss_func', type=str, default="MemorySoftplusEnergyWeightedAlignment", help='the name of loss function')
    parser.add_argument('--updating_type', type=str, default="entropy", help='updating type')
    parser.add_argument('--selection_ratio', type=float, default=0.5, help='the ratio for sample selection in EnergyEntropy_selected')
    parser.add_argument('--mt', type=float, default=0.9, help='the momentum value for teacher model')
    parser.add_argument('--loss_weights', type=float_list, default=[1.0, 1.0, 1.0], help='weights for 3 loss components')
    parser.add_argument('--perplexity', type=int, default=30, help='parameter for visulization')
    
    args = parser.parse_args()

    data_name = args.dataset_name
    data_save = args.data_save
    data_path = args.data_path
    data_path_MI = args.data_path_MI
    log_path = args.log_path
    gpu_idx = args.gpu_idx
    use_pretrained_model = args.use_pretrained_model
    finetune = args.finetune
    ft_volume = args.ft_volume
    momentum = args.momentum
    momentum_param = args.momentum_param
    align = args.align
    batch_size = args.batch_size
    batch_size_online = args.batch_size_online
    lr = args.lr
    epoch = args.epoch
    backbone = args.backbone
    param_runs = args.param_runs
    lr_online = args.lr_online
    use_BN = args.use_BN
    stride = args.stride
    loss_func = args.loss_func
    updating_type = args.updating_type
    selection_ratio = args.selection_ratio
    mt = args.mt
    loss_weights = args.loss_weights

    print('dataset_name: {}, type: {}'.format(data_name, type(data_name)))
    print('data_save: {}, type: {}'.format(data_save, type(data_save)))
    print('data_path_MI: {}, type: {}'.format(data_path_MI, type(data_path_MI)))
    print('data_path: {}, type: {}'.format(data_path, type(data_path)))
    print('log_path: {}, type: {}'.format(log_path, type(log_path)))
    print('gpu_idx: {}, type: {}'.format(gpu_idx, type(gpu_idx)))

    data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'BNCI2014001-4', 'MI-hand_elbow','MI-elbow_rest', 'MI-hand_rest', 
                      'BNCI2014001-4-all', 'BNCI2014001-4-test', 'BNCI2014001-4-train', 'BNCI2014_004-train', 'BNCI2014_004-test',
                      'WBCIC-SHU-3C']
    dct = pd.DataFrame(columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13'])

    if data_name in data_name_list:
        # N: number of subjects, chn: number of channels
        if backbone == 'EEGNet':
            if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
            if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
            if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
            if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 576, 496
            if data_name == 'BNCI2014001-4-train': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 496
            if data_name == 'MI-hand_elbow': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
            if data_name == 'MI-elbow_rest': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
            if data_name == 'MI-hand_rest': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
            if data_name == 'BNCI2014001-4-all': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 576, 496
            if data_name == 'BNCI2014001-4-test': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 496
            if data_name == 'BNCI2014_004-train': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 3, 2, 1126, 250, 400, 560
            if data_name == 'BNCI2014_004-test': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 3, 2, 1126, 250, 400, 560
            if data_name == 'WBCIC-SHU-3C': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 11, 58, 3, 1000, 250, 900, 496
        if backbone == 'EEGNet-4,2':
            if data_name == 'BNCI2014001-4-train': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248
            if data_name == 'MI-hand_elbow': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
            if data_name == 'MI-elbow_rest': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
            if data_name == 'MI-hand_rest': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
            if data_name == 'BNCI2014001-4-all': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 576, 248
            if data_name == 'BNCI2014001-4-test': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248
            if data_name == 'BNCI2014_004-train': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 3, 2, 1126, 250, 400, 280
            if data_name == 'BNCI2014_004-test': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 3, 2, 1126, 250, 400, 280
            if data_name == 'WBCIC-SHU-3C': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 11, 58, 3, 1000, 250, 900, 248
        
        # whether to use pretrained model
        # if source models have not been trained, set use_pretrained_model to False to train them
        # alternatively, run dnn.py to train source models, in seperating the steps
        if use_pretrained_model:
            # no training
            max_epoch = 0
        else:
            # training epochs
            max_epoch = epoch

        # learning rate
        lr = lr

        # test batch size
        test_batch = batch_size_online

        # update step
        steps = 1

        # whether to use EA
        align = align

        # whether to test balanced or imbalanced (2:1) target subject
        balanced = True

        # whether to record running time
        calc_time = False

        # whether to use finetuning methods for some of the MI tasks and set how much data for finetuning
        if finetune:
            print('finetune: {}, ft_volume: {}'.format(finetune, ft_volume))

        # whether to use momentum updating method
        if momentum:
            print('momentum: {}, momentum_param: {}'.format(momentum, momentum_param))

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, align=align, lr=lr, max_epoch=max_epoch,
                                  trial_num=trial_num, time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, stride=stride, steps=steps, calc_time=calc_time,
                                  paradigm=paradigm, test_batch=test_batch, data_name=data_name, balanced=balanced, data_path_MI = data_path_MI,
                                  finetune=finetune,ft_volume=ft_volume,momentum=momentum,momentum_param=momentum_param, mt=mt)

        args.method = 'proposed_method'
        args.backbone = backbone

        args.epoch = epoch
        # train batch size
        args.batch_size = batch_size
        args.lr_online = lr_online  # learning rate for online adaptation

        # path for saving the offline models
        args.param_runs = param_runs
        args.runs_path = str(args.param_runs)  + str(args.data_name) + '_' + str(args.backbone) + '_b' + str(args.batch_size) + '_e' + str(args.epoch) + '_lr' + str(args.lr)
        
        # GPU device id
        try:
            device_id = gpu_idx
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
        except:
            args.data_env = 'local'

        # hyperparameters
        args.paras_optim = Box({
            "name": "Adam",   
            "lr": args.lr_online,
            "beta": 0.9,        
            "wd": 0.0,
            "two_stage": True,
        })
        args.EnergyAlignment = Box({
            "ratio":selection_ratio,
            "lambda_1": loss_weights[0],
            "lambda_2": loss_weights[1],
            "lambda_3": loss_weights[2],
            "temp": 2.0,
        })
        args.capacity = 64
        args.bn_alpha = 0.1
        args.update_frequency = args.stride
        args.update_counter = 'each'
        args.confidence_threshold = 0.33
        args.uncertainty_threshold = 0.75
        args.prune_ratio = 0.5
        args.pruning_strategy = 'ln_structured'
        args.pruning_module = 'conv'
        args.metric_name = 'mean_probs_dropout'
        args.use_BN = use_BN
        args.loss_name = loss_func
        args.updating_type = updating_type

        total_acc = []

        dataset_001 = {
            "buffer_name":
            ["Ours", "HUS", "FIFO", "CSTU"],
            "log_path":
            ["./logs/Baselines-001-test-e300-b64-debugs/ours_debug_m_cls_process_1-BNCI2014001-4-all-EEGNet-4,2-e300-b64/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-2-visualization/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p1",
            "./logs/Baselines-001-test-e300-b64-debugs/ours_debug_m_cls_process_1-BNCI2014001-4-all-EEGNet-4,2-e300-b64/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation4/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visulization-1",
            "./logs/Baselines-001-test-e300-b64-debugs/ours_debug_m_cls_process_1-BNCI2014001-4-all-EEGNet-4,2-e300-b64/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation5/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visualization1",
            "./logs/Baselines-001-test-e300-b64-debugs/ours_debug_m_cls_process_1-BNCI2014001-4-all-EEGNet-4,2-e300-b64/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation6/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visualization1",],
            "sub_num": 9,
        }
        dataset_004 = {
            "buffer_name":
            ["Ours", "HUS", "FIFO", "CSTU"],
            "log_path": 
            ["./logs/Baselines-004-test-e300-b64/proposed/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-visualization/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p6",
            "./logs/Baselines-004-test-e300-b64/proposed/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation4/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visulization1",
            "./logs/Baselines-004-test-e300-b64/proposed/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation5/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visualization1",
            "./logs/Baselines-004-test-e300-b64/proposed/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation6/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visualization1",],
            "sub_num": 9,
        }
            
        dataset_SHU = {
            "buffer_name":
            ["Ours", "HUS", "FIFO", "CSTU"],
            "log_path": 
            ["/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-WBCIC-SHU-3C-e300-b64/proposed/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-visualization/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p1",
            "/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-WBCIC-SHU-3C-e300-b64/proposed/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation4/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visualization1",
            "/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-WBCIC-SHU-3C-e300-b64/proposed/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation5/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visualization1",
            "/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-WBCIC-SHU-3C-e300-b64/proposed/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation6/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visualization1",],
            "sub_num": 11,
        } 
        
        if data_name in ['WBCIC-SHU-3C']:
            args.dataset_info = dataset_SHU
        elif data_name in ["BNCI2014001-4-all"]:
            args.dataset_info = dataset_001
        elif data_name in ["BNCI2014_004-test"]:
            args.dataset_info = dataset_004
        else:
            args.dataset_info = None

        # update multiple models, independently, from the source models
        total_acc = []
        total_acc_1 = []
        total_acc_2 = []
        total_acc_3 = []
        for s in [1]:
            args.SEED = s

            fix_random_seed(args.SEED)
            torch.backends.cudnn.deterministic = True

            args.data = data_name

            args.local_dir = data_path + str(data_name) + '/'
            args.result_dir = log_path
            
            mean_sub_acc = []
            mean_sub_acc_1 = []
            mean_sub_acc_2 = []
            mean_sub_acc_3 = []
            for idt in range(N):
                fix_random_seed(args.SEED)  # fix the seed
                args.idt = idt
                source_str = 'Except_S' + str(idt)
                target_str = 'S' + str(idt)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                
                # 用法示例
                grad_visual(str(args.result_dir), args.SEED, args.idt)
            
    
            
