# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import csv

from tl.utils.utils import str2bool
from tl.utils.utils import float_list
from tl.utils.network import backbone_net
from tl.utils.LogRecord import LogRecord
from tl.utils.dataloader import read_mi_combine_tar
from tl.utils.utils import fix_random_seed, cal_acc_comb, data_loader, cal_auc_comb, cal_score_online, makedir_if_not_exist, build_optimizer, \
    save_features_predictions, load_features_predictions
from tl.utils.alg_utils import EA, EA_online
from scipy.linalg import fractional_matrix_power
from tl.models.proposed_method_14 import proposed_TTA
from sklearn.metrics import roc_auc_score, accuracy_score

from box import Box
from collections import OrderedDict

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

def visualize_tsne(feas, preds, labels, args, use_true_labels=True):
    """
    使用t-SNE可视化特征嵌入并按类别着色
    参数:
        feas: 特征数组 (N, D)
        preds: 预测标签数组 (N,)
        labels: 真实标签数组 (N,)
        args: 包含配置参数的对象
        use_true_labels: 是否使用真实标签着色 (默认True)
    """
    # 检查输入维度
    assert feas.shape[0] == labels.shape[0], "特征与标签数量不匹配"
    
    # 选择标签类型（真实标签或预测标签）
    target_labels = labels if use_true_labels else preds
    unique_labels = np.unique(target_labels)
    n_classes = len(unique_labels)
    
    # 创建输出目录
    os.makedirs(os.path.join(args.result_dir, "tsne_plots"), exist_ok=True)
    
    # 1. 执行t-SNE降维
    tsne = TSNE(n_components=2, 
                perplexity=args.perplexity if hasattr(args, 'perplexity') else 30,
                random_state=args.SEED)
    tsne_results = tsne.fit_transform(feas)
    
    # 2. 创建可视化
    plt.figure(figsize=(10, 8))
    
    # 使用传统基础颜色方案
    base_colors = ['red', 'blue', 'green', 'purple', 'orange', 
                  'cyan', 'magenta', 'yellow', 'brown', 'pink',
                  'lime', 'teal', 'navy', 'maroon', 'olive']
    
    # 循环使用基础颜色（当类别数超过颜色列表长度时）
    colors = [base_colors[i % len(base_colors)] for i in range(n_classes)]
    
    # 绘制每个类别的散点
    for i, label in enumerate(unique_labels):
        indices = np.where(target_labels == label)
        plt.scatter(tsne_results[indices, 0], 
                    tsne_results[indices, 1],
                    color=colors[i],
                    label=f'Class {label}',
                    alpha=0.7,
                    edgecolor='w',
                    s=50)
    
    # 3. 图表美化
    plt.title(f't-SNE Visualization ({args.method}, Seed={args.SEED})', fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.grid(alpha=0.2)
    
    # 智能图例放置
    plt.legend(ncol=2 if n_classes > 10 else 1, 
               fontsize=10,
               frameon=True,
               loc='best')
    
    # 4. 保存图像
    file_name = f"{args.data_name}_{args.method}_seed_{args.SEED}_sub_{args.idt}_tsne.png"
    save_path = os.path.join(args.result_dir, "tsne_plots", file_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE可视化图已保存至: {save_path}")
    return save_path

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

        # update multiple models, independently, from the source models
        for s in [1, 2, 3, 4, 5]:
            args.SEED = s

            fix_random_seed(args.SEED)
            torch.backends.cudnn.deterministic = True

            args.data = data_name

            args.local_dir = data_path + str(data_name) + '/'
            args.result_dir = log_path
            
            sub_acc_all = np.zeros(N)
            for idt in range(N):
                fix_random_seed(args.SEED)  # fix the seed
                args.idt = idt
                source_str = 'Except_S' + str(idt)
                target_str = 'S' + str(idt)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                
                feas, preds, labels = load_features_predictions(args)
                visualize_tsne(feas, preds, labels, args, use_true_labels=True)
                print("ok")

