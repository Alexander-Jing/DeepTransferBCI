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

def load_two_stage(args, result_path, seed, idx, instance_num=None):
    """
    加载二阶段存储的特征和预测信息
    :param args: 参数对象
    :param result_path: 结果主目录
    :param seed: 随机种子
    :param idx: 子任务索引
    :param instance_num: 指定实例编号（可选），不指定则加载所有
    :return: 加载的内容（dict 或 dict 列表）
    """
    dir_path = os.path.join(result_path, 'two_stage', f'seed{seed}_sub_idx{idx}')
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    if instance_num is not None:
        file_path = os.path.join(dir_path, f'memory_buffer_instance_{instance_num}.pt')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        data = torch.load(file_path, map_location='cpu')
        return data
    else:
        # 加载所有实例
        file_list = sorted(glob.glob(os.path.join(dir_path, 'memory_buffer_instance_*.pt')))
        if not file_list:
            raise FileNotFoundError(f"No memory_buffer_instance_*.pt files found in {dir_path}")
        data_list = [torch.load(f, map_location='cpu') for f in file_list]
        return data_list


def calc_review_acc(result_path, seed, idx, capacity=64):
    dir_path = os.path.join(result_path, 'two_stage', f'seed{seed}_sub_idx{idx}')
    file_list = sorted(glob.glob(os.path.join(dir_path, 'memory_buffer_instance_*.pt')))
    if not file_list:
        raise FileNotFoundError(f"No memory_buffer_instance_*.pt files found in {dir_path}")

    acc_review = []
    acc_review_1 = []
    for file_path in file_list:
        data = torch.load(file_path, map_location='cpu')
        # 提取logits和标签
        preds_of_data_review = data['preds_of_data_review']  # shape: [N, num_classes]
        preds_of_data_review_1 = data['preds_of_data_review_1']  # shape: [N, num_classes]
        review_data_class = data['review_data_class']  # shape: [N]
        if preds_of_data_review.shape[0] < capacity//2:
            continue  # 跳过空的
        pred_label = preds_of_data_review.argmax(dim=1)
        pred_label_1 = preds_of_data_review_1.argmax(dim=1)
        true_label = review_data_class
        acc = (pred_label == true_label).float().mean().item()
        acc1 = (pred_label_1 == true_label).float().mean().item()
        acc_review.append(acc)
        acc_review_1.append(acc1)
    # 输出平均准确率
    avg_acc = np.mean(acc_review) if acc_review else 0.0
    avg_acc1 = np.mean(acc_review_1) if acc_review_1 else 0.0
    #print(f"preds_of_data_review accuracy: {avg_acc:.4f}")
    #print(f"preds_of_data_review_1 accuracy: {avg_acc1:.4f}")
    return avg_acc, avg_acc1


def calc_pred_acc(result_path, seed, idx, capacity=64):
    dir_path = os.path.join(result_path, 'two_stage', f'seed{seed}_sub_idx{idx}')
    file_list = sorted(glob.glob(os.path.join(dir_path, 'memory_buffer_instance_*.pt')))
    if not file_list:
        raise FileNotFoundError(f"No memory_buffer_instance_*.pt files found in {dir_path}")

    acc_preds = []
    acc_preds_1 = []
    acc_preds_2 = []
    acc_preds_3 = []
   
    instance_nums = []
    for file_path in file_list:
        # 提取num_instance
        filename = os.path.basename(file_path)
        match = re.search(r'memory_buffer_instance_(\d+)\.pt', filename)
        if match:
            num_instance = int(match.group(1))
            instance_nums.append(num_instance)
        else:
            num_instance = None  # 未匹配到编号
        
        online_buffer_path = os.path.join(result_path, 'online_visulization_records', f'seed{seed}_sub_idx{idx}', f'online_buffer_instance_{num_instance}.pt')
        online_buffer_data = torch.load(online_buffer_path, map_location='cpu')
        true_labels = [_instance['label'] for _instance in online_buffer_data]
        true_labels = torch.tensor(true_labels, dtype=torch.long)

        
        online_memory_path = os.path.join(result_path, 'memory_visulization_records', f'seed{seed}_sub_idx{idx}', f'memory_buffer_instance_{num_instance}.pt')
        online_memory_data = torch.load(online_memory_path, map_location='cpu')
        true_labels_memory = []
        for class_items in online_memory_data:
            for item in class_items:
                true_labels_memory.append(item['label'])
        true_labels_memory = torch.tensor(true_labels_memory, dtype=torch.long)


        data = torch.load(file_path, map_location='cpu')
        preds_of_data = data['preds_of_data']
        preds_of_data_1 = data['preds_of_data_1']
        preds_of_data_review = data['preds_of_data_review']
        preds_of_data_review_1 = data['preds_of_data_review_1']
        
        if preds_of_data_review.shape[0] < capacity // 2:
            continue
        
        pred_label = preds_of_data.argmax(dim=1)
        pred_label_1 = preds_of_data_1.argmax(dim=1)
        pred_label_review = preds_of_data_review.argmax(dim=1)
        pred_label_review_1 = preds_of_data_review_1.argmax(dim=1)

        acc = (pred_label == true_labels).float().mean().item()
        acc_1 = (pred_label_1 == true_labels).float().mean().item()
        acc_2 = (pred_label_review == true_labels_memory).float().mean().item()
        acc_3 = (pred_label_review_1 == true_labels_memory).float().mean().item()

        acc_preds.append(acc)
        acc_preds_1.append(acc_1)
        acc_preds_2.append(acc_2)
        acc_preds_3.append(acc_3)

    avg_acc = np.mean(acc_preds) if acc_preds else 0.0
    avg_acc1 = np.mean(acc_preds_1) if acc_preds_1 else 0.0
    avg_acc2 = np.mean(acc_preds_2) if acc_preds_2 else 0.0
    avg_acc3 = np.mean(acc_preds_3) if acc_preds_3 else 0.0
    #print(f"Average preds_of_data accuracy: {avg_acc:.4f}")
    #print(f"Average preds_of_data_review_1 accuracy: {avg_acc1:.4f}")
    return avg_acc, avg_acc1, avg_acc2, avg_acc3


def review_visualization_1(result_path, seed, idx, capacity=64, pseduo_label=True, types='logits'):
    dir_path = os.path.join(result_path, 'two_stage', f'seed{seed}_sub_idx{idx}')
    file_list = sorted(glob.glob(os.path.join(dir_path, 'memory_buffer_instance_*.pt')))
    if not file_list:
        raise FileNotFoundError(f"No memory_buffer_instance_*.pt files found in {dir_path}")

    for file_path in file_list:
        if file_path.endswith('memory_buffer_instance_320.pt'):
            filename = os.path.basename(file_path)
            match = re.search(r'memory_buffer_instance_(\d+)\.pt', filename)
            if match:
                num_instance = int(match.group(1))
            else:
                num_instance = None

            online_buffer_path = os.path.join(result_path, 'online_visulization_records', f'seed{seed}_sub_idx{idx}', f'online_buffer_instance_{num_instance}.pt')
            online_buffer_data = torch.load(online_buffer_path, map_location='cpu')
            true_labels = [_instance['label'] for _instance in online_buffer_data]
            true_labels = torch.tensor(true_labels, dtype=torch.long)

            online_memory_path = os.path.join(result_path, 'memory_visulization_records', f'seed{seed}_sub_idx{idx}', f'memory_buffer_instance_{num_instance}.pt')
            online_memory_data = torch.load(online_memory_path, map_location='cpu')
            true_labels_memory = []
            for class_items in online_memory_data:
                for item in class_items:
                    true_labels_memory.append(item['label'])
            true_labels_memory = torch.tensor(true_labels_memory, dtype=torch.long)

            data = torch.load(file_path, map_location='cpu')
            if types in ['logits','logit']:
                preds_of_data_1 = data['preds_of_data_1']
                preds_of_data_2 = data['preds_of_data_2']
                preds_of_data_review_1 = data['preds_of_data_review_1']
                preds_of_data_review_2 = data['preds_of_data_review_2']
            
            if preds_of_data_review_1.shape[0] < capacity // 2:
                continue
            preds_1 = torch.vstack((preds_of_data_1, preds_of_data_review_1)).cpu().numpy()
            preds_2 = torch.vstack((preds_of_data_2, preds_of_data_review_2)).cpu().numpy()
            if pseduo_label:
                preds_labels_1 = np.argmax(preds_1, axis=1)
                preds_labels_2 = np.argmax(preds_2, axis=1)
            else:
                true_labels_combined = torch.cat((true_labels, true_labels_memory), dim=0).cpu().numpy()

            # t-SNE projection
            tsne_1 = TSNE(n_components=2, random_state=42)
            tsne_2 = TSNE(n_components=2, random_state=42)
            preds_1_tsne = tsne_1.fit_transform(preds_1)
            preds_2_tsne = tsne_2.fit_transform(preds_2)

            # 保存文件夹
            if pseduo_label:
                figures_dir = os.path.join(result_path, 'two_stage_figures', f'seed{seed}_sub_idx{idx}_pesudo_label_1')
            else:
                figures_dir = os.path.join(result_path, 'two_stage_figures', f'seed{seed}_sub_idx{idx}_1')
                
            os.makedirs(figures_dir, exist_ok=True)

            # Plot preds_1
            plt.figure(figsize=(7, 6))
            if pseduo_label:
                scatter = plt.scatter(preds_1_tsne[:, 0], preds_1_tsne[:, 1], c=preds_labels_1, cmap='tab10', s=60)
            else:
                scatter = plt.scatter(preds_1_tsne[:, 0], preds_1_tsne[:, 1], c=true_labels_combined, cmap='tab10', s=60)
            plt.xticks([])
            plt.yticks([])
            # plt.title(f't-SNE of preds_1 (instance {num_instance})')
            plt.savefig(os.path.join(figures_dir, f'instance_{num_instance}_preds1.png'))
            plt.savefig(os.path.join(figures_dir, f'instance_{num_instance}_preds1.svg'))
            plt.close()

            # Plot preds_2
            plt.figure(figsize=(7, 6))
            if pseduo_label:
                scatter = plt.scatter(preds_2_tsne[:, 0], preds_2_tsne[:, 1], c=preds_labels_2, cmap='tab10', s=60)
            else:
                scatter = plt.scatter(preds_2_tsne[:, 0], preds_2_tsne[:, 1], c=true_labels_combined, cmap='tab10', s=60)
            plt.xticks([])
            plt.yticks([])
            # plt.title(f't-SNE of preds_2 (instance {num_instance})')
            plt.savefig(os.path.join(figures_dir, f'instance_{num_instance}_preds2.png'))
            plt.savefig(os.path.join(figures_dir, f'instance_{num_instance}_preds2.svg'))
            plt.close()        
    

def review_visualization(result_path, seed, idx, capacity=64, pseduo_label=True, types='logits'):
    dir_path = os.path.join(result_path, 'two_stage', f'seed{seed}_sub_idx{idx}')
    file_list = sorted(glob.glob(os.path.join(dir_path, 'memory_buffer_instance_*.pt')))
    if not file_list:
        raise FileNotFoundError(f"No memory_buffer_instance_*.pt files found in {dir_path}")

    for file_path in file_list:
        filename = os.path.basename(file_path)
        match = re.search(r'memory_buffer_instance_(\d+)\.pt', filename)
        if match:
            num_instance = int(match.group(1))
        else:
            num_instance = None

        online_buffer_path = os.path.join(result_path, 'online_visulization_records', f'seed{seed}_sub_idx{idx}', f'online_buffer_instance_{num_instance}.pt')
        online_buffer_data = torch.load(online_buffer_path, map_location='cpu')
        true_labels = [_instance['label'] for _instance in online_buffer_data]
        true_labels = torch.tensor(true_labels, dtype=torch.long)

        online_memory_path = os.path.join(result_path, 'memory_visulization_records', f'seed{seed}_sub_idx{idx}', f'memory_buffer_instance_{num_instance}.pt')
        online_memory_data = torch.load(online_memory_path, map_location='cpu')
        true_labels_memory = []
        for class_items in online_memory_data:
            for item in class_items:
                true_labels_memory.append(item['label'])
        true_labels_memory = torch.tensor(true_labels_memory, dtype=torch.long)

        data = torch.load(file_path, map_location='cpu')
        if types in ['logits','logit']:
            preds_of_data_1 = data['preds_of_data_1']
            preds_of_data_2 = data['preds_of_data_2']
            preds_of_data_review_1 = data['preds_of_data_review_1']
            preds_of_data_review_2 = data['preds_of_data_review_2']
        
        if preds_of_data_review_1.shape[0] < capacity // 2:
            continue
        preds_1 = torch.vstack((preds_of_data_1, preds_of_data_review_1)).cpu().numpy()
        preds_2 = torch.vstack((preds_of_data_2, preds_of_data_review_2)).cpu().numpy()
        if pseduo_label:
            preds_labels_1 = np.argmax(preds_1, axis=1)
            preds_labels_2 = np.argmax(preds_2, axis=1)
        else:
            true_labels_combined = torch.cat((true_labels, true_labels_memory), dim=0).cpu().numpy()

        # t-SNE projection
        tsne_1 = TSNE(n_components=2, random_state=42)
        tsne_2 = TSNE(n_components=2, random_state=42)
        preds_1_tsne = tsne_1.fit_transform(preds_1)
        preds_2_tsne = tsne_2.fit_transform(preds_2)

        # 保存文件夹
        if pseduo_label:
            figures_dir = os.path.join(result_path, 'two_stage_figures', f'seed{seed}_sub_idx{idx}_pesudo_label')
        else:
            figures_dir = os.path.join(result_path, 'two_stage_figures', f'seed{seed}_sub_idx{idx}')
            
        os.makedirs(figures_dir, exist_ok=True)

        # Plot preds_1
        plt.figure(figsize=(8, 6))
        if pseduo_label:
            scatter = plt.scatter(preds_1_tsne[:, 0], preds_1_tsne[:, 1], c=preds_labels_1, cmap='tab10', s=10)
        else:
            scatter = plt.scatter(preds_1_tsne[:, 0], preds_1_tsne[:, 1], c=true_labels_combined, cmap='tab10', s=10)
        plt.colorbar(scatter)
        plt.title(f't-SNE of preds_1 (instance {num_instance})')
        plt.savefig(os.path.join(figures_dir, f'instance_{num_instance}_preds1.png'))
        plt.close()

        # Plot preds_2
        plt.figure(figsize=(8, 6))
        if pseduo_label:
            scatter = plt.scatter(preds_2_tsne[:, 0], preds_2_tsne[:, 1], c=preds_labels_2, cmap='tab10', s=10)
        else:
            scatter = plt.scatter(preds_2_tsne[:, 0], preds_2_tsne[:, 1], c=true_labels_combined, cmap='tab10', s=10)
        plt.colorbar(scatter)
        plt.title(f't-SNE of preds_2 (instance {num_instance})')
        plt.savefig(os.path.join(figures_dir, f'instance_{num_instance}_preds2.png'))
        plt.close()        


def review_visualization_feas(result_path, seed, idx, capacity=64, pseduo_label=True, types='feas'):
    dir_path = os.path.join(result_path, 'two_stage', f'seed{seed}_sub_idx{idx}')
    file_list = sorted(glob.glob(os.path.join(dir_path, 'memory_buffer_instance_*.pt')))
    if not file_list:
        raise FileNotFoundError(f"No memory_buffer_instance_*.pt files found in {dir_path}")

    for file_path in file_list:
        filename = os.path.basename(file_path)
        match = re.search(r'memory_buffer_instance_(\d+)\.pt', filename)
        if match:
            num_instance = int(match.group(1))
        else:
            num_instance = None

        online_buffer_path = os.path.join(result_path, 'online_visulization_records', f'seed{seed}_sub_idx{idx}', f'online_buffer_instance_{num_instance}.pt')
        online_buffer_data = torch.load(online_buffer_path, map_location='cpu')
        true_labels = [_instance['label'] for _instance in online_buffer_data]
        true_labels = torch.tensor(true_labels, dtype=torch.long)

        online_memory_path = os.path.join(result_path, 'memory_visulization_records', f'seed{seed}_sub_idx{idx}', f'memory_buffer_instance_{num_instance}.pt')
        online_memory_data = torch.load(online_memory_path, map_location='cpu')
        true_labels_memory = []
        for class_items in online_memory_data:
            for item in class_items:
                true_labels_memory.append(item['label'])
        true_labels_memory = torch.tensor(true_labels_memory, dtype=torch.long)

        data = torch.load(file_path, map_location='cpu')
        
        preds_of_data_1 = data['preds_of_data_1']
        preds_of_data_2 = data['preds_of_data_2']
        preds_of_data_review_1 = data['preds_of_data_review_1']
        preds_of_data_review_2 = data['preds_of_data_review_2']
        feas_of_data_1 = data['feas_of_data_1']
        feas_of_data_2 = data['feas_of_data_2']
        feas_of_data_review_1 = data['feas_of_data_review_1']
        feas_of_data_review_2 = data['feas_of_data_review_2']
        
        if preds_of_data_review_1.shape[0] < capacity // 2:
            continue
        preds_1 = torch.vstack((preds_of_data_1, preds_of_data_review_1)).cpu().numpy()
        preds_2 = torch.vstack((preds_of_data_2, preds_of_data_review_2)).cpu().numpy()
        feas_1 = torch.vstack((feas_of_data_1, feas_of_data_review_1)).cpu().numpy()
        feas_2 = torch.vstack((feas_of_data_2, feas_of_data_review_2)).cpu().numpy()

        if pseduo_label:
            preds_labels_1 = np.argmax(preds_1, axis=1)
            preds_labels_2 = np.argmax(preds_2, axis=1)
        else:
            true_labels_combined = torch.cat((true_labels, true_labels_memory), dim=0).cpu().numpy()

        # t-SNE projection
        if types in ['logits', 'logit']:
            tsne_1 = TSNE(n_components=2, random_state=42)
            tsne_2 = TSNE(n_components=2, random_state=42)
            preds_1_tsne = tsne_1.fit_transform(preds_1)
            preds_2_tsne = tsne_2.fit_transform(preds_2)
            save_dir_name = 'two_stage_figures'
        else:
            tsne_1 = TSNE(n_components=2, random_state=42)
            tsne_2 = TSNE(n_components=2, random_state=42)
            preds_1_tsne = tsne_1.fit_transform(feas_1)
            preds_2_tsne = tsne_2.fit_transform(feas_2)
            save_dir_name = 'two_stage_figures_feas'

        # 保存文件夹
        if pseduo_label:
            figures_dir = os.path.join(result_path, save_dir_name, f'seed{seed}_sub_idx{idx}_pesudo_label')
        else:
            figures_dir = os.path.join(result_path, save_dir_name, f'seed{seed}_sub_idx{idx}')
            
        os.makedirs(figures_dir, exist_ok=True)

        # Plot preds_1
        plt.figure(figsize=(8, 6))
        if pseduo_label:
            scatter = plt.scatter(preds_1_tsne[:, 0], preds_1_tsne[:, 1], c=preds_labels_1, cmap='tab10', s=10)
        else:
            scatter = plt.scatter(preds_1_tsne[:, 0], preds_1_tsne[:, 1], c=true_labels_combined, cmap='tab10', s=10)
        plt.colorbar(scatter)
        plt.title(f't-SNE of preds_1 (instance {num_instance})')
        plt.savefig(os.path.join(figures_dir, f'instance_{num_instance}_preds1.png'))
        plt.close()

        # Plot preds_2
        plt.figure(figsize=(8, 6))
        if pseduo_label:
            scatter = plt.scatter(preds_2_tsne[:, 0], preds_2_tsne[:, 1], c=preds_labels_2, cmap='tab10', s=10)
        else:
            scatter = plt.scatter(preds_2_tsne[:, 0], preds_2_tsne[:, 1], c=true_labels_combined, cmap='tab10', s=10)
        plt.colorbar(scatter)
        plt.title(f't-SNE of preds_2 (instance {num_instance})')
        plt.savefig(os.path.join(figures_dir, f'instance_{num_instance}_preds2.png'))
        plt.close()   


def review_visualization_feas_1(result_path, seed, idx, capacity=64, pseduo_label=True, types='feas'):
    dir_path = os.path.join(result_path, 'two_stage', f'seed{seed}_sub_idx{idx}')
    file_list = sorted(glob.glob(os.path.join(dir_path, 'memory_buffer_instance_*.pt')))
    if not file_list:
        raise FileNotFoundError(f"No memory_buffer_instance_*.pt files found in {dir_path}")

    for file_path in file_list:
        if file_path.endswith('memory_buffer_instance_320.pt'):
            filename = os.path.basename(file_path)
            match = re.search(r'memory_buffer_instance_(\d+)\.pt', filename)
            if match:
                num_instance = int(match.group(1))
            else:
                num_instance = None

            online_buffer_path = os.path.join(result_path, 'online_visulization_records', f'seed{seed}_sub_idx{idx}', f'online_buffer_instance_{num_instance}.pt')
            online_buffer_data = torch.load(online_buffer_path, map_location='cpu')
            true_labels = [_instance['label'] for _instance in online_buffer_data]
            true_labels = torch.tensor(true_labels, dtype=torch.long)

            online_memory_path = os.path.join(result_path, 'memory_visulization_records', f'seed{seed}_sub_idx{idx}', f'memory_buffer_instance_{num_instance}.pt')
            online_memory_data = torch.load(online_memory_path, map_location='cpu')
            true_labels_memory = []
            for class_items in online_memory_data:
                for item in class_items:
                    true_labels_memory.append(item['label'])
            true_labels_memory = torch.tensor(true_labels_memory, dtype=torch.long)

            data = torch.load(file_path, map_location='cpu')
            
            preds_of_data_1 = data['preds_of_data_1']
            preds_of_data_2 = data['preds_of_data_2']
            preds_of_data_review_1 = data['preds_of_data_review_1']
            preds_of_data_review_2 = data['preds_of_data_review_2']
            feas_of_data_1 = data['feas_of_data_1']
            feas_of_data_2 = data['feas_of_data_2']
            feas_of_data_review_1 = data['feas_of_data_review_1']
            feas_of_data_review_2 = data['feas_of_data_review_2']
            
            if preds_of_data_review_1.shape[0] < capacity // 2:
                continue
            preds_1 = torch.vstack((preds_of_data_1, preds_of_data_review_1)).cpu().numpy()
            preds_2 = torch.vstack((preds_of_data_2, preds_of_data_review_2)).cpu().numpy()
            feas_1 = torch.vstack((feas_of_data_1, feas_of_data_review_1)).cpu().numpy()
            feas_2 = torch.vstack((feas_of_data_2, feas_of_data_review_2)).cpu().numpy()

            if pseduo_label:
                preds_labels_1 = np.argmax(preds_1, axis=1)
                preds_labels_2 = np.argmax(preds_2, axis=1)
            else:
                true_labels_combined = torch.cat((true_labels, true_labels_memory), dim=0).cpu().numpy()

            # t-SNE projection
            if types in ['logits', 'logit']:
                tsne_1 = TSNE(n_components=2, random_state=42)
                tsne_2 = TSNE(n_components=2, random_state=42)
                preds_1_tsne = tsne_1.fit_transform(preds_1)
                preds_2_tsne = tsne_2.fit_transform(preds_2)
                save_dir_name = 'two_stage_figures'
            else:
                tsne_1 = TSNE(n_components=2, random_state=42)
                tsne_2 = TSNE(n_components=2, random_state=42)
                preds_1_tsne = tsne_1.fit_transform(feas_1)
                preds_2_tsne = tsne_2.fit_transform(feas_2)
                save_dir_name = 'two_stage_figures_feas'

            # 保存文件夹
            if pseduo_label:
                figures_dir = os.path.join(result_path, save_dir_name, f'seed{seed}_sub_idx{idx}_pesudo_label_1')
            else:
                figures_dir = os.path.join(result_path, save_dir_name, f'seed{seed}_sub_idx{idx}_1')
                
            os.makedirs(figures_dir, exist_ok=True)

            # Plot preds_1
            plt.figure(figsize=(7, 6))
            if pseduo_label:
                scatter = plt.scatter(preds_1_tsne[:, 0], preds_1_tsne[:, 1], c=preds_labels_1, cmap='tab10', s=60)
            else:
                scatter = plt.scatter(preds_1_tsne[:, 0], preds_1_tsne[:, 1], c=true_labels_combined, cmap='tab10', s=60)
            plt.xticks([])
            plt.yticks([])
            # plt.title(f't-SNE of preds_1 (instance {num_instance})')
            plt.savefig(os.path.join(figures_dir, f'instance_{num_instance}_preds1.png'))
            plt.savefig(os.path.join(figures_dir, f'instance_{num_instance}_preds1.svg'))
            plt.close()

            # Plot preds_2
            plt.figure(figsize=(7, 6))
            if pseduo_label:
                scatter = plt.scatter(preds_2_tsne[:, 0], preds_2_tsne[:, 1], c=preds_labels_2, cmap='tab10', s=60)
            else:
                scatter = plt.scatter(preds_2_tsne[:, 0], preds_2_tsne[:, 1], c=true_labels_combined, cmap='tab10', s=60)
            plt.xticks([])
            plt.yticks([])
            # plt.title(f't-SNE of preds_2 (instance {num_instance})')
            plt.savefig(os.path.join(figures_dir, f'instance_{num_instance}_preds2.png'))
            plt.savefig(os.path.join(figures_dir, f'instance_{num_instance}_preds2.svg'))
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

        """
        # update multiple models, independently, from the source models
        total_acc = []
        total_acc_1 = []
        total_acc_2 = []
        total_acc_3 = []
        for s in [1,2,3,4,5]:
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
                # load_two_stage(args, result_path=str(args.result_dir), seed=args.SEED, idx=args.idt)
                #avg_acc, avg_acc1 = calc_review_acc(str(args.result_dir), args.SEED, args.idt)
                avg_acc, avg_acc1, avg_acc2, avg_acc3 = calc_pred_acc(str(args.result_dir), args.SEED, args.idt)
                
                mean_sub_acc.append(avg_acc)
                mean_sub_acc_1.append(avg_acc1)
                mean_sub_acc_2.append(avg_acc2)
                mean_sub_acc_3.append(avg_acc3)
                
            #print('Mean accuracy over subjects: {:.4f}'.format(sum(mean_sub_acc)/len(mean_sub_acc)))
            total_acc.append(sum(mean_sub_acc)/len(mean_sub_acc))
            total_acc_1.append(sum(mean_sub_acc_1)/len(mean_sub_acc_1))
            total_acc_2.append(sum(mean_sub_acc_2)/len(mean_sub_acc_2))
            total_acc_3.append(sum(mean_sub_acc_3)/len(mean_sub_acc_3))
        print('Overall average accuracy over subjects: {:.4f}'.format(sum(total_acc)/len(total_acc)))
        print('Overall average accuracy 1 over subjects: {:.4f}'.format(sum(total_acc_1)/len(total_acc_1)))
        print('Overall average accuracy 2 over subjects: {:.4f}'.format(sum(total_acc_2)/len(total_acc_2)))
        print('Overall average accuracy 3 over subjects: {:.4f}'.format(sum(total_acc_3)/len(total_acc_3)))
        """
        for s in [1,]:
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
            for idt in [0]:
                fix_random_seed(args.SEED)  # fix the seed
                args.idt = idt
                source_str = 'Except_S' + str(idt)
                target_str = 'S' + str(idt)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                review_visualization_feas_1(str(args.result_dir), args.SEED, args.idt)
            
    
            
