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

def load_all_buffers(args, buffer_type, result_path, seed, idx):
    """
    按 num_instance 顺序自动读取指定 seed 和 idx 下所有 buffer 文件内容。
    Args:
        buffer_type (str): 'online' 或 'memory'
        result_path (str): 根目录
        seed (int): 实验 seed
        idx (int): id 号
    Returns:
        List[buffer_data]: 按 num_instance 顺序排列的 buffer 内容列表
    """
    if buffer_type == 'online':
        folder = 'online_visulization_records'
        file_prefix = 'online_buffer_instance_'
    elif buffer_type == 'memory':
        folder = 'memory_visulization_records'
        file_prefix = 'memory_buffer_instance_'
    else:
        raise ValueError("buffer_type must be 'online' or 'memory'")

    dir_path = os.path.join(result_path, folder, f'seed{seed}_sub_idx{idx}')
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    # 匹配所有 buffer_instance 文件
    buffer_files = []
    pattern = re.compile(rf'{file_prefix}(\d+)\.pt$')
    for fname in os.listdir(dir_path):
        match = pattern.match(fname)
        if match:
            num_instance = int(match.group(1))
            buffer_files.append((num_instance, fname))
    # 按 num_instance 排序
    buffer_files.sort(key=lambda x: x[0])

    # 依次读取
    buffers = []
    for num_instance, fname in buffer_files:
        if args.data_name in ["WBCIC-SHU-3C","BNCI2014_004-test"]:
            if num_instance % args.test_batch==0:
                file_path = os.path.join(dir_path, fname)
                buffers.append(torch.load(file_path))
        else:
            file_path = os.path.join(dir_path, fname)
            buffers.append(torch.load(file_path))
    return buffers


def visualize_memory_time_stamp_interval(args, result_path, seed, idx):
    buffers = load_all_buffers(args, 'memory', result_path, seed, idx)
    mean_intervals = []
    for buffer in buffers:
        intervals = []
        for class_items in buffer:
            for i, instance in enumerate(class_items):
                if i > 0:
                    #interval = instance['time_stamp_interval'] if isinstance(instance, dict) else instance.time_stamp_interval
                    interval = instance['time_stamp'] - class_items[i-1]['time_stamp'] if isinstance(instance, dict) else instance.time_stamp - class_items[i-1].time_stamp
                    intervals.append(interval)
        if intervals:
            mean_intervals.append(np.mean(intervals))
        else:
            mean_intervals.append(0)
    plt.figure()
    plt.plot(range(len(mean_intervals)), mean_intervals, marker='o')
    plt.xlabel('num_instance')
    plt.ylabel('Mean time_stamp_interval')
    plt.title('Memory Buffer Mean time_stamp_interval')
    plt.grid()
    # 保存图片
    if args.data_name in ["WBCIC-SHU-3C"]:
        # 提取result_path的最后两级目录
        last2, last1 = os.path.basename(os.path.dirname(result_path)), os.path.basename(result_path)
        fig_dir = os.path.join("./logs/Baselines-WBCIC-SHU-3C-e300-b64/proposed", last2, last1, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f'memory_buffer_mean_time_stamp_interval_seed{seed}_sub{idx}.png')
    else:
        fig_dir = os.path.join(result_path, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f'memory_buffer_mean_time_stamp_interval_seed{seed}_sub{idx}.png')
    plt.savefig(fig_path)
    plt.close()


def visualize_memory_time_stamp_interval_subs_methods(args):
    seed = args.SEED
    buffer_names = args.dataset_info['buffer_name']
    log_paths = args.dataset_info['log_path']
    sub_num = args.dataset_info['sub_num']
    mean_intervals_methods = []

    for buffer_name, result_path in zip(buffer_names, log_paths):
        mean_intervals_subs = []
        for idx in range(sub_num):
            buffers = load_all_buffers(args, 'memory', result_path, seed, idx)
            mean_intervals = []
            for buffer in buffers:
                intervals = []
                for class_items in buffer:
                    for i, instance in enumerate(class_items):
                        if args.data_name not in ["BNCI2014_004-test"]: 
                            if i > 0:
                                interval = instance['time_stamp'] - class_items[i-1]['time_stamp'] if isinstance(instance, dict) else instance.time_stamp - class_items[i-1].time_stamp
                                intervals.append(interval)
                        elif buffer_name in ['Ours']:
                            if i > 0:
                                interval = instance['time_stamp_interval'] if isinstance(instance, dict) else instance.time_stamp_interval
                                intervals.append(interval)
                        else:
                            if i > 0:
                                interval = instance['time_stamp'] - class_items[i-1]['time_stamp'] if isinstance(instance, dict) else instance.time_stamp - class_items[i-1].time_stamp
                                intervals.append(interval)
                if intervals:
                    mean_intervals.append(np.mean(intervals))
                else:
                    mean_intervals.append(0)
            if args.data_name not in ["BNCI2014_004-test"]:
                mean_intervals_subs.append(mean_intervals)
            elif idx != 1:
                mean_intervals_subs.append(mean_intervals)
        # 对 subs 维度取均值
        mean_intervals_avg = np.mean(mean_intervals_subs, axis=0)
        mean_intervals_methods.append(mean_intervals_avg)

    # 绘图
    plt.figure()
    for i, mean_intervals_avg in enumerate(mean_intervals_methods):
        plt.plot(range(len(mean_intervals_avg)), mean_intervals_avg, marker='o', label=str(buffer_names[i]))

    plt.xlabel('num_instance')
    plt.ylabel('Mean time_stamp_interval')
    plt.title('Memory Buffer Mean time_stamp_interval (Mean over subs)')
    plt.legend()
    plt.grid()

    # 保存图片
    if args.data_name in ["WBCIC-SHU-3C"]:
        fig_dir = os.path.join("./logs/Baselines-WBCIC-SHU-3C-e300-b64/proposed", "figures")
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f'memory_buffer_mean_time_stamp_interval_all_seed{seed}.png')
    else:
        fig_dir = os.path.join(log_paths[0], 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, f'memory_buffer_mean_time_stamp_interval_all_seed{seed}.png')
    plt.savefig(fig_path)
    plt.close()


def visualize_online_buffer_accuracy_entropy(args, result_path, seed, idx):
    buffers = load_all_buffers(args, 'online', result_path, seed, idx)
    accuracies = []
    mean_entropies = []
    for buffer in buffers:
        preds = []
        labels = []
        entropies = []
        for instance in buffer:
            logit = instance['logit'] if isinstance(instance, dict) else instance.logit
            label = instance['label'] if isinstance(instance, dict) else instance.label
            pred = torch.argmax(logit).item() if torch.is_tensor(logit) else np.argmax(logit)
            preds.append(pred)
            labels.append(label)
            if torch.is_tensor(logit):
                prob = torch.softmax(logit, dim=-1).cpu().numpy()
            else:
                prob = np.exp(logit) / np.sum(np.exp(logit))
            C = prob.shape[0]
            entropies.append(entropy(prob)/np.log(C))
        if labels:
            acc = np.mean(np.array(preds) == np.array(labels))
            accuracies.append(acc)
            mean_entropies.append(np.mean(entropies))
        else:
            accuracies.append(0)
            mean_entropies.append(0)
    # 保存图片
    fig, ax1 = plt.subplots()
    x = range(len(accuracies))
    ax1.plot(x, accuracies, 'b-o', label='Accuracy')
    ax1.set_xlabel('num_instance')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(x, mean_entropies, 'r-s', label='Mean Entropy')
    ax2.set_ylabel('Mean Entropy', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.title('Online Buffer Accuracy & Mean Entropy')
    fig.tight_layout()
    # 创建 figures 文件夹
    fig_dir = os.path.join(result_path, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f'online_buffer_accuracy_entropy_seed{seed}_sub{idx}.png')
    plt.savefig(fig_path)
    plt.close(fig)


def compare_online_memory_entropy_accuracy(args, result_path, seed, idx):
    # 加载 buffer
    memory_buffers = load_all_buffers(args, 'memory', result_path, seed, idx)
    online_buffers = load_all_buffers(args, 'online', result_path, seed, idx)

    # 计算 memory buffer 平均熵
    memory_mean_entropies = []
    for buffer in memory_buffers:
        entropies = []
        for class_items in buffer:
            for instance in class_items:
                logit = instance['logit'] if isinstance(instance, dict) else instance.logit
                if torch.is_tensor(logit):
                    prob = torch.softmax(logit, dim=-1).cpu().numpy()
                else:
                    prob = np.exp(logit) / np.sum(np.exp(logit))
                C = prob.shape[0]
                entropies.append(entropy(prob)/np.log(C))
        if entropies:
            memory_mean_entropies.append(np.mean(entropies))
        else:
            memory_mean_entropies.append(0)

    # 计算 online buffer 平均熵和准确率
    online_mean_entropies = []
    online_accuracies = []
    for buffer in online_buffers:
        preds = []
        labels = []
        entropies = []
        for instance in buffer:
            logit = instance['logit'] if isinstance(instance, dict) else instance.logit
            label = instance['label'] if isinstance(instance, dict) else instance.label
            pred = torch.argmax(logit).item() if torch.is_tensor(logit) else np.argmax(logit)
            preds.append(pred)
            labels.append(label)
            if torch.is_tensor(logit):
                prob = torch.softmax(logit, dim=-1).cpu().numpy()
            else:
                prob = np.exp(logit) / np.sum(np.exp(logit))
            C = prob.shape[0]
            entropies.append(entropy(prob)/np.log(C))
        if labels:
            acc = np.mean(np.array(preds) == np.array(labels))
            online_accuracies.append(acc)
            online_mean_entropies.append(np.mean(entropies))
        else:
            online_accuracies.append(0)
            online_mean_entropies.append(0)

    # 对齐长度
    min_len = min(len(online_mean_entropies), len(memory_mean_entropies))
    online_mean_entropies = online_mean_entropies[:min_len]
    memory_mean_entropies = memory_mean_entropies[:min_len]
    online_accuracies = online_accuracies[:min_len]

    # 计算熵比值
    entropy_ratios = [o/m if m > 0 else 0 for o, m in zip(online_mean_entropies, memory_mean_entropies)]

    # 绘图
    fig, ax1 = plt.subplots()
    x = range(min_len)
    ax1.plot(x, online_accuracies, 'b-o', label='Online Buffer Accuracy')
    ax1.set_xlabel('num_instance')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(x, online_mean_entropies, 'r-s', label='Online Mean Entropy')
    ax2.plot(x, entropy_ratios, 'g-^', label='Entropy Ratio (Online/Memory)')
    ax2.set_ylabel('Entropy / Ratio', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.title('Online Buffer Accuracy, Entropy, and Entropy Ratio')
    fig.tight_layout()
    # 保存图片
    fig_dir = os.path.join(result_path, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f'online_vs_memory_entropy_accuracy_seed{seed}_sub{idx}.png')
    plt.savefig(fig_path)
    plt.close(fig)

    # 可视化熵比值与accuracy的关系
    plt.figure()
    sns.scatterplot(x=entropy_ratios, y=online_accuracies)
    slope, intercept, r_value, p_value, std_err = linregress(entropy_ratios, online_accuracies)
    x_vals = np.array(entropy_ratios)
    plt.plot(x_vals, intercept + slope * x_vals, 'r--', label=f'Fit: $R^2$={r_value**2:.2f}')
    plt.xlabel('Entropy Ratio (Online/Memory)')
    plt.ylabel('Online Buffer Accuracy')
    plt.title('Accuracy vs. Entropy Ratio')
    plt.legend()
    plt.grid()
    # 保存图片
    fig_path2 = os.path.join(fig_dir, f'accuracy_vs_entropy_ratio_seed{seed}_sub{idx}.png')
    plt.savefig(fig_path2)
    plt.close()

    # 可视化熵比值与accuracy的关系
    plt.figure()
    sns.scatterplot(x=online_mean_entropies, y=online_accuracies)
    slope, intercept, r_value, p_value, std_err = linregress(online_mean_entropies, online_accuracies)
    x_vals = np.array(online_mean_entropies)
    plt.plot(x_vals, intercept + slope * x_vals, 'r--', label=f'Fit: $R^2$={r_value**2:.2f}')
    plt.xlabel('Entropy Ratio')
    plt.ylabel('Online Buffer Accuracy')
    plt.title('Accuracy vs. Entropy Ratio')
    plt.legend()
    plt.grid()
    # 保存图片
    fig_path2 = os.path.join(fig_dir, f'accuracy_vs_entropy_seed{seed}_sub{idx}.png')
    plt.savefig(fig_path2)
    plt.close()




def compare_online_memory_entropy_accuracy_Wasserstein(args, result_path, seed, idx):
    # 加载 buffer
    memory_buffers = load_all_buffers(args, 'memory', result_path, seed, idx)
    online_buffers = load_all_buffers(args, 'online', result_path, seed, idx)

    # 计算 memory buffer 每步的熵分布
    memory_entropies_list = []
    for buffer in memory_buffers:
        entropies = []
        for class_items in buffer:
            for instance in class_items:
                logit = instance['logit'] if isinstance(instance, dict) else instance.logit
                if torch.is_tensor(logit):
                    prob = torch.softmax(logit, dim=-1).cpu().numpy()
                else:
                    prob = np.exp(logit) / np.sum(np.exp(logit))
                C = prob.shape[0]
                entropies.append(entropy(prob)/np.log(C))
        memory_entropies_list.append(entropies)

    # 计算 online buffer 每步的熵分布和准确率
    online_entropies_list = []
    online_accuracies = []
    for buffer in online_buffers:
        preds = []
        labels = []
        entropies = []
        for instance in buffer:
            logit = instance['logit'] if isinstance(instance, dict) else instance.logit
            label = instance['label'] if isinstance(instance, dict) else instance.label
            pred = torch.argmax(logit).item() if torch.is_tensor(logit) else np.argmax(logit)
            preds.append(pred)
            labels.append(label)
            if torch.is_tensor(logit):
                prob = torch.softmax(logit, dim=-1).cpu().numpy()
            else:
                prob = np.exp(logit) / np.sum(np.exp(logit))
            C = prob.shape[0]
            entropies.append(entropy(prob)/np.log(C))
        online_entropies_list.append(entropies)
        if labels:
            acc = np.mean(np.array(preds) == np.array(labels))
            online_accuracies.append(acc)
        else:
            online_accuracies.append(0)

    # 对齐长度
    min_len = min(len(online_entropies_list), len(memory_entropies_list))
    online_entropies_list = online_entropies_list[:min_len]
    memory_entropies_list = memory_entropies_list[:min_len]
    online_accuracies = online_accuracies[:min_len]

    # 计算每步 Wasserstein 距离
    wasserstein_distances = []
    for online_e, memory_e in zip(online_entropies_list, memory_entropies_list):
        if len(online_e) > 0 and len(memory_e) > 0:
            wd = wasserstein_distance(online_e, memory_e)
        else:
            wd = 0
        wasserstein_distances.append(wd)

    # 绘图
    fig, ax1 = plt.subplots()
    x = range(min_len)
    ax1.plot(x, online_accuracies, 'b-o', label='Online Buffer Accuracy')
    ax1.set_xlabel('num_instance')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(x, wasserstein_distances, 'g-^', label='Wasserstein Distance (Entropy)')
    ax2.set_ylabel('Wasserstein Distance', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.title('Online Buffer Accuracy and Wasserstein Distance of Entropy')
    fig.tight_layout()
    # 保存图片
    fig_dir = os.path.join(result_path, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f'online_vs_memory_entropy_wasserstein_seed{seed}_sub{idx}.png')
    plt.savefig(fig_path)
    plt.close(fig)
    
    # 可视化熵比值与accuracy的关系
    plt.figure()
    sns.scatterplot(x=wasserstein_distances, y=online_accuracies)
    slope, intercept, r_value, p_value, std_err = linregress(wasserstein_distances, online_accuracies)
    x_vals = np.array(wasserstein_distances)
    plt.plot(x_vals, intercept + slope * x_vals, 'r--', label=f'Fit: $R^2$={r_value**2:.2f}')
    plt.xlabel('Wasserstein Distances (Online/Memory)')
    plt.ylabel('Online Buffer Accuracy')
    plt.title('Accuracy vs. Entropy Ratio')
    plt.legend()
    plt.grid()
    # 保存图片
    fig_path2 = os.path.join(fig_dir, f'accuracy_vs_entropy_wasserstein_seed{seed}_sub{idx}.png')
    plt.savefig(fig_path2)
    plt.close()


def compare_online_memory_entropy_accuracy_ZScore(args, result_path, seed, idx):
    # 加载 buffer
    memory_buffers = load_all_buffers(args, 'memory', result_path, seed, idx)
    online_buffers = load_all_buffers(args, 'online', result_path, seed, idx)

    # 计算 memory buffer 每步的熵分布
    memory_entropies_list = []
    for buffer in memory_buffers:
        entropies = []
        for class_items in buffer:
            for instance in class_items:
                logit = instance['logit'] if isinstance(instance, dict) else instance.logit
                if torch.is_tensor(logit):
                    prob = torch.softmax(logit, dim=-1).cpu().numpy()
                else:
                    prob = np.exp(logit) / np.sum(np.exp(logit))
                C = prob.shape[0]
                entropies.append(entropy(prob)/np.log(C))
        memory_entropies_list.append(entropies)

    # 计算 online buffer 每步的熵分布和准确率
    online_entropies_list = []
    online_accuracies = []
    for buffer in online_buffers:
        preds = []
        labels = []
        entropies = []
        for instance in buffer:
            logit = instance['logit'] if isinstance(instance, dict) else instance.logit
            label = instance['label'] if isinstance(instance, dict) else instance.label
            pred = torch.argmax(logit).item() if torch.is_tensor(logit) else np.argmax(logit)
            preds.append(pred)
            labels.append(label)
            if torch.is_tensor(logit):
                prob = torch.softmax(logit, dim=-1).cpu().numpy()
            else:
                prob = np.exp(logit) / np.sum(np.exp(logit))
            C = prob.shape[0]
            entropies.append(entropy(prob)/np.log(C))
        online_entropies_list.append(entropies)
        if labels:
            acc = np.mean(np.array(preds) == np.array(labels))
            online_accuracies.append(acc)
        else:
            online_accuracies.append(0)

    # 对齐长度
    min_len = min(len(online_entropies_list), len(memory_entropies_list))
    online_entropies_list = online_entropies_list[:min_len]
    memory_entropies_list = memory_entropies_list[:min_len]
    online_accuracies = online_accuracies[:min_len]

    # 计算每步 Z-Score
    z_scores = []
    for online_e, memory_e in zip(online_entropies_list, memory_entropies_list):
        if len(memory_e) > 1:
            mem_mean = np.mean(memory_e)
            mem_std = np.std(memory_e)
            online_mean = np.mean(online_e) if len(online_e) > 0 else 0
            z = (online_mean - mem_mean) / (mem_std + 1e-8)
        else:
            z = 0
        z_scores.append(z)

    """
    # 绘图
    fig, ax1 = plt.subplots()
    x = range(min_len)
    ax1.plot(x, online_accuracies, 'b-o', label='Online Buffer Accuracy')
    ax1.set_xlabel('num_instance')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(x, z_scores, 'r-^', label='Z-Score (Entropy)')
    ax2.set_ylabel('Z-Score', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.title('Online Buffer Accuracy and Z-Score of Entropy')
    fig.tight_layout()
    # 保存图片
    fig_dir = os.path.join(result_path, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f'online_vs_memory_entropy_zscore_seed{seed}_sub{idx}.png')
    plt.savefig(fig_path)
    plt.close(fig)

    # 可视化熵比值与accuracy的关系
    plt.figure()
    sns.scatterplot(x=z_scores, y=online_accuracies)
    slope, intercept, r_value, p_value, std_err = linregress(z_scores, online_accuracies)
    x_vals = np.array(z_scores)
    plt.plot(x_vals, intercept + slope * x_vals, 'r--', label=f'Fit: $R^2$={r_value**2:.2f}')
    plt.xlabel('Wasserstein Distances (Online/Memory)')
    plt.ylabel('Online Buffer Accuracy')
    plt.title('Accuracy vs. Entropy Ratio')
    plt.legend()
    plt.grid()
    # 保存图片
    fig_path2 = os.path.join(fig_dir, f'accuracy_vs_entropy_zscore_seed{seed}_sub{idx}.png')
    plt.savefig(fig_path2)
    plt.close()
    """

    rho, p_value = spearmanr(z_scores, online_accuracies)
    print(f"Spearman's Rank Correlation (ρ) between Z-Scores and Online Accuracies: {rho:.4f}, p-value: {p_value:.4g}")


def compare_online_memory_entropy_accuracy_ZScore_bins(args, result_path, seed, idx, smooth_zscore=False):
    # 加载 buffer
    memory_buffers = load_all_buffers(args, 'memory', result_path, seed, idx)
    online_buffers = load_all_buffers(args, 'online', result_path, seed, idx)

    # 1. 计算 memory buffer 每步的熵分布
    memory_entropies_list = []
    for buffer in memory_buffers:
        entropies = []
        for class_items in buffer:
            for instance in class_items:
                logit = instance['logit'] if isinstance(instance, dict) else instance.logit
                # 兼容 Tensor 和 numpy
                if torch.is_tensor(logit):
                    prob = torch.softmax(logit, dim=-1).cpu().numpy()
                else:
                    # 假设 logit 是 numpy 数组，增加稳定性防止溢出
                    logit_safe = logit - np.max(logit) 
                    prob = np.exp(logit_safe) / np.sum(np.exp(logit_safe))
                
                C = prob.shape[0]
                # 归一化熵到 [0, 1]
                entropies.append(entropy(prob) / np.log(C))
        memory_entropies_list.append(entropies)

    # 2. 计算 online buffer 每步的熵分布和准确率
    online_entropies_list = []
    online_accuracies = []
    for buffer in online_buffers:
        preds = []
        labels = []
        entropies = []
        for instance in buffer:
            logit = instance['logit'] if isinstance(instance, dict) else instance.logit
            label = instance['label'] if isinstance(instance, dict) else instance.label
            pred = torch.argmax(logit).item() if torch.is_tensor(logit) else np.argmax(logit)
            
            preds.append(pred)
            labels.append(label)
            
            if torch.is_tensor(logit):
                prob = torch.softmax(logit, dim=-1).cpu().numpy()
            else:
                logit_safe = logit - np.max(logit)
                prob = np.exp(logit_safe) / np.sum(np.exp(logit_safe))
            C = prob.shape[0]
            entropies.append(entropy(prob) / np.log(C))
            
        online_entropies_list.append(entropies)
        
        if labels:
            acc = np.mean(np.array(preds) == np.array(labels))
            online_accuracies.append(acc)
        else:
            online_accuracies.append(0)

    # 3. 对齐长度
    min_len = min(len(online_entropies_list), len(memory_entropies_list))
    online_entropies_list = online_entropies_list[:min_len]
    memory_entropies_list = memory_entropies_list[:min_len]
    online_accuracies = online_accuracies[:min_len]

    # 4. 计算每步 Z-Score (引入平滑机制)
    z_scores = []
    
    # --- 新增配置：平滑参数 ---
    beta = 0.8        # 动量因子 (0.7~0.9 效果较好)
    smooth_z = 0.0    # 初始化平滑变量
    
    for i, (online_e, memory_e) in enumerate(zip(online_entropies_list, memory_entropies_list)):
        if len(memory_e) > 1:
            mem_mean = np.mean(memory_e)
            mem_std = np.std(memory_e)
            online_mean = np.mean(online_e) if len(online_e) > 0 else 0
            
            # 计算原始 Z-Score
            raw_z = (online_mean - mem_mean) / (mem_std + 1e-8)
        else:
            raw_z = 0.0
        
        # --- 核心修改：Momentum Smoothing ---
        if i == 0:
            smooth_z = raw_z # 第一步直接赋值，避免冷启动偏差
        else:
            smooth_z = beta * smooth_z + (1 - beta) * raw_z

        if smooth_zscore:   
            z_scores.append(smooth_z)
        else:
            z_scores.append(raw_z)

    # --- 核心修改：分箱分析 (Binning Analysis) 使用 0.5 阈值 ---
    print(f"\n--- Binning Analysis for Subject {idx} (Smoothed Z, Thr=0.5) ---")
    
    z_scores_arr = np.array(z_scores)
    accuracies_arr = np.array(online_accuracies)
    
    # 定义分组掩码 (阈值修改为 0.5)
    # Group 1: Safe (Z <= 0)
    mask_safe = z_scores_arr <= 0.0
    # Group 2: Warning (0 < Z <= 0.5)
    mask_warning = (z_scores_arr > 0.0) & (z_scores_arr <= 0.50)
    # Group 3: Critical (Z > 0.5)
    mask_critical = z_scores_arr > 0.50
    
    # 计算各组的平均精度
    acc_safe = np.mean(accuracies_arr[mask_safe]) if np.any(mask_safe) else 0.0
    acc_warning = np.mean(accuracies_arr[mask_warning]) if np.any(mask_warning) else 0.0
    acc_critical = np.mean(accuracies_arr[mask_critical]) if np.any(mask_critical) else 0.0
    
    # 计算样本数量
    count_safe = np.sum(mask_safe)
    count_warning = np.sum(mask_warning)
    count_critical = np.sum(mask_critical)
    
    print(f"Safe Zone     (Z <= 0)  : Acc = {acc_safe:.4f} (Count: {count_safe})")
    print(f"Warning Zone  (0<Z<=0.5): Acc = {acc_warning:.4f} (Count: {count_warning})")
    print(f"Critical Zone (Z > 0.5) : Acc = {acc_critical:.4f} (Count: {count_critical})")

    # 绘图保存：Smoothed Z-Score Binning 柱状图
    if args.data_name in ["WBCIC-SHU-3C"]:
        # 提取result_path的最后两级目录
        last2, last1 = os.path.basename(os.path.dirname(result_path)), os.path.basename(result_path)
        fig_dir = os.path.join("./logs/Baselines-WBCIC-SHU-3C-e300-b64/proposed", last2, last1, "figures")
        os.makedirs(fig_dir, exist_ok=True)
    else:
        fig_dir = os.path.join(result_path, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    groups = ['Safe\n(Z<=0)', 'Warning\n(0<Z<=0.5)', 'Critical\n(Z>0.5)']
    acc_values = [acc_safe, acc_warning, acc_critical]
    counts = [count_safe, count_warning, count_critical]
    
    bars = plt.bar(groups, acc_values, color=['#2ecc71', '#f1c40f', '#e74c3c'], alpha=0.8)
    plt.ylabel('Average Accuracy')
    plt.title(f'Acc by Smoothed Z-Score (Sub {idx})')
    plt.ylim(0, 1.0)
    
    for bar, acc, count in zip(bars, acc_values, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{acc:.2f}\n(n={count})',
                 ha='center', va='bottom', fontsize=10)
    
    if smooth_zscore:
        bin_fig_path = os.path.join(fig_dir, f'zscore_binning_smooth_seed{seed}_sub{idx}.png')
    else:
        bin_fig_path = os.path.join(fig_dir, f'zscore_binning_seed{seed}_sub{idx}.png')
    plt.savefig(bin_fig_path)
    plt.close()
    
    # 5. 计算 Spearman 相关性 (依然计算，看看平滑后是否有提升)
    rho, p_value = spearmanr(z_scores, online_accuracies)
    print(f"Spearman's Rank Correlation (ρ): {rho:.4f}, p-value: {p_value:.4g}")
    
    return rho, p_value, [acc_safe, acc_warning, acc_critical]

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
        for s in [1]:
            args.SEED = s

            fix_random_seed(args.SEED)
            torch.backends.cudnn.deterministic = True

            args.data = data_name

            args.local_dir = data_path + str(data_name) + '/'
            args.result_dir = log_path
            
            """for idt in range(N):
                fix_random_seed(args.SEED)  # fix the seed
                args.idt = idt
                source_str = 'Except_S' + str(idt)
                target_str = 'S' + str(idt)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                
                visualize_memory_time_stamp_interval(args, result_path=str(args.result_dir), seed=args.SEED, idx=args.idt)
                #visualize_online_buffer_accuracy_entropy(args, result_path=str(args.result_dir), seed=args.SEED, idx=args.idt)
                #compare_online_memory_entropy_accuracy_ZScore_bins(args, result_path=str(args.result_dir), seed=args.SEED, idx=args.idt)
            """
            
            visualize_memory_time_stamp_interval_subs_methods(args)