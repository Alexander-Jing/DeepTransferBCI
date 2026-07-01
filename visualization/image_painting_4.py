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

import matplotlib.ticker as ticker
from matplotlib.lines import Line2D



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


def compare_online_memory_entropy_accuracy_ZScore_bins(args, result_path, seed, idx, smooth_zscore=False, save_fig=True):
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
    if save_fig:
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
    
    return rho, p_value, [acc_safe, acc_warning, acc_critical], [int(count_safe), int(count_warning), int(count_critical)]

def plot_average_zscore_bins_all_subjects(args, result_path, seed, N, smooth_zscore=False):
    """
    对所有被试运行 Z-Score 分箱分析，计算各 bin 的平均准确率和平均更新数量，绘制汇总柱状图。

    Args:
        args: 参数对象（需包含 data_name 等字段）
        result_path (str): buffer 文件根目录
        seed (int): 实验 seed
        N (int): 被试总数
        smooth_zscore (bool): 是否使用平滑后的 Z-Score
    """
    all_accs = []    # shape: [valid_N, 3]
    all_counts = []  # shape: [valid_N, 3]

    for idx in range(N):
        try:
            _, _, accs, counts = compare_online_memory_entropy_accuracy_ZScore_bins(
                args, result_path, seed, idx, smooth_zscore=smooth_zscore, save_fig=False
            )
            all_accs.append(accs)
            all_counts.append(counts)
        except Exception as e:
            print(f"Subject {idx} skipped: {e}")
            continue

    valid_N = len(all_accs)
    if valid_N == 0:
        print("No valid subjects found for averaging.")
        return

    all_accs = np.array(all_accs)      # [valid_N, 3]
    all_counts = np.array(all_counts)  # [valid_N, 3]

    mean_accs = np.mean(all_accs, axis=0)
    std_accs = np.std(all_accs, axis=0)
    mean_counts = np.mean(all_counts, axis=0)

    print(f"\n--- Average Binning Results across {valid_N} Subjects ---")
    group_labels = ['Safe (Z<=0)', 'Warning (0<Z<=0.5)', 'Critical (Z>0.5)']
    for label, acc, std, cnt in zip(group_labels, mean_accs, std_accs, mean_counts):
        print(f"{label}: Acc = {acc:.4f} ± {std:.4f} (avg count = {cnt:.1f})")

    # 绘制汇总柱状图
    groups_display = ['Safe\n(Z<=0)', 'Warning\n(0<Z<=0.5)', 'Critical\n(Z>0.5)']
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    x = np.arange(len(groups_display))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, mean_accs, color=colors, alpha=0.8, width=0.5)
    # ax.errorbar(x, mean_accs, yerr=std_accs, fmt='none', color='black', capsize=5, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(groups_display)
    ax.set_ylabel('Average Accuracy')
    ax.set_ylim(0, 1.05)
    suffix_title = 'Smoothed Z' if smooth_zscore else 'Raw Z'
    ax.set_title(f'Avg Acc by {suffix_title}-Score Bins (N={valid_N} Subjects, Seed {seed})')

    for bar, acc, std_val, cnt in zip(bars, mean_accs, std_accs, mean_counts):
        top = bar.get_height() + std_val
        ax.text(bar.get_x() + bar.get_width() / 2., top + 0.012,
                f'{acc:.3f}±{std_val:.3f}\n(avg n={cnt:.1f})',
                ha='center', va='bottom', fontsize=9)

    # 确定保存路径（与单被试函数逻辑一致）
    fig_dir = os.path.join('./visualization/figures', 'memory_buffer_z_score')
    os.makedirs(fig_dir, exist_ok=True)

    suffix = 'smooth' if smooth_zscore else 'raw'
    fig_path = os.path.join(fig_dir, f'dataset_{args.data_name}_zscore_binning_{suffix}_avg_allsubs_seed{seed}.png')
    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved average figure to: {fig_path}")




# ==========================================
# 1. 全局样式配置 (严格对齐你之前的学术风格)
# ==========================================
def setup_plot_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 22
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['svg.fonttype'] = 'none'

def generate_zscore_accuracy_plot(save_dir):
    setup_plot_style()
    
    # --- [数据定义] ---
    zones = ['Safe\n(Z≤0)', 'Warning\n(0<Z≤0.5)', 'Critical\n(Z>0.5)']
    x_indexes = np.arange(len(zones))
    
    # 数据集 1: BNCI-2a (Subject 8)
    acc_2a = [0.7936, 0.7250, 0.5312]
    count_2a = [43, 25, 4]
    
    # 数据集 2: BNCI-2b (Average)
    acc_2b = [0.8244, 0.7829, 0.8262]
    count_2b = [24.1, 11.4, 3.9]
    
    # 数据集 3: SHU-3C (Average)
    acc_shu = [0.8001, 0.7370, 0.6572]
    count_shu = [29.8, 45.5, 36.6]

    # --- [绘图开始] ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.15, 0.18, 0.80, 0.75])
    
    # 彻底关闭网格
    ax.grid(False)
    
    # 设置黑色封闭边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)

    # 设置向外的刻度
    ax.tick_params(axis='both', direction='out', bottom=True, left=True, 
                   top=False, right=False, length=6, width=1.2, labelsize=20)

    # --- [核心绘制：折线 + 气泡点] ---
    # 定义基础样式
    datasets = [
        {'name': 'BCI-IV 2a', 'acc': acc_2a, 'count': count_2a, 'color': '#e41a1c', 'marker': 'o', 'ls': '-'},
        {'name': 'BCI-IV 2b', 'acc': acc_2b, 'count': count_2b, 'color': '#377eb8', 'marker': 's', 'ls': '--'},
        {'name': 'SHU-3C',    'acc': acc_shu, 'count': count_shu, 'color': '#4daf4a', 'marker': '^', 'ls': '-.'}
    ]

    for ds in datasets:
        # 先画折线
        ax.plot(x_indexes, ds['acc'], label=ds['name'], color=ds['color'], 
                linestyle=ds['ls'], linewidth=2.5, zorder=2)
        
        # 再画气泡点：s 参数控制面积，这里通过 count 线性映射
        # 为了让气泡清晰，乘一个系数（如 15），你可以根据视觉效果微调
        sizes = np.array(ds['count']) * 15 
        ax.scatter(x_indexes, ds['acc'], s=sizes, color=ds['color'], 
                   marker=ds['marker'], edgecolor='black', linewidth=1, alpha=0.8, zorder=3)

    # --- [坐标轴配置] ---
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(zones)
    ax.set_xlabel('Entropy Z-score Zones', labelpad=20, fontsize=24)
    ax.set_ylabel('Average Accuracy', labelpad=20, fontsize=24)
    
    # 纵轴显示百分比格式
    ax.set_ylim(0.4, 0.95)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    # --- [图例配置] ---
    # 由于 scatter 点的大小不同，图例可能会变样，这里我们通过 proxy artist 修复图例大小统一
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=ds['color'], lw=2, ls=ds['ls'], 
                              marker=ds['marker'], markersize=10, 
                              markerfacecolor=ds['color'], markeredgecolor='black', label=ds['name']) 
                       for ds in datasets]
    
    legend = ax.legend(handles=legend_elements, loc='lower left', frameon=True, 
                       edgecolor='black', fontsize=18, framealpha=1.0, facecolor='white')
    legend.set_zorder(10)

    # --- [添加 Count 说明注记] ---
    ax.text(0.98, 0.05, '* Marker size represents relative count in each zone', 
            transform=ax.transAxes, fontsize=14, ha='right', fontstyle='italic')

    # --- [保存] ---
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    file_base = "Zscore_Accuracy_Bubble_Analysis"
    plt.savefig(os.path.join(save_dir, f'{file_base}.svg'), format='svg')
    plt.savefig(os.path.join(save_dir, f'{file_base}.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 气泡趋势图已生成: {file_base}.svg")


def generate_zscore_accuracy_dodged_plot(save_dir):
    setup_plot_style()
    
    # --- [数据定义] ---
    zones = ['Safe\n(Z≤0)', 'Warning\n(0<Z≤0.5)', 'Critical\n(Z>0.5)']
    x_base = np.arange(len(zones))
    
    # 数据集 1: BNCI-2a (Subject 8)
    acc_2a = [0.7936, 0.7250, 0.5312]
    count_2a = [43, 25, 4]
    
    # 数据集 2: BNCI-2b (Average)
    acc_2b = [0.8244, 0.7829, 0.8262]
    count_2b = [24.1, 11.4, 3.9]
    
    # 数据集 3: SHU-3C (Average)
    acc_shu = [0.8001, 0.7370, 0.6572]
    count_shu = [29.8, 45.5, 36.6]

    # --- [绘图开始] ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.15, 0.18, 0.80, 0.75])
    
    # 彻底关闭网格
    ax.grid(False)
    
    # 设置黑色封闭边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    # 设置向外的刻度，仅左下显示
    ax.tick_params(axis='both', direction='out', bottom=True, left=True, 
                   top=False, right=False, length=6, width=1.2, labelsize=20)

    # --- [核心优化：设置水平偏移量] ---
    # 为三个数据集分别设置左偏、不偏、右偏
    offsets = [-0.0, 0.0, 0.0] 
    
    datasets = [
        {'name': 'BCI-IV 2a', 'acc': acc_2a, 'count': count_2a, 'color': '#e41a1c', 'marker': 'o', 'ls': '-'},
        {'name': 'BCI-IV 2b', 'acc': acc_2b, 'count': count_2b, 'color': '#377eb8', 'marker': 's', 'ls': '--'},
        {'name': 'SHU-3C',    'acc': acc_shu, 'count': count_shu, 'color': '#4daf4a', 'marker': '^', 'ls': '-.'}
    ]

    for i, ds in enumerate(datasets):
        # 计算偏移后的 X 坐标
        curr_x = x_base + offsets[i]
        
        # 绘制折线
        ax.plot(curr_x, ds['acc'], color=ds['color'], linestyle=ds['ls'], 
                linewidth=2.5, zorder=2, alpha=0.9)
        
        # 绘制气泡点 (设置 alpha 增加通透感，设置 edgecolor 增加辨识度)
        sizes = np.array(ds['count']) * 18  # 略微放大系数
        ax.scatter(curr_x, ds['acc'], s=sizes, color=ds['color'], 
                   marker=ds['marker'], edgecolor='black', linewidth=1.2, 
                   alpha=0.7, zorder=3)

    # --- [坐标轴配置] ---
    ax.set_xticks(x_base)
    ax.set_xticklabels(zones)
    ax.set_xlabel('Entropy $Z$-score Zones', labelpad=20, fontsize=24)
    ax.set_ylabel('Average Accuracy', labelpad=20, fontsize=24)
    
    # 纵轴显示百分比格式，设置合理的范围
    ax.set_ylim(0.4, 0.95)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # 强制显示整数百分比
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    # --- [图例配置] ---
    # 使用 Proxy Artists 确保图例标记大小统一
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=ds['color'], lw=2, ls=ds['ls'], 
                              marker=ds['marker'], markersize=10, 
                              markerfacecolor=ds['color'], markeredgecolor='black', label=ds['name']) 
                       for ds in datasets]
    
    legend = ax.legend(handles=legend_elements, loc='lower left', frameon=True, 
                       edgecolor='black', fontsize=18, framealpha=1.0, facecolor='white')
    legend.set_zorder(10)

    # 添加注释
    ax.text(0.98, 0.05, '* Marker size indicates the sample count in each zone', 
            transform=ax.transAxes, fontsize=14, ha='right', fontstyle='italic')

    # --- [保存] ---
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    file_base = "Zscore_Accuracy_Dodged_Bubble"
    plt.savefig(os.path.join(save_dir, f'{file_base}.svg'), format='svg')
    plt.savefig(os.path.join(save_dir, f'{file_base}.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 偏移优化后的趋势图已生成: {file_base}.svg")


def generate_zscore_accuracy_no_border_plot(save_dir):
    setup_plot_style()
    
    # --- [数据定义] ---
    zones = ['Safe\n(Z≤0)', 'Warning\n(0<Z≤0.5)', 'Critical\n(Z>0.5)']
    x_base = np.arange(len(zones))
    
    # 数据集 1: BNCI-2a (Subject 8)
    acc_2a = [0.7936, 0.7250, 0.5312]
    count_2a = [43, 25, 4]
    
    # 数据集 2: BNCI-2b (Average)
    acc_2b = [0.8244, 0.7829, 0.8262]
    count_2b = [24.1, 11.4, 3.9]
    
    # 数据集 3: SHU-3C (Average)
    acc_shu = [0.8001, 0.7370, 0.6572]
    count_shu = [29.8, 45.5, 36.6]

    # --- [绘图开始] ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.15, 0.18, 0.80, 0.75])
    
    # 彻底关闭网格
    ax.grid(False)
    
    # 设置黑色封闭边框 (矩形框)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    # 设置向外的刻度，仅左下显示
    ax.tick_params(axis='both', direction='out', bottom=True, left=True, 
                   top=False, right=False, length=6, width=1.2, labelsize=20)

    # --- [水平偏移量设置] ---
    offsets = [-0.02, 0.0, 0.02] 
    
    datasets = [
        {'name': 'BCI-IV 2a', 'acc': acc_2a, 'count': count_2a, 'color': '#e41a1c', 'marker': 'o', 'ls': '-'},
        {'name': 'BCI-IV 2b', 'acc': acc_2b, 'count': count_2b, 'color': '#377eb8', 'marker': 's', 'ls': '--'},
        {'name': 'SHU-3C',    'acc': acc_shu, 'count': count_shu, 'color': '#4daf4a', 'marker': '^', 'ls': '-.'}
    ]

    for i, ds in enumerate(datasets):
        curr_x = x_base + offsets[i]
        
        # 绘制折线
        ax.plot(curr_x, ds['acc'], color=ds['color'], linestyle=ds['ls'], 
                linewidth=2.5, zorder=2, alpha=0.9)
        
        # 绘制气泡点：去掉了 edgecolor 和 linewidth
        # 稍微调高 alpha (0.8) 增加色彩饱和度，因为没有了黑边
        sizes = np.array(ds['count']) * 30 
        ax.scatter(curr_x, ds['acc'], s=sizes, color=ds['color'], 
                   marker=ds['marker'], alpha=0.8, zorder=3, edgecolors='none')

    # --- [坐标轴配置] ---
    ax.set_xticks(x_base)
    ax.set_xlim(-0.2, 2.2)
    ax.set_xticklabels(zones)
    ax.set_xlabel('Entropy $Z$-score Zones', labelpad=20, fontsize=24)
    ax.set_ylabel('Average Accuracy', labelpad=20, fontsize=24)
    
    # 纵轴显示百分比格式
    ax.set_ylim(0.45, 0.95)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    # --- [图例配置：去掉了标记点的黑边] ---
    legend_elements = [Line2D([0], [0], color=ds['color'], lw=2, ls=ds['ls'], 
                              marker=ds['marker'], markersize=10, 
                              markerfacecolor=ds['color'], 
                              markeredgecolor='none', # 关键修改：去掉图例黑边
                              label=ds['name']) 
                       for ds in datasets]
    
    legend = ax.legend(handles=legend_elements, loc='lower left', frameon=True, 
                       edgecolor='black', fontsize=18, framealpha=1.0, facecolor='white')
    legend.set_zorder(10)

    # 添加注释
    # ax.text(0.98, 0.05, '* Marker size indicates the sample count in each zone', 
    #        transform=ax.transAxes, fontsize=14, ha='right', fontstyle='italic')

    # --- [保存输出] ---
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    file_base = "Zscore_Accuracy_NoBorder_Dodged"
    plt.savefig(os.path.join(save_dir, f'{file_base}.svg'), format='svg')
    plt.savefig(os.path.join(save_dir, f'{file_base}.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 无黑边气泡图已生成: {file_base}.svg")


def generate_zscore_accuracy_no_border_plot_1(save_dir):
    setup_plot_style()
    
    # --- [数据定义] ---
    zones = ['Safe\n(Z≤0)', 'Warning\n(0<Z≤0.5)', 'Critical\n(Z>0.5)']
    x_base = np.arange(len(zones))
    
    # 数据集 1: BNCI-2a (Subject 8)
    acc_2a = [0.7936, 0.7250, 0.5312]
    count_2a = [43, 25, 4]
    
    # 数据集 2: BNCI-2b (Average)
    acc_2b = [0.8244, 0.7829, 0.8262]
    count_2b = [24.1, 11.4, 3.9]
    
    # 数据集 3: SHU-3C (Average)
    acc_shu = [0.8001, 0.7370, 0.6572]
    count_shu = [29.8, 45.5, 36.6]

    # --- [绘图开始] ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.15, 0.18, 0.80, 0.75])
    
    # 彻底关闭网格
    ax.grid(False)
    
    # 设置黑色封闭边框 (矩形框)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    # 设置向外的刻度，仅左下显示
    ax.tick_params(axis='both', direction='out', bottom=True, left=True, 
                   top=False, right=False, length=6, width=1.2, labelsize=20)

    # --- [水平偏移量设置] ---
    offsets = [-0.1, 0.0, 0.1] 
    
    datasets = [
        {'name': 'BCI-IV 2a', 'acc': acc_2a, 'count': count_2a, 'color': '#e41a1c', 'marker': 'o', 'ls': '-'},
        {'name': 'BCI-IV 2b', 'acc': acc_2b, 'count': count_2b, 'color': '#377eb8', 'marker': 's', 'ls': '--'},
        {'name': 'SHU-3C',    'acc': acc_shu, 'count': count_shu, 'color': '#4daf4a', 'marker': '^', 'ls': '-.'}
    ]

    for i, ds in enumerate(datasets):
        curr_x = x_base + offsets[i]
        
        # 绘制折线
        ax.plot(curr_x, ds['acc'], color=ds['color'], linestyle=ds['ls'], 
                linewidth=2.5, zorder=2, alpha=0.9)
        
        # 绘制气泡点：去掉了 edgecolor 和 linewidth
        # 稍微调高 alpha (0.8) 增加色彩饱和度，因为没有了黑边
        sizes = np.array(ds['count']) * 30 
        ax.scatter(curr_x, ds['acc'], s=sizes, color=ds['color'], 
                   marker=ds['marker'], alpha=0.8, zorder=3, edgecolors='none')

    # --- [坐标轴配置] ---
    ax.set_xticks(x_base)
    ax.set_xlim(-0.35, 2.35)
    ax.set_xticklabels(zones)
    ax.set_xlabel('Entropy $Z$-score Zones', labelpad=20, fontsize=24)
    ax.set_ylabel('Average Accuracy', labelpad=20, fontsize=24)
    
    # 纵轴显示百分比格式
    ax.set_ylim(0.45, 0.95)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    # --- [图例配置：去掉了标记点的黑边] ---
    legend_elements = [Line2D([0], [0], color=ds['color'], lw=2, ls=ds['ls'], 
                              marker=ds['marker'], markersize=10, 
                              markerfacecolor=ds['color'], 
                              markeredgecolor='none', # 关键修改：去掉图例黑边
                              label=ds['name']) 
                       for ds in datasets]
    
    legend = ax.legend(handles=legend_elements, loc='lower left', frameon=True, 
                       edgecolor='black', fontsize=18, framealpha=1.0, facecolor='white')
    legend.set_zorder(10)

    # 添加注释
    # ax.text(0.98, 0.05, '* Marker size indicates the sample count in each zone', 
    #        transform=ax.transAxes, fontsize=14, ha='right', fontstyle='italic')

    # --- [保存输出] ---
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    file_base = "Zscore_Accuracy_NoBorder_Dodged"
    plt.savefig(os.path.join(save_dir, f'{file_base}.svg'), format='svg')
    plt.savefig(os.path.join(save_dir, f'{file_base}.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 无黑边气泡图已生成: {file_base}.svg")

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

        if data_name in ["BNCI2014001-4-all"]:
            args.result_dir = './logs/Baselines-001-test-e300-b64-debugs/ours_debug_m_cls_process_1-BNCI2014001-4-all-EEGNet-4,2-e300-b64/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-2-visualization/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p1'
        elif data_name in ["BNCI2014_004-test"]:
            args.result_dir = './logs/Baselines-004-test-e300-b64/proposed/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-visualization/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p6'
        elif data_name in ["WBCIC-SHU-3C"]:
            args.result_dir = '/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-WBCIC-SHU-3C-e300-b64/proposed/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-visualization/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p1'

        # update multiple models, independently, from the source models
        for s in [2]:
            args.SEED = s

            fix_random_seed(args.SEED)
            torch.backends.cudnn.deterministic = True

            args.data = data_name

            args.local_dir = data_path + str(data_name) + '/'
            # args.result_dir = log_path
            
            # plot_average_zscore_bins_all_subjects(args, result_path=str(args.result_dir), seed=args.SEED, N=N)
            """
            "BNCI2014001-4-all"
            --- Binning Analysis for Subject 8 (Smoothed Z, Thr=0.5) ---
            Safe Zone     (Z <= 0)  : Acc = 0.7936 (Count: 43)
            Warning Zone  (0<Z<=0.5): Acc = 0.7250 (Count: 25)
            Critical Zone (Z > 0.5) : Acc = 0.5312 (Count: 4)
            Spearman's Rank Correlation (ρ): -0.5141, p-value: 3.86e-06

            "BNCI2014_004-test"
            --- Average Binning Results across 9 Subjects ---
            Safe (Z<=0): Acc = 0.8244 ± 0.1467 (avg count = 24.1)
            Warning (0<Z<=0.5): Acc = 0.7829 ± 0.1405 (avg count = 11.4)
            Critical (Z>0.5): Acc = 0.8262 ± 0.0969 (avg count = 3.9)

            "WBCIC-SHU-3C"
            --- Average Binning Results across 11 Subjects ---
            Safe (Z<=0): Acc = 0.8001 ± 0.1808 (avg count = 29.8)
            Warning (0<Z<=0.5): Acc = 0.7370 ± 0.1464 (avg count = 45.5)
            Critical (Z>0.5): Acc = 0.6572 ± 0.1193 (avg count = 36.6)
            """
            # generate_zscore_accuracy_plot(os.path.join("./visualization/figures", "memory_buffer_z_score"))
            # generate_zscore_accuracy_dodged_plot(os.path.join("./visualization/figures", "memory_buffer_z_score"))
            generate_zscore_accuracy_no_border_plot_1(os.path.join("./visualization/figures", "memory_buffer_z_score"))