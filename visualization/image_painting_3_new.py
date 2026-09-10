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
import matplotlib.ticker as ticker
import numpy as np
import os
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
    
    fig_dir = os.path.join('./visualization/figures', 'memory_buffer_time_stamp_interval')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f'memory_buffer_mean_time_stamp_interval_all_seed{seed}_dataset{args.data_name}.png')
    plt.savefig(fig_path)
    plt.close()



def setup_plot_style():
    """全局样式配置，确保与之前风格一致"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 22
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['svg.fonttype'] = 'none'  # 关键：Visio 中文字可编辑

def visualize_memory_time_stamp_interval_subs_methods_lines(args):
    # --- [前半部分数据处理逻辑保持不变] ---
    seed = args.SEED
    buffer_names = args.dataset_info['buffer_name']
    log_paths = args.dataset_info['log_path']
    sub_num = args.dataset_info['sub_num']
    mean_intervals_methods = []

    for buffer_name, result_path in zip(buffer_names, log_paths):
        mean_intervals_subs = []
        for idx in range(sub_num):
            # 假设 load_all_buffers 已在外部定义
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
        
        mean_intervals_avg = np.mean(mean_intervals_subs, axis=0)
        mean_intervals_methods.append(mean_intervals_avg)

    # ==========================================
    # 修改后的绘图部分 (纯折线、学术风格)
    # ==========================================
    setup_plot_style()
    
    fig = plt.figure(figsize=(10, 7))
    # 手动设置坐标轴位置，确保 Visio 导入时坐标稳定
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])

    # 定义不同算法的样式映射 (与之前 buffer 分析图一致)
    style_map = {
        'Ours': {'color': '#E31A1C', 'ls': '-',  'lw': 3.5, 'z': 5},
        'HUS':  {'color': '#1F78B4', 'ls': '--', 'lw': 2.5, 'z': 4},
        'CSTU': {'color': '#33A02C', 'ls': '-.', 'lw': 2.5, 'z': 3},
        'FIFO': {'color': '#6A3D9A', 'ls': ':',  'lw': 2.0, 'z': 2}
    }
    # 备选样式（如果 buffer_name 不在上述列表中）
    default_styles = [('-', '#377eb8'), ('--', '#e41a1c'), ('-.', '#4daf4a'), (':', '#984ea3')]

    for i, mean_intervals_avg in enumerate(mean_intervals_methods):
        name = str(buffer_names[i])
        # 获取样式配置
        s = style_map.get(name, {
            'color': default_styles[i % 4][1], 
            'ls': default_styles[i % 4][0], 
            'lw': 2.5, 'z': 1
        })
        
        # 绘制纯折线 (不带 marker)
        ax.plot(range(len(mean_intervals_avg)), mean_intervals_avg, 
                label=name, 
                color=s['color'], 
                linestyle=s['ls'], 
                linewidth=s['lw'], 
                zorder=s['z'])

    # 坐标轴标签与字号
    ax.set_xlabel('Number of Updates', labelpad=15, fontsize=24)
    ax.set_ylabel('Mean Time Interval', labelpad=15, fontsize=24)
    
    # 坐标轴范围与刻度设置 (根据你的数据动态调整，这里演示常用配置)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) # 保证横轴是整数
    # ax.set_ylim(0, 25) # 如果需要固定纵轴范围请取消注释
    
    # 图例配置
    legend = ax.legend(loc='upper left', frameon=True, edgecolor='black', 
                       fontsize=18, framealpha=1.0, facecolor='white')
    legend.set_zorder(10)

    # 刻度标签
    ax.tick_params(axis='both', labelsize=18)

    # 采用你指定的网格线风格
    ax.grid(True, axis='y', linestyle=':', alpha=0.8, color='#333333', linewidth=1.0, zorder=0)

    # 保存图片
    fig_dir = os.path.join('./visualization/figures', 'memory_buffer_time_stamp_interval')
    os.makedirs(fig_dir, exist_ok=True)
    
    file_base = f'memory_buffer_mean_interval_seed{seed}_{args.data_name}_line'
    
    # 同时保存 SVG 和 PNG
    plt.savefig(os.path.join(fig_dir, f'{file_base}.svg'), format='svg')
    plt.savefig(os.path.join(fig_dir, f'{file_base}.png'), format='png', dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"✅ 绘图已完成并保存至: {fig_dir}")



def visualize_memory_time_stamp_interval_subs_methods_bins(args):
    """
    自适应数据量：提取数据 -> 动态分桶均值 -> 绘制学术折线图
    """
    # --- [A. 数据提取逻辑: 保持您的原始实现] ---
    seed = args.SEED
    buffer_names = args.dataset_info['buffer_name']
    log_paths = args.dataset_info['log_path']
    sub_num = args.dataset_info['sub_num']
    mean_intervals_methods = []

    for buffer_name, result_path in zip(buffer_names, log_paths):
        mean_intervals_subs = []
        for idx in range(sub_num):
            # 注意：load_all_buffers 需在您的工程代码中已定义
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
        
        # 计算该方法在所有 subjects 上的平均值
        mean_intervals_avg = np.mean(mean_intervals_subs, axis=0)
        mean_intervals_methods.append(mean_intervals_avg)

    # ==========================================
    # B. 绘图部分 (自适应 X 轴范围)
    # ==========================================
    setup_plot_style()
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0.15, 0.18, 0.80, 0.77])

    bin_size = 5  # 分桶大小
    
    # 获取所有方法中最大的数据长度，用于确定 X 轴上限
    max_updates = max(len(data) for data in mean_intervals_methods) if mean_intervals_methods else 0
    
    # 统一样式映射
    style_map = {
        'Ours': {'color': '#E31A1C', 'ls': '-',  'marker': 'o', 'lw': 3.5, 'z': 5},
        'HUS':  {'color': '#1F78B4', 'ls': '--', 'marker': 's', 'lw': 2.5, 'z': 4},
        'CSTU': {'color': '#33A02C', 'ls': '-.', 'marker': '^', 'lw': 2.5, 'z': 3},
        'FIFO': {'color': '#6A3D9A', 'ls': ':',  'marker': 'D', 'lw': 2.0, 'z': 2}
    }

    for i, data_full in enumerate(mean_intervals_methods):
        name = str(buffer_names[i])
        
        # --- [关键：自适应分桶计算] ---
        binned_data = []
        x_coords = []
        
        # 使用 range 步进 bin_size 进行切片
        for j in range(0, len(data_full), bin_size):
            chunk = data_full[j : j + bin_size]
            binned_data.append(np.mean(chunk))
            # X 坐标对应为当前桶的“中心”或“末尾”，这里取末尾，但不超过实际长度
            pos = min(j + bin_size, len(data_full))
            x_coords.append(pos)
        
        s = style_map.get(name, {'color': 'gray', 'ls': '-', 'marker': 'v', 'lw': 2, 'z': 1})
        
        ax.plot(x_coords, binned_data, 
                label=name, 
                color=s['color'], 
                linestyle=s['ls'], 
                marker=s['marker'], 
                linewidth=s['lw'], 
                markersize=10, 
                zorder=s['z'])

    # --- [核心修改：动态设置横轴范围] ---
    # 根据实际的数据量 max_updates 自动设置 xlim
    # 留出 2% 的余量，防止最右侧的点被坐标轴压住
    ax.set_xlim(0, max_updates * 1.02) 
    
    # 动态确定刻度间隔：如果数据量很大，间隔就设为 20；如果较小，设为 5 或 10
    if max_updates > 150:
        major_space = 50
    elif max_updates > 50:
        major_space = 10
    else:
        major_space = 5
        
    ax.xaxis.set_major_locator(ticker.MultipleLocator(major_space))
    
    # 标签与字号
    ax.set_xlabel('Number of Updates', labelpad=15, fontsize=24)
    ax.set_ylabel('Mean Time Interval', labelpad=15, fontsize=24)
    ax.tick_params(axis='both', labelsize=20)

    # 图例配置
    legend = ax.legend(loc='upper left', frameon=True, edgecolor='black', 
                       fontsize=18, framealpha=1.0, facecolor='white')
    legend.set_zorder(10)

    # 网格线
    ax.grid(True, axis='y', linestyle=':', alpha=0.8, color='#333333', linewidth=1.0, zorder=0)

    # --- 保存输出 ---
    fig_dir = os.path.join('./visualization/figures', 'memory_buffer_time_stamp_interval')
    os.makedirs(fig_dir, exist_ok=True)
    
    file_base = f'dynamic_binned_interval_seed{seed}_{args.data_name}_bins'
    
    plt.savefig(os.path.join(fig_dir, f'{file_base}.svg'), format='svg')
    plt.savefig(os.path.join(fig_dir, f'{file_base}.png'), format='png', dpi=300)
    
    plt.close()
    print(f"✅ 自适应绘图已完成 (最大更新次数: {max_updates}): {file_base}.svg")


def visualize_memory_time_stamp_interval_subs_methods_bins_1(args):
    """
    自适应数据量：提取数据 -> 动态分桶均值 -> 绘制学术折线图 (彻底无网格版)
    """
    # --- [A. 数据提取逻辑: 保持您的原始实现] ---
    seed = args.SEED
    buffer_names = args.dataset_info['buffer_name']
    log_paths = args.dataset_info['log_path']
    sub_num = args.dataset_info['sub_num']
    mean_intervals_methods = []

    for buffer_name, result_path in zip(buffer_names, log_paths):
        mean_intervals_subs = []
        for idx in range(sub_num):
            # 注意：load_all_buffers 需在您的工程代码中已定义
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
        
        mean_intervals_avg = np.mean(mean_intervals_subs, axis=0)
        mean_intervals_methods.append(mean_intervals_avg)

    # ==========================================
    # B. 绘图部分 (强制关闭网格线)
    # ==========================================
    setup_plot_style()
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0.15, 0.18, 0.80, 0.77])

    # 【关键修改 1】显式强制关闭所有网格线
    ax.grid(False) 

    bin_size = 5  
    max_updates = max(len(data) for data in mean_intervals_methods) if mean_intervals_methods else 0
    
    style_map = {
        'Ours': {'color': '#E31A1C', 'ls': '-',  'marker': 'o', 'lw': 3.5, 'z': 5},
        'HUS':  {'color': '#1F78B4', 'ls': '--', 'marker': 's', 'lw': 2.5, 'z': 4},
        'CSTU': {'color': '#33A02C', 'ls': '-.', 'marker': '^', 'lw': 2.5, 'z': 3},
        'FIFO': {'color': '#6A3D9A', 'ls': ':',  'marker': 'D', 'lw': 2.0, 'z': 2}
    }

    for i, data_full in enumerate(mean_intervals_methods):
        name = str(buffer_names[i])
        
        binned_data = []
        x_coords = []
        for j in range(0, len(data_full), bin_size):
            chunk = data_full[j : j + bin_size]
            binned_data.append(np.mean(chunk))
            pos = min(j + bin_size, len(data_full))
            x_coords.append(pos)
        
        s = style_map.get(name, {'color': 'gray', 'ls': '-', 'marker': 'v', 'lw': 2, 'z': 1})
        
        ax.plot(x_coords, binned_data, 
                label=name, 
                color=s['color'], 
                linestyle=s['ls'], 
                marker=s['marker'], 
                linewidth=s['lw'], 
                markersize=10, 
                zorder=s['z'])

    # 坐标轴范围自适应
    if args.data_name not in ["BNCI2014_004-test"]: 
        ax.set_xlim(0, max_updates * 1.05) 
    else:
        ax.set_xlim(2.5, max_updates * 1.05)
    
    if max_updates > 150:
        major_space = 50
    elif max_updates > 50:
        major_space = 10
    else:
        major_space = 5
        
    ax.xaxis.set_major_locator(ticker.MultipleLocator(major_space))
    
    ax.set_xlabel('Number of Updates', labelpad=15, fontsize=24)
    ax.set_ylabel('Mean Time Interval', labelpad=15, fontsize=24)
    ax.tick_params(axis='both', labelsize=20)

    # 图例配置
    legend = ax.legend(loc='upper left', frameon=True, edgecolor='black', 
                       fontsize=18, framealpha=1.0, facecolor='white')
    legend.set_zorder(10)

    # 【关键修改 2】再次确保网格线不会出现在任何轴上
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    # --- 保存输出 ---
    fig_dir = os.path.join('./visualization/figures', 'memory_buffer_time_stamp_interval')
    os.makedirs(fig_dir, exist_ok=True)
    
    file_base = f'dynamic_binned_interval_seed{seed}_{args.data_name}_bins'
    
    # 强制先保存 SVG 再保存 PNG，确保渲染一致性
    plt.savefig(os.path.join(fig_dir, f'{file_base}.svg'), format='svg')
    plt.savefig(os.path.join(fig_dir, f'{file_base}.png'), format='png', dpi=300)
    
    plt.close(fig)
    print(f"✅ 严格无网格图已生成: {file_base}.png")


def visualize_memory_time_stamp_interval_subs_methods_bins_2(args):
    """
    自适应数据量：分桶均值 -> 封闭黑框 -> 刻度向外 (无网格版，强制整数刻度)
    """
    # --- [A. 数据提取逻辑: 保持不变] ---
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
        
        mean_intervals_avg = np.mean(mean_intervals_subs, axis=0)
        mean_intervals_methods.append(mean_intervals_avg)

    # ==========================================
    # B. 绘图部分 (封闭框 + 刻度向外 + 整数刻度)
    # ==========================================
    setup_plot_style()
    
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_axes([0.15, 0.18, 0.80, 0.77])
    
    # 1. 设置封闭黑框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0) # 对应参考图的线条粗细

    # 2. 设置刻度属性 (向外，仅左下)
    ax.tick_params(axis='both', which='major', direction='out', 
                   bottom=True, top=False, left=True, right=False, 
                   length=6, width=1.2, labelsize=20)

    # 3. 彻底关闭网格
    ax.grid(False)

    bin_size = 5  
    max_updates = max(len(data) for data in mean_intervals_methods) if mean_intervals_methods else 0
    
    style_map = {
        'Ours': {'color': '#e41a1c', 'ls': '--', 'marker': 's', 'lw': 2.5, 'z': 5}, 
        'HUS':  {'color': '#377eb8', 'ls': '-',  'marker': 'o', 'lw': 2.5, 'z': 4}, 
        'CSTU': {'color': '#4daf4a', 'ls': '-.', 'marker': '^', 'lw': 2.5, 'z': 3}, 
        'FIFO': {'color': '#984ea3', 'ls': ':',  'marker': 'D', 'lw': 2.0, 'z': 2}  
    }

    for i, data_full in enumerate(mean_intervals_methods):
        name = str(buffer_names[i])
        binned_data = []
        x_coords = []
        for j in range(0, len(data_full), bin_size):
            chunk = data_full[j : j + bin_size]
            binned_data.append(np.mean(chunk))
            pos = min(j + bin_size, len(data_full))
            x_coords.append(pos)
        
        s = style_map.get(name, {'color': 'gray', 'ls': '-', 'marker': 'v', 'lw': 2, 'z': 1})
        ax.plot(x_coords, binned_data, label=name, color=s['color'], linestyle=s['ls'], 
                marker=s['marker'], linewidth=s['lw'], markersize=10, zorder=s['z'])

    # --- 【核心修改：强制整数刻度】 ---
    # 横轴设置
    if args.data_name not in ["BNCI2014_004-test"]: 
        ax.set_xlim(0, max_updates * 1.05)    
    else:
        ax.set_xlim(2.5, max_updates * 1.05) 
    
    major_space = 10 if max_updates > 50 else 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(major_space))
    # 强制横轴显示为整数
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    # 纵轴设置 (强制显示为整数)
    if args.data_name not in ["BNCI2014_004-test"]: 
        y_interval = 5 # 如果数据范围很小（比如只有0-10），可以改成 2
    else:
        y_interval = 1 # 根据实际数据范围调整
    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_interval))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    # 4. 标签与图例
    ax.set_xlabel('Number of Updates', labelpad=20, fontsize=24)
    ax.set_ylabel('Mean Time Interval', labelpad=20, fontsize=24)

    legend = ax.legend(loc='upper left', frameon=True, edgecolor='black', 
                       fontsize=18, framealpha=1.0, facecolor='white')
    legend.set_zorder(10)

    # 保存输出
    fig_dir = os.path.join('./visualization/figures', 'memory_buffer_time_stamp_interval_new')
    os.makedirs(fig_dir, exist_ok=True)
    file_base = f'strictly_integer_outward_ticks_seed{seed}_{args.data_name}'
    
    plt.savefig(os.path.join(fig_dir, f'{file_base}.svg'), format='svg')
    plt.savefig(os.path.join(fig_dir, f'{file_base}.png'), format='png', dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"✅ 绘图完成（封闭框+向外整数刻度）: {file_base}.svg")

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
            ["/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-001-test-e300-b64-debugs/ours_debug_m_cls_process_1-BNCI2014001-4-all-EEGNet-4,2-e300-b64-new/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-2-visualization/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p1",
            "/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-001-test-e300-b64-debugs/ours_debug_m_cls_process_1-BNCI2014001-4-all-EEGNet-4,2-e300-b64-new/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation4/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visulization-1",
            "/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-001-test-e300-b64-debugs/ours_debug_m_cls_process_1-BNCI2014001-4-all-EEGNet-4,2-e300-b64-new/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation5/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visualization1",
            "/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-001-test-e300-b64-debugs/ours_debug_m_cls_process_1-BNCI2014001-4-all-EEGNet-4,2-e300-b64-new/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation6/proposed_57_BNoff_batch8stride8_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visualization1",],
            "sub_num": 9,
        }
        dataset_004 = {
            "buffer_name":
            ["Ours", "HUS", "FIFO", "CSTU"],
            "log_path": 
            ["/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-004-test-e300-b64/proposed_new/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-visualization/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p6",
            "/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-004-test-e300-b64/proposed_new/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation4/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visulization1",
            "/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-004-test-e300-b64/proposed_new/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation5/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visualization1",
            "/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-004-test-e300-b64/proposed_new/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation6/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visualization1",],
            "sub_num": 9,
        }
            
        dataset_SHU = {
            "buffer_name":
            ["Ours", "HUS", "FIFO", "CSTU"],
            "log_path": 
            ["/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-WBCIC-SHU-3C-e300-b64/proposed-new/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-visualization/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p1",
            "/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-WBCIC-SHU-3C-e300-b64/proposed-new/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation4/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visualization1",
            "/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-WBCIC-SHU-3C-e300-b64/proposed-new/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation5/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visualization1",
            "/data/datasets_Jyt/DeepTransferBCI/logs_backup/Baselines-WBCIC-SHU-3C-e300-b64/proposed-new/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-ablation6/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p-visualization1",],
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
        for s in [5]:
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
            
            visualize_memory_time_stamp_interval_subs_methods_bins_2(args)