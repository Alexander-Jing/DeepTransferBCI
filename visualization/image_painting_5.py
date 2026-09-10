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

import matplotlib.ticker as ticker


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




# ==========================================
# 1. 全局样式配置 (针对 Visio 编辑优化)
# ==========================================
def setup_plot_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 20
    plt.rcParams['mathtext.fontset'] = 'stix'
    # 核心：导出为文字对象，方便在 Visio 中双击编辑
    plt.rcParams['svg.fonttype'] = 'none'

def grad_visual(result_path, seed, idx, capacity=64):
    """
    梯度分析可视化：纯折线图 (无点) -> 封闭黑框 -> 仅横向网格
    """
    # --- [A. 数据加载逻辑: 保持不变] ---
    dir_path = os.path.join(result_path, 'gradients', f'seed{seed}_sub_idx{idx}')
    file_list = glob.glob(os.path.join(dir_path, 'memory_buffer_instance_*.pt'))
    if not file_list:
        raise FileNotFoundError(f"No memory_buffer_instance_*.pt files found in {dir_path}")

    def extract_instance_num(filename):
        match = re.search(r'memory_buffer_instance_(\d+)\.pt', filename)
        return int(match.group(1)) if match else -1

    file_list = sorted(file_list, key=extract_instance_num)

    grads_1_norms = []
    grads_2_norms = []
    cosine_similarities = []

    for file_path in file_list:
        data = torch.load(file_path, map_location='cpu')
        grads_1_norms.append(data['norm_1'])
        grads_2_norms.append(data['norm_2'])
        cosine_similarities.append(data['cosine_similarity'])

    # --- [B. 绘图配置] ---
    setup_plot_style()
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0.12, 0.18, 0.83, 0.77]) 

    # 使用 1, 2, 3... 顺序编号作为横轴
    x_range = np.arange(1, len(grads_1_norms) + 1)

    # --- 1. 绘制纯折线 (不带 Marker，通过线型区分) ---
    # Grads 1 Norm: 蓝色 实线
    ax.plot(x_range, grads_1_norms, label='Grads 1 Norm', 
            color='#377eb8', linestyle='-', linewidth=2.5, zorder=3)
    
    # Grads 2 Norm: 红色 虚线
    ax.plot(x_range, grads_2_norms, label='Grads 2 Norm', 
            color='#e41a1c', linestyle='--', linewidth=2.5, zorder=3)
    
    # Cosine Similarity: 绿色 点划线
    ax.plot(x_range, cosine_similarities, label='Cosine Similarity', 
            color='#4daf4a', linestyle='-.', linewidth=2.5, zorder=3)

    # --- 2. 边框与刻度配置 ---
    # 确保四面边框都有黑线
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)

    # 设置刻度向外，且仅在左、下显示刻度线
    ax.tick_params(axis='both', which='major', direction='out', 
                   bottom=True, top=False, left=True, right=False, 
                   length=6, width=1.2, labelsize=18)

    # 强制刻度显示为整数
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # --- 3. 标签与网格 ---
    ax.set_xlabel('Update Steps (Sequential Index)', labelpad=15, fontsize=22)
    ax.set_ylabel('Metric Value', labelpad=15, fontsize=22)

    # 【关键要求】仅显示横向网格线 (与之前风格一致)
    ax.grid(True, axis='y', linestyle=':', alpha=0.8, color='#333333', linewidth=1.0, zorder=0)

    # --- 4. 图例配置 ---
    legend = ax.legend(loc='upper right', frameon=True, edgecolor='black', 
                       fontsize=16, framealpha=1.0, facecolor='white')
    legend.set_zorder(10)

    # --- 5. 保存输出 ---
    fig_dir = os.path.join(result_path, 'grad_figures_1')
    os.makedirs(fig_dir, exist_ok=True)
    
    file_base = f'grad_analysis_pure_line_seed{seed}_sub{idx}'
    # 保存 SVG (Visio 专用)
    plt.savefig(os.path.join(fig_dir, f'{file_base}.svg'), format='svg')
    # 保存 PNG (预览专用)
    plt.savefig(os.path.join(fig_dir, f'{file_base}.png'), format='png', dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"✅ 梯度分析纯折线图已生成: {file_base}.svg")

def grad_visual_1(result_path, seed, idx, capacity=64):
    """
    梯度分析可视化：纯折线图 (无点) -> 封闭黑框 -> 仅横向网格
    """
    # --- [A. 数据加载逻辑: 保持不变] ---
    dir_path = os.path.join(result_path, 'gradients', f'seed{seed}_sub_idx{idx}')
    file_list = glob.glob(os.path.join(dir_path, 'memory_buffer_instance_*.pt'))
    if not file_list:
        raise FileNotFoundError(f"No memory_buffer_instance_*.pt files found in {dir_path}")

    def extract_instance_num(filename):
        match = re.search(r'memory_buffer_instance_(\d+)\.pt', filename)
        return int(match.group(1)) if match else -1

    file_list = sorted(file_list, key=extract_instance_num)

    grads_1_norms = []
    grads_2_norms = []
    cosine_similarities = []

    for file_path in file_list:
        data = torch.load(file_path, map_location='cpu')
        grads_1_norms.append(data['norm_1'])
        grads_2_norms.append(data['norm_2'])
        cosine_similarities.append(data['cosine_similarity'])

    # 对 grads_1_norms 做滑动平均平滑处理 (window=5，减少陡峭波动)
    smooth_window = 5
    grads_1_norms_smooth = pd.Series(grads_1_norms).rolling(
        window=smooth_window, center=True, min_periods=1).mean().values

    # --- [B. 绘图配置] ---
    setup_plot_style()
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0.12, 0.18, 0.83, 0.77]) 

    # 使用 1, 2, 3... 顺序编号作为横轴
    x_range = np.arange(1, len(grads_1_norms) + 1)

    # --- 1. 绘制纯折线 (不带 Marker，通过线型区分) ---
    # Grads 1 Norm: 蓝色 实线 (平滑后)
    ax.plot(x_range, grads_1_norms_smooth, label='Grads 1 Norm', 
            color='#377eb8', linestyle='-', linewidth=2.5, zorder=3)
    
    # Grads 2 Norm: 红色 虚线
    ax.plot(x_range, grads_2_norms, label='Grads 2 Norm', 
            color='#e41a1c', linestyle='--', linewidth=2.5, zorder=3)
    
    # Cosine Similarity: 绿色 点划线
    ax.plot(x_range, cosine_similarities, label='Cosine Similarity', 
            color='#4daf4a', linestyle='-.', linewidth=2.5, zorder=3)

    # --- 2. 边框与刻度配置 ---
    # 确保四面边框都有黑线
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)

    # 设置刻度向外，且仅在左、下显示刻度线
    ax.tick_params(axis='both', which='major', direction='out', 
                   bottom=True, top=False, left=True, right=False, 
                   length=6, width=1.2, labelsize=18)

    # 强制刻度显示为整数
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # --- 3. 标签与网格 ---
    ax.set_xlabel('Update Steps (Sequential Index)', labelpad=15, fontsize=22)
    ax.set_ylabel('Metric Value', labelpad=15, fontsize=22)

    # 仅显示横向网格线，显式关闭纵向网格线
    ax.grid(False, axis='x')
    ax.grid(True, axis='y', linestyle=':', alpha=0.8, color='#333333', linewidth=1.0, zorder=0)

    # 设置 y 轴范围与刻度间隔
    ax.set_ylim(-0.40, 3.75)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    # --- 4. 图例配置 (轴内左上角，避免遮挡右侧数据) ---
    legend = ax.legend(loc='upper left', frameon=True, edgecolor='black', 
                       fontsize=16, framealpha=1.0, facecolor='white')
    legend.set_zorder(10)

    # --- 5. 保存输出 ---
    fig_dir = os.path.join(result_path, 'grad_figures_1')
    os.makedirs(fig_dir, exist_ok=True)
    
    file_base = f'grad_analysis_pure_line_seed{seed}_sub{idx}'
    # 保存 SVG (Visio 专用)
    plt.savefig(os.path.join(fig_dir, f'{file_base}.svg'), format='svg')
    # 保存 PNG (预览专用)
    plt.savefig(os.path.join(fig_dir, f'{file_base}.png'), format='png', dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"✅ 梯度分析纯折线图已生成: {file_base}.svg")


def grad_visual_broken_axis(result_path, seed, idx, capacity=64):
    """
    自适应量级断轴图：解决 Norm 1 和 Norm 2 量级悬殊
    封闭黑框风格 + 刻度向外 + 顺序编号 -> 一张图内显示
    """
    # --- [A. 数据加载逻辑: 保持不变] ---
    dir_path = os.path.join(result_path, 'gradients', f'seed{seed}_sub_idx{idx}')
    file_list = glob.glob(os.path.join(dir_path, 'memory_buffer_instance_*.pt'))
    if not file_list:
        raise FileNotFoundError(f"No memory_buffer_instance_*.pt files found in {dir_path}")

    def extract_instance_num(filename):
        match = re.search(r'memory_buffer_instance_(\d+)\.pt', filename)
        return int(match.group(1)) if match else -1

    file_list = sorted(file_list, key=extract_instance_num)

    grads_1_norms = []
    grads_2_norms = []
    cosine_similarities = []

    for file_path in file_list:
        data = torch.load(file_path, map_location='cpu')
        grads_1_norms.append(data['norm_1'])
        grads_2_norms.append(data['norm_2'])
        cosine_similarities.append(data['cosine_similarity'])

    # 平滑 Ours 方法的 Grads 1
    smooth_window = 5
    grads_1_series = pd.Series(grads_1_norms)
    # min_periods=1 保证最右侧数据不会变少
    grads_1_smooth = grads_1_series.rolling(window=smooth_window, center=True, min_periods=1).mean().values

    # --- [B. 断轴逻辑配置] ---
    setup_plot_style()
    
    # 手动调整边距，预留 Y 轴标签空间
    fig = plt.figure(figsize=(11, 7))
    fig.subplots_adjust(left=0.12, bottom=0.18, right=0.95, top=0.9)

    # 【关键】使用两个 Axes 组合成一个图
    # [left, bottom, width, height]
    ax_upper = fig.add_axes([0.15, 0.58, 0.80, 0.37]) # 上半部分，显示大数值
    ax_lower = fig.add_axes([0.15, 0.18, 0.80, 0.37]) # 下半部分，显示小数值

    x_range = np.arange(1, len(grads_1_norms) + 1)

    # --- 1. 绘制纯折线 (在两个 ax 上都画，通过 ylim 截断) ---
    line_styles = [
        {'name': 'Grads 1 Norm', 'data': grads_1_smooth, 'color': '#377eb8', 'ls': '-'},
        {'name': 'Grads 2 Norm', 'data': grads_2_norms, 'color': '#e41a1c', 'ls': '--'},
        {'name': 'Cosine Similarity', 'data': cosine_similarities, 'color': '#4daf4a', 'ls': '-.'}
    ]

    for ax in [ax_upper, ax_lower]:
        ax.plot(x_range, line_styles[0]['data'], label=line_styles[0]['name'], 
                color=line_styles[0]['color'], linestyle=line_styles[0]['ls'], linewidth=2.5, zorder=3)
        ax.plot(x_range, line_styles[1]['data'], label=line_styles[1]['name'], 
                color=line_styles[1]['color'], linestyle=line_styles[1]['ls'], linewidth=2.5, zorder=3)
        ax.plot(x_range, line_styles[2]['data'], label=line_styles[2]['name'], 
                color=line_styles[2]['color'], linestyle=line_styles[2]['ls'], linewidth=2.5, zorder=3)

    # --- 2. 【核心】设置 Y 轴范围和断裂逻辑 ---
    # 根据你的数据范围 (Norm 1 ~3, Norm 2 ~0.5) 自动设置
    # 上半显示 2.0-3.75
    ax_upper.set_ylim(2.0, 3.75)
    # 下半显示 -0.1-1.0
    ax_lower.set_ylim(-0.1, 1.0) 

    # --- 3. 封闭黑框风格与刻度向外 (断轴专用布局) ---
    # 封闭顶部和右侧
    ax_upper.spines['bottom'].set_visible(False) # 禁用上部 ax 的底部边框
    ax_upper.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # 禁用上部 ax 的 X轴刻度和标签

    # 封闭底部和左侧
    ax_lower.spines['top'].set_visible(False) # 禁用下部 ax 的顶部边框

    # 封闭黑框和线条粗细 (L型刻度向外保持)
    for ax in [ax_upper, ax_lower]:
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)
        
        # 刻度向外且仅显示左下显示
        ax.tick_params(axis='y', which='major', direction='out', 
                       bottom=False, top=False, left=True, right=False, 
                       length=6, width=1.2, labelsize=18)
    
    # 底部 ax 开启 X 轴
    ax_lower.tick_params(axis='x', which='major', direction='out', 
                       bottom=True, top=False, length=6, width=1.2, labelsize=18)
    # 强制 X 轴为整数
    ax_lower.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax_lower.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # --- 4. 标签与网格 ---
    # X轴标签贴在底部 ax
    ax_lower.set_xlabel('Update Steps (Sequential Index)', labelpad=15, fontsize=22)
    # Y轴标签贴在两个 ax 之间，通过 text 手动定位
    fig.text(0.02, 0.55, 'Metric Value', fontsize=22, rotation='vertical', va='center')

    # 【关键要求】仅显示横向网格线，分别绘制
    ax_upper.grid(True, axis='y', linestyle=':', alpha=0.8, color='#333333', linewidth=1.0, zorder=0)
    ax_lower.grid(True, axis='y', linestyle=':', alpha=0.8, color='#333333', linewidth=1.0, zorder=0)

    # 纵轴刻度间隔设置
    ax_upper.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax_lower.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    # --- 5. 合并图例 (轴内左上角) ---
    handles, labels = ax_lower.get_legend_handles_labels()
    # 代理图例对象，确保样式一致
    legend = ax_lower.legend(handles, labels, loc='upper left', frameon=True, edgecolor='black', 
                        fontsize=16, framealpha=1.0, facecolor='white')
    legend.set_zorder(10)

    # --- 6. 【核心】绘制断裂符号 (Broken Marker) ---
    d = .015 # 符号的大小，可以微调
    # 在上部 ax 底部绘制
    # left top side
    kwargs = dict(transform=ax_upper.transAxes, color='black', clip_on=False, linewidth=1.0)
    ax_upper.plot((-d, +d), (-d, +d), **kwargs) # bottom-left diagonal

    # 在下部 ax 顶部绘制
    kwargs.update(transform=ax_lower.transAxes) 
    ax_lower.plot((-d, +d), (1 - d, 1 + d), **kwargs) # top-left diagonal
    #ax_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs) # top-right diagonal (如果顶部需要封闭可以用这个)

    # --- 7. 保存保存 ---
    fig_dir = os.path.join(result_path, 'grad_figures_broken')
    os.makedirs(fig_dir, exist_ok=True)
    file_base = f'grad_broken_axis_no_grid_seed{seed}_sub{idx}'
    # 保存 SVG (用于 Visio 高质量拼接)
    plt.savefig(os.path.join(fig_dir, f'{file_base}.svg'), format='svg')
    # 保存 PNG (预览专用)
    plt.savefig(os.path.join(fig_dir, f'{file_base}.png'), format='png', dpi=300)
    
    plt.close()
    print(f"✅ 断轴图已生成: {file_base}.svg")


def grad_visual_2(result_path, seed, idx, capacity=64):
    """
    梯度分析可视化：纯折线图 (无点) -> 封闭黑框 -> 仅横向网格
    """
    # --- [A. 数据加载逻辑: 保持不变] ---
    dir_path = os.path.join(result_path, 'gradients', f'seed{seed}_sub_idx{idx}')
    file_list = glob.glob(os.path.join(dir_path, 'memory_buffer_instance_*.pt'))
    if not file_list:
        raise FileNotFoundError(f"No memory_buffer_instance_*.pt files found in {dir_path}")

    def extract_instance_num(filename):
        match = re.search(r'memory_buffer_instance_(\d+)\.pt', filename)
        return int(match.group(1)) if match else -1

    file_list = sorted(file_list, key=extract_instance_num)

    grads_1_norms = []
    grads_2_norms = []
    cosine_similarities = []

    for file_path in file_list:
        data = torch.load(file_path, map_location='cpu')
        grads_1_norms.append(data['norm_1'])
        grads_2_norms.append(data['norm_2'])
        cosine_similarities.append(data['cosine_similarity'])

    # 对 grads_1_norms 做滑动平均平滑处理 (window=5，减少陡峭波动)
    smooth_window = 5
    grads_1_norms_smooth = pd.Series(grads_1_norms).rolling(
        window=smooth_window, center=True, min_periods=1).mean().values

    # --- [B. 绘图配置] ---
    setup_plot_style()
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0.12, 0.18, 0.83, 0.77]) 

    # 使用 1, 2, 3... 顺序编号作为横轴
    x_range = np.arange(1, len(grads_1_norms) + 1)

    # --- 1. 绘制纯折线 (不带 Marker，通过线型区分) ---
    # Grads 1 Norm: 蓝色 实线 (平滑后)
    ax.plot(x_range, grads_1_norms_smooth, label='Grads 1 Norm', 
            color='#377eb8', linestyle='-', linewidth=2.5, zorder=3)
    
    # Grads 2 Norm: 红色 虚线
    ax.plot(x_range, grads_2_norms, label='Grads 2 Norm', 
            color='#e41a1c', linestyle='--', linewidth=2.5, zorder=3)
    
    # Cosine Similarity: 绿色 点划线
    ax.plot(x_range, cosine_similarities, label='Cosine Similarity', 
            color='#4daf4a', linestyle='-.', linewidth=2.5, zorder=3)

    # --- 2. 边框与刻度配置 ---
    # 确保四面边框都有黑线
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)

    # 设置刻度向外，且仅在左、下显示刻度线
    ax.tick_params(axis='both', which='major', direction='out', 
                   bottom=True, top=False, left=True, right=False, 
                   length=6, width=1.2, labelsize=18)

    # 强制刻度显示为整数，间隔设置为 10
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    # --- 3. 标签与网格 ---
    ax.set_xlabel('Update Steps (Sequential Index)', labelpad=15, fontsize=22)
    ax.set_ylabel('Metric Value', labelpad=15, fontsize=22)

    # 仅显示横向网格线，显式关闭纵向网格线
    ax.grid(False, axis='x')
    ax.grid(True, axis='y', linestyle=':', alpha=0.8, color='#333333', linewidth=1.0, zorder=0)

    # 设置 y 轴范围与刻度间隔
    ax.set_ylim(-0.40, 3.75)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    # --- 4. 图例配置 (轴内左上角，避免遮挡右侧数据) ---
    legend = ax.legend(loc='upper left', frameon=True, edgecolor='black', 
                       fontsize=16, framealpha=1.0, facecolor='white')
    legend.set_zorder(10)

    # --- 5. 保存输出 ---
    fig_dir = os.path.join(result_path, 'grad_figures_1')
    os.makedirs(fig_dir, exist_ok=True)
    
    file_base = f'grad_analysis_pure_line_seed{seed}_sub{idx}'
    # 保存 SVG (Visio 专用)
    plt.savefig(os.path.join(fig_dir, f'{file_base}.svg'), format='svg')
    # 保存 PNG (预览专用)
    plt.savefig(os.path.join(fig_dir, f'{file_base}.png'), format='png', dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"✅ 梯度分析纯折线图已生成: {file_base}.svg")


def grad_visual_3(result_path, seed, idx, capacity=64):
    """
    梯度分析可视化：纯折线图 (无点) -> 封闭黑框 -> 仅横向网格
    """
    # --- [A. 数据加载逻辑: 保持不变] ---
    dir_path = os.path.join(result_path, 'gradients', f'seed{seed}_sub_idx{idx}')
    file_list = glob.glob(os.path.join(dir_path, 'memory_buffer_instance_*.pt'))
    if not file_list:
        raise FileNotFoundError(f"No memory_buffer_instance_*.pt files found in {dir_path}")

    def extract_instance_num(filename):
        match = re.search(r'memory_buffer_instance_(\d+)\.pt', filename)
        return int(match.group(1)) if match else -1

    file_list = sorted(file_list, key=extract_instance_num)

    grads_1_norms = []
    grads_2_norms = []
    cosine_similarities = []

    for file_path in file_list:
        data = torch.load(file_path, map_location='cpu')
        grads_1_norms.append(data['norm_1'])
        grads_2_norms.append(data['norm_2'])
        cosine_similarities.append(data['cosine_similarity'])

    # 对 grads_1_norms 做滑动平均平滑处理 (window=5，减少陡峭波动)
    smooth_window = 5
    grads_1_norms_smooth = pd.Series(grads_1_norms).rolling(
        window=smooth_window, center=True, min_periods=1).mean().values

    # --- [B. 绘图配置] ---
    setup_plot_style()
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0.12, 0.18, 0.83, 0.77]) 

    # 使用 1, 2, 3... 顺序编号作为横轴
    x_range = np.arange(1, len(grads_1_norms) + 1)

    # --- 1. 绘制纯折线 (不带 Marker，通过线型区分) ---
    # Grads 1 Norm: 蓝色 实线 (平滑后)
    ax.plot(x_range, grads_1_norms_smooth, label='Grads 1 Norm', 
            color='#377eb8', linestyle='-', linewidth=2.5, zorder=3)
    
    # Grads 2 Norm: 红色 虚线
    ax.plot(x_range, grads_2_norms, label='Grads 2 Norm', 
            color='#e41a1c', linestyle='--', linewidth=2.5, zorder=3)
    
    # Cosine Similarity: 绿色 点划线
    ax.plot(x_range, cosine_similarities, label='Cosine Similarity', 
            color='#4daf4a', linestyle='-.', linewidth=2.5, zorder=3)

    # --- 2. 边框与刻度配置 ---
    # 确保四面边框都有黑线
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)

    # 设置刻度向外，且仅在左、下显示刻度线
    ax.tick_params(axis='both', which='major', direction='out', 
                   bottom=True, top=False, left=True, right=False, 
                   length=6, width=1.2, labelsize=18)

    # 强制刻度显示为整数，间隔设置为 10
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    # --- 3. 标签与网格 ---
    ax.set_xlabel('Update Steps (Sequential Index)', labelpad=15, fontsize=22)
    ax.set_ylabel('Metric Value', labelpad=15, fontsize=22)

    # 仅显示横向网格线，显式关闭纵向网格线
    ax.grid(False, axis='x')
    ax.grid(True, axis='y', linestyle=':', alpha=0.8, color='#333333', linewidth=1.0, zorder=0)

    # 设置 y 轴范围与刻度间隔
    ax.set_ylim(-0.40, 3.75)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    # --- 4. 图例配置 (轴内左上角，避免遮挡右侧数据) ---
    legend = ax.legend(loc='upper left', frameon=True, edgecolor='black', 
                       fontsize=16, framealpha=1.0, facecolor='white')
    legend.set_zorder(10)

    # --- 5. 保存输出 ---
    fig_dir = os.path.join(result_path, 'grad_figures_1')
    os.makedirs(fig_dir, exist_ok=True)
    
    file_base = f'grad_analysis_pure_line_seed{seed}_sub{idx}'
    # 保存 SVG (Visio 专用)
    plt.savefig(os.path.join(fig_dir, f'{file_base}.svg'), format='svg')
    # 保存 PNG (预览专用)
    plt.savefig(os.path.join(fig_dir, f'{file_base}.png'), format='png', dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"✅ 梯度分析纯折线图已生成: {file_base}.svg")


def grad_visual_avg(result_path, seed, sub_num, capacity=64):
    """
    多被试梯度均值可视化：对所有被试在相同 instance_num 上计算均值±标准差，
    均值绘折线，标准差绘阴影。Grads 1 Norm 均值额外做滑动平均平滑。
    """
    def extract_instance_num(filename):
        match = re.search(r'memory_buffer_instance_(\d+)\.pt', filename)
        return int(match.group(1)) if match else -1

    # --- [A. 多被试数据加载，按 instance_num 聚合] ---
    all_grads_1 = {}   # {instance_num: [val_sub0, val_sub1, ...]}
    all_grads_2 = {}
    all_cosines = {}

    for idx in range(sub_num):
        dir_path = os.path.join(result_path, 'gradients', f'seed{seed}_sub_idx{idx}')
        file_list = glob.glob(os.path.join(dir_path, 'memory_buffer_instance_*.pt'))
        if not file_list:
            print(f"⚠️  跳过 sub_idx{idx}：未找到梯度文件")
            continue
        for file_path in sorted(file_list, key=extract_instance_num):
            inst = extract_instance_num(file_path)
            data = torch.load(file_path, map_location='cpu')
            all_grads_1.setdefault(inst, []).append(data['norm_1'])
            all_grads_2.setdefault(inst, []).append(data['norm_2'])
            all_cosines.setdefault(inst, []).append(data['cosine_similarity'])

    if not all_grads_1:
        raise FileNotFoundError(f"No gradient data found under {result_path}/gradients/")

    # --- [B. 计算均值和标准差] ---
    sorted_keys = sorted(all_grads_1.keys())
    x_range = np.arange(1, len(sorted_keys) + 1)

    mean_1   = np.array([np.mean(all_grads_1[k]) for k in sorted_keys])
    std_1    = np.array([np.std(all_grads_1[k])  for k in sorted_keys])
    mean_2   = np.array([np.mean(all_grads_2[k]) for k in sorted_keys])
    std_2    = np.array([np.std(all_grads_2[k])  for k in sorted_keys])
    mean_cos = np.array([np.mean(all_cosines[k]) for k in sorted_keys])
    std_cos  = np.array([np.std(all_cosines[k])  for k in sorted_keys])

    # 对 Grads 1 均值做滑动平均平滑
    smooth_window = 1
    mean_1_smooth = pd.Series(mean_1).rolling(
        window=smooth_window, center=True, min_periods=1).mean().values

    # --- [C. 绘图配置] ---
    setup_plot_style()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0.12, 0.18, 0.83, 0.77])

    # --- 1. 绘制均值折线 + 标准差阴影 ---
    # Grads 1 Norm: 蓝色 实线 (平滑均值) + 阴影
    ax.plot(x_range, mean_1_smooth, label='Grads 1 Norm',
            color='#377eb8', linestyle='-', linewidth=2.5, zorder=3)
    ax.fill_between(x_range, mean_1_smooth - std_1, mean_1_smooth + std_1,
                    color='#377eb8', alpha=0.18, zorder=2)

    # Grads 2 Norm: 红色 虚线 + 阴影
    ax.plot(x_range, mean_2, label='Grads 2 Norm',
            color='#e41a1c', linestyle='--', linewidth=2.5, zorder=3)
    ax.fill_between(x_range, mean_2 - std_2, mean_2 + std_2,
                    color='#e41a1c', alpha=0.18, zorder=2)

    # Cosine Similarity: 绿色 点划线 + 阴影
    ax.plot(x_range, mean_cos, label='Cosine Similarity',
            color='#4daf4a', linestyle='-.', linewidth=2.5, zorder=3)
    ax.fill_between(x_range, mean_cos - std_cos, mean_cos + std_cos,
                    color='#4daf4a', alpha=0.18, zorder=2)

    # --- 2. 边框与刻度配置 ---
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)

    ax.tick_params(axis='both', which='major', direction='out',
                   bottom=True, top=False, left=True, right=False,
                   length=6, width=1.2, labelsize=18)

    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    # --- 3. 标签与网格 ---
    ax.set_xlabel('Update Steps (Sequential Index)', labelpad=15, fontsize=22)
    ax.set_ylabel('Metric Value', labelpad=15, fontsize=22)

    ax.grid(False, axis='x')
    ax.grid(True, axis='y', linestyle=':', alpha=0.8, color='#333333', linewidth=1.0, zorder=0)

    ax.set_ylim(-0.40, 4.8)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    # --- 4. 图例配置 ---
    legend = ax.legend(loc='upper left', frameon=True, edgecolor='black',
                       fontsize=16, framealpha=1.0, facecolor='white')
    legend.set_zorder(10)

    # --- 5. 保存输出 ---
    fig_dir = os.path.join(result_path, 'grad_figures_avg')
    os.makedirs(fig_dir, exist_ok=True)

    file_base = f'grad_avg_seed{seed}_allsubs'
    plt.savefig(os.path.join(fig_dir, f'{file_base}.svg'), format='svg')
    plt.savefig(os.path.join(fig_dir, f'{file_base}.png'), format='png', dpi=300, bbox_inches='tight')

    plt.close()
    print(f"✅ 多被试均值梯度图已生成: {file_base}.svg")

def grad_visual_avg_stacked(result_path, seed, sub_num, capacity=64):
    """
    多被试梯度均值可视化（堆叠版）：
    (a) Norm 1 & Norm 2 (均值±标准差阴影)
    (b) Cosine Similarity (均值±标准差阴影)
    """
    def extract_instance_num(filename):
        match = re.search(r'memory_buffer_instance_(\d+)\.pt', filename)
        return int(match.group(1)) if match else -1

    # --- [A. 数据聚合逻辑] ---
    all_grads_1 = {}
    all_grads_2 = {}
    all_cosines = {}

    for idx in range(sub_num):
        dir_path = os.path.join(result_path, 'gradients', f'seed{seed}_sub_idx{idx}')
        file_list = glob.glob(os.path.join(dir_path, 'memory_buffer_instance_*.pt'))
        if not file_list:
            continue
        for file_path in sorted(file_list, key=extract_instance_num):
            inst = extract_instance_num(file_path)
            data = torch.load(file_path, map_location='cpu')
            all_grads_1.setdefault(inst, []).append(data['norm_1'])
            all_grads_2.setdefault(inst, []).append(data['norm_2'])
            all_cosines.setdefault(inst, []).append(data['cosine_similarity'])

    if not all_grads_1:
        raise FileNotFoundError("未找到梯度数据")

    # --- [B. 计算均值和标准差] ---
    sorted_keys = sorted(all_grads_1.keys())
    x_range = np.arange(1, len(sorted_keys) + 1)

    mean_1   = np.array([np.mean(all_grads_1[k]) for k in sorted_keys])
    std_1    = np.array([np.std(all_grads_1[k])  for k in sorted_keys])
    mean_2   = np.array([np.mean(all_grads_2[k]) for k in sorted_keys])
    std_2    = np.array([np.std(all_grads_2[k])  for k in sorted_keys])
    mean_cos = np.array([np.mean(all_cosines[k]) for k in sorted_keys])
    std_cos  = np.array([np.std(all_cosines[k])  for k in sorted_keys])

    # 平滑处理 Norm 1 均值
    smooth_window = 1
    mean_1_smooth = pd.Series(mean_1).rolling(window=smooth_window, center=True, min_periods=1).mean().values

    # --- [C. 绘图配置] ---
    setup_plot_style()
    # 创建共享 X 轴的上下子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    # 调整间距，使两个面板紧凑对齐
    fig.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.12, hspace=0.12)

    # --- 1. 子图 (a): 绘制梯度幅值 (Norms) ---
    # Norm 1
    ax1.plot(x_range, mean_1_smooth, label=r'Current $\|\mathbf{g}_1\|$', 
             color='#377eb8', linestyle='-', linewidth=2.5, zorder=3)
    ax1.fill_between(x_range, mean_1_smooth - std_1, mean_1_smooth + std_1, 
                     color='#377eb8', alpha=0.15, zorder=2, edgecolor='none')
    
    # Norm 2
    ax1.plot(x_range, mean_2, label=r'Replay $\|\mathbf{g}_2\|$', 
             color='#e41a1c', linestyle='--', linewidth=2.5, zorder=3)
    ax1.fill_between(x_range, mean_2 - std_2, mean_2 + std_2, 
                     color='#e41a1c', alpha=0.15, zorder=2, edgecolor='none')

    # --- 2. 子图 (b): 绘制方向一致性 (Similarity) ---
    ax2.plot(x_range, mean_cos, label=r'Cosine Similarity $\cos(\theta)$', 
             color='#4daf4a', linestyle='-.', linewidth=2.5, zorder=3)
    ax2.fill_between(x_range, mean_cos - std_cos, mean_cos + std_cos, 
                     color='#4daf4a', alpha=0.15, zorder=2, edgecolor='none')

    # --- 3. 统一风格深度定制 ---
    for ax in [ax1, ax2]:
        # 封闭黑框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)
        
        # 向外刻度
        ax.tick_params(axis='both', which='major', direction='out', 
                       bottom=True, top=False, left=True, right=False, 
                       length=6, width=1.2, labelsize=16)
        
        # 仅横向网格线 (仅 y 轴)
        ax.grid(False, axis='x')
        ax.grid(True, axis='y', linestyle=':', alpha=0.8, color='#333333', linewidth=1.0, zorder=0)

    # --- 4. 坐标轴个性化设置 ---
    # 子图 (a)
    ax1.set_ylabel('Gradient Norms', fontsize=20, labelpad=15)
    ax1.set_ylim(0, max(mean_1_smooth + std_1) * 1.3)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    
    # 子图 (b)
    ax2.set_ylabel('Cosine Similarity', fontsize=20, labelpad=15)
    ax2.set_ylim(-0.2, 1.0)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # 底部 X 轴
    ax2.set_xlabel('Update Steps (Sequential Index)', labelpad=15, fontsize=20)
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))

    # --- 5. 图例配置 ---
    ax1.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=14, framealpha=1.0, facecolor='white')
    ax2.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=14, framealpha=1.0, facecolor='white')

    # --- 6. 面板标注 (a) (b) ---
    ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes, fontsize=22, fontweight='bold', va='top')
    ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes, fontsize=22, fontweight='bold', va='top')

    # --- 7. 保存输出 ---
    fig_dir = os.path.join(result_path, 'grad_figures_avg_stacked')
    os.makedirs(fig_dir, exist_ok=True)
    file_base = f'grad_avg_stacked_seed{seed}_allsubs'
    
    plt.savefig(os.path.join(fig_dir, f'{file_base}.svg'), format='svg')
    plt.savefig(os.path.join(fig_dir, f'{file_base}.png'), format='png', dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"✅ 多被试均叠梯度图已生成: {file_base}.svg")

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
            # 用法示例：汇聚所有被试，绘制均值±方差图
            grad_visual_avg_stacked(str(args.result_dir), args.SEED, N)

            
    
            
