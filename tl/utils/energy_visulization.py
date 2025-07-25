import os.path as osp
import os
import numpy as np
import random
import argparse

import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data as Data
import moabb
import mne
import learn2learn as l2l
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
from scipy.linalg import fractional_matrix_power
import torch.optim as optim
import matplotlib.pyplot as plt

from tl.utils.alg_utils import EA, EA_online
from tl.utils.utils import makedir_if_not_exist

def cal_energy_source(loader, model, args):
    
    # Initialize list to store free energy for each sample
    free_energies = []
    model.eval()
    
    # Initialize test reference matrix for Incremental EA if alignment needed
    if args.align:
        R = 0
    
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cpu()
            if i == 0:
                data_cum = inputs.float().cpu()
            else:
                data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)
            
            # Move to GPU if not running locally
            if args.data_env != 'local':
                inputs = inputs.cuda()
            
            # Get model outputs
            _, outputs = model(inputs)
            outputs = outputs.float().cpu()
            
            # Calculate free energy for each sample in batch
            log_sum_exp = tr.logsumexp(outputs, dim=1)  # log(∑exp(z_i))
            batch_energies = -log_sum_exp  # F = -log(∑exp(z_i))
            
            # Store energies for this batch
            free_energies.extend(batch_energies.tolist())

    # Calculate mean free energy
    mean_free_energy = tr.mean(tr.tensor(free_energies)).item()
    # print(f"Average Free Energy: {mean_free_energy:.4f}")

    return mean_free_energy, free_energies


def cal_energy_online(loader, model, args):
    
    # Initialize list to store free energy for each sample
    free_energies = []
    model.eval()
    
    # Initialize test reference matrix for Incremental EA if alignment needed
    if args.align:
        R = 0
    
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cpu()
            if i == 0:
                data_cum = inputs.float().cpu()
            else:
                data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)
            
            # Apply online alignment if required
            if args.align:
                # Update reference matrix
                R = EA_online(inputs.reshape(args.chn, args.time_sample_num), R, i)
                sqrtRefEA = fractional_matrix_power(R, -0.5)
                # Transform current test sample
                inputs = np.dot(sqrtRefEA, inputs)
                inputs = inputs.reshape(1, 1, args.chn, args.time_sample_num)
                inputs = torch.from_numpy(inputs).to(torch.float32)
            
            # Move to GPU if not running locally
            if args.data_env != 'local':
                inputs = inputs.cuda()
            
            # Get model outputs
            _, outputs = model(inputs)
            outputs = outputs.float().cpu()
            
            # Calculate free energy for each sample in batch
            log_sum_exp = tr.logsumexp(outputs, dim=1)  # log(∑exp(z_i))
            batch_energies = -log_sum_exp  # F = -log(∑exp(z_i))
            
            # Store energies for this batch
            free_energies.extend(batch_energies.tolist())

    # Calculate mean free energy
    mean_free_energy = tr.mean(tr.tensor(free_energies)).item()
    # print(f"Average Free Energy: {mean_free_energy:.4f}")

    return mean_free_energy, free_energies





def cal_EnergyEntropy_source(loader, model, args):
    y_true = []       # Store true labels
    y_pred = []       # Store predicted labels
    free_energies = []  # Store free energy for each sample
    entropies = []      # Store entropy for each sample
    model.eval()
    
    # Initialize test reference matrix for Incremental EA if alignment is enabled
    if args.align:
        R = 0
        
    with tr.no_grad():  # Disable gradient calculation for efficiency
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cpu()
            labels = data[1]
            
            # Cumulative data storage (not used in entropy/free energy calculation)
            if i == 0:
                data_cum = inputs.float().cpu()
            else:
                data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)
            
            # Move to GPU if not running locally
            if args.data_env != 'local':
                inputs = inputs.cuda()
                
            # Get model outputs
            _, outputs = model(inputs)
            outputs = outputs.float().cpu()
            labels = labels.float().cpu()
            
            # Get predictions
            _, predict = tr.max(outputs, 1)
            pred = tr.squeeze(predict).float()
            y_pred.extend(pred.cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())

            # ================== Free Energy Calculation ==================
            # Free energy formula: F = -log(∑exp(z_i))
            log_sum_exp = tr.logsumexp(outputs, dim=1)  # log(∑exp(z_i))
            free_energy = -log_sum_exp  # F = -log(∑exp(z_i))
            free_energies.extend(free_energy.tolist())  

            # ================== Entropy Calculation ==================
            # Compute softmax probabilities
            probs = F.softmax(outputs, dim=1)
            
            # Entropy formula: H(p) = -Σp_i * log(p_i)
            # Add epsilon to avoid log(0) - gradient-safe calculation
            log_probs = torch.log(probs + 1e-10)
            sample_entropies = -torch.sum(probs * log_probs, dim=1)
            entropies.extend(sample_entropies.tolist())
            # =========================================================

            # Store outputs and labels for score calculation
            if i == 0:
                all_output = outputs.float().cpu()
                all_label = labels.float()
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)

    # Calculate and print mean free energy
    mean_free_energy = tr.mean(tr.tensor(free_energies)).item()
    print(f"Average Free Energy: {mean_free_energy:.4f}")  
    
    # Calculate and print mean entropy
    mean_entropy = tr.mean(tr.tensor(entropies)).item()
    print(f"Average Entropy: {mean_entropy:.4f}")  

    # Compute final score
    if hasattr(args, 'balanced') and args.balanced:
        score = accuracy_score(y_true, y_pred)
    else:
        all_output = nn.Softmax(dim=1)(all_output)
        true = all_label.cpu()
        pred = all_output[:, 1].detach().numpy()
        score = roc_auc_score(true, pred)

    # Return metrics and values
    return (y_true, y_pred), free_energies, entropies


def cal_EnergyEntropy_online(loader, model, args):
    
    y_true = []       # Store true labels
    y_pred = []       # Store predicted labels
    free_energies = []  # Store free energy for each sample
    entropies = []      # Store entropy for each sample
    model.eval()
    
    # Initialize test reference matrix for Incremental EA if alignment is enabled
    if args.align:
        R = 0
        
    with tr.no_grad():  # Disable gradient calculation for efficiency
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cpu()
            labels = data[1]
            
            # Cumulative data storage (not used in entropy/free energy calculation)
            if i == 0:
                data_cum = inputs.float().cpu()
            else:
                data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)

            # Apply online alignment if enabled
            if args.align:
                # Update reference matrix
                R = EA_online(inputs.reshape(args.chn, args.time_sample_num), R, i)
                sqrtRefEA = fractional_matrix_power(R, -0.5)
                # Transform current test sample
                inputs = np.dot(sqrtRefEA, inputs)
                inputs = inputs.reshape(1, 1, args.chn, args.time_sample_num)
                inputs = torch.from_numpy(inputs).to(torch.float32)
            
            # Move to GPU if not running locally
            if args.data_env != 'local':
                inputs = inputs.cuda()
                
            # Get model outputs
            _, outputs = model(inputs)
            outputs = outputs.float().cpu()
            labels = labels.float().cpu()
            
            # Get predictions
            _, predict = tr.max(outputs, 1)
            pred = tr.squeeze(predict).float()
            y_pred.append(pred.item())
            y_true.append(labels.item())

            # ================== Free Energy Calculation ==================
            # Free energy formula: F = -log(∑exp(z_i))
            log_sum_exp = tr.logsumexp(outputs, dim=1)  # log(∑exp(z_i))
            free_energy = -log_sum_exp  # F = -log(∑exp(z_i))
            free_energies.extend(free_energy.tolist())  

            # ================== Entropy Calculation ==================
            # Compute softmax probabilities
            probs = F.softmax(outputs, dim=1)
            
            # Entropy formula: H(p) = -Σp_i * log(p_i)
            # Add epsilon to avoid log(0) - gradient-safe calculation
            log_probs = torch.log(probs + 1e-10)
            sample_entropies = -torch.sum(probs * log_probs, dim=1)
            entropies.extend(sample_entropies.tolist())
            # =========================================================

            # Store outputs and labels for score calculation
            if i == 0:
                all_output = outputs.float().cpu()
                all_label = labels.float()
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)

    # Calculate and print mean free energy
    mean_free_energy = tr.mean(tr.tensor(free_energies)).item()
    print(f"Average Free Energy: {mean_free_energy:.4f}")  
    
    # Calculate and print mean entropy
    mean_entropy = tr.mean(tr.tensor(entropies)).item()
    print(f"Average Entropy: {mean_entropy:.4f}")  

    # Compute final score
    if hasattr(args, 'balanced') and args.balanced:
        score = accuracy_score(y_true, y_pred)
    else:
        all_output = nn.Softmax(dim=1)(all_output)
        true = all_label.cpu()
        pred = all_output[:, 1].detach().numpy()
        score = roc_auc_score(true, pred)

    # Return metrics and values
    return (y_true, y_pred), free_energies, entropies





def EnergyEntropyVisualization(src_data, tar_data, args, visual_path="visual_EnergyEnrotpy"):
    """
    可视化源域和目标域样本在能量-熵空间的分布
    
    参数:
    src_data: 源域数据元组 (free_energies_src, entropies_src)
    tar_data: 目标域数据元组 (y_true_tar, y_pred_tar, free_energies_tar, entropies_tar)
    """
    # 解包数据
    free_energies_src, entropies_src = src_data
    y_true_tar, y_pred_tar, free_energies_tar, entropies_tar = tar_data
    
    plt.figure(figsize=(10, 8))
    
    # 1. 绘制源域样本 (使用蓝色叉号标记)
    plt.scatter(free_energies_src, entropies_src, 
                c='blue', marker='x', alpha=0.6, 
                label='Source Domain', s=30)
    
    # 2. 绘制目标域样本 (使用灰色圆圈标记)
    plt.scatter(free_energies_tar, entropies_tar, 
                c='red', marker='o', alpha=0.4, 
                label='Target Domain (All)', s=25)
    
    # 3. 标记预测正确的目标域样本 (使用绿色星号)
    correct_mask = np.array(y_true_tar) == np.array(y_pred_tar)
    correct_energies = np.array(free_energies_tar)[correct_mask]
    correct_entropies = np.array(entropies_tar)[correct_mask]
    
    plt.scatter(correct_energies, correct_entropies, 
                c='limegreen', marker='*', alpha=0.8, 
                label='Target Domain (Correct)', s=70)
    
    # 添加决策参考线 (根据数据分布动态计算位置)
    energy_thresh = np.median(free_energies_tar)
    entropy_thresh = np.median(entropies_tar)
    
    plt.axvline(x=energy_thresh, color='red', linestyle='--', alpha=0.3)
    plt.axhline(y=entropy_thresh, color='purple', linestyle='--', alpha=0.3)
    
    # 添加图例和标签
    plt.title('Energy-Entropy Domain Analysis', fontsize=14)
    plt.xlabel('Free Energy →', fontsize=12)
    plt.ylabel('Entropy →', fontsize=12)
    plt.legend(loc='upper right')
    
    # 添加网格和背景色增强可读性
    plt.grid(alpha=0.2)
    plt.gca().set_facecolor('#f8f8f8')
    
    # 自动调整坐标轴范围
    all_energies = np.concatenate([free_energies_src, free_energies_tar])
    all_entropies = np.concatenate([entropies_src, entropies_tar])
    
    plt.xlim(min(all_energies)*0.9, max(all_energies)*1.05)
    plt.ylim(min(all_entropies)*0.9, max(all_entropies)*1.05)
    
    plt.tight_layout()
    makedir_if_not_exist(os.path.join("./visualization", visual_path))
    plt.savefig(os.path.join("./visualization", visual_path, "visualization_EnergyEntropy_{}.png".format(args.idt)))


def EnergyEntropyVisualization_4dims(src_data, tar_data, args, visual_path="visual_EnergyEntropy"):
    """
    可视化源域和目标域样本在能量-熵空间的分布，并计算目标域在四个象限的准确率
    
    参数:
    src_data: 源域数据元组 (free_energies_src, entropies_src)
    tar_data: 目标域数据元组 (y_true_tar, y_pred_tar, free_energies_tar, entropies_tar)
    """
    # 解包数据
    free_energies_src, entropies_src = src_data
    y_true_tar, y_pred_tar, free_energies_tar, entropies_tar = tar_data
    
    # 计算源域的均值和目标域的正确预测掩码
    src_energy_mean = np.mean(free_energies_src)
    src_entropy_mean = np.mean(entropies_src)
    correct_mask = np.array(y_true_tar) == np.array(y_pred_tar)
    
    # 划分目标域四个象限
    tar_energies = np.array(free_energies_tar)
    tar_entropies = np.array(entropies_tar)
    
    # 定义四个象限的掩码
    quadrants = {
    "Q1 (High-E High-S)": (tar_energies >= src_energy_mean) & (tar_entropies >= src_entropy_mean),
    "Q2 (Low-E High-S)": (tar_energies < src_energy_mean) & (tar_entropies >= src_entropy_mean),
    "Q3 (Low-E Low-S)": (tar_energies < src_energy_mean) & (tar_entropies < src_entropy_mean),
    "Q4 (High-E Low-S)": (tar_energies >= src_energy_mean) & (tar_entropies < src_entropy_mean)
    }
    
    # 计算各象限准确率 [7,8](@ref)
    accuracies = {}
    for name, mask in quadrants.items():
        total = np.sum(mask)
        correct = np.sum(correct_mask & mask)
        accuracies[name] = correct / total if total > 0 else 0.0
    
    # 可视化设置
    plt.figure(figsize=(12, 9))
    # 1. 源域样本（蓝色叉号）
    plt.scatter(free_energies_src, entropies_src, c='blue', marker='x', alpha=0.6, label='Source Domain', s=30)
    # 2. 目标域全集（红色圆圈）
    plt.scatter(tar_energies, tar_entropies, c='red', marker='o', alpha=0.3, label='Target Domain (All)', s=20)
    # 3. 正确预测的目标域样本（绿色星号）
    plt.scatter(tar_energies[correct_mask], tar_entropies[correct_mask], 
                c='limegreen', marker='*', alpha=0.9, label='Correct Prediction', s=80)
    
    # 添加源域均值参考线
    plt.axvline(x=src_energy_mean, color='red', linestyle='-', alpha=0.7, lw=1.5)
    plt.axhline(y=src_entropy_mean, color='purple', linestyle='-', alpha=0.7, lw=1.5)
    
    # 标注象限准确率
    accuracy_text = "\n".join([f"{name}: {acc*100:.1f}%" for name, acc in accuracies.items()])
    plt.text(0.95, 0.95, accuracy_text, transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # 标签和标题
    plt.title(f'Energy-Entropy Analysis (Domain: {args.idt})', fontsize=14)
    plt.xlabel('Free Energy (E) →', fontsize=12)
    plt.ylabel('Entropy (S) →', fontsize=12)
    plt.legend(loc='lower left')
    
    # 优化显示范围
    all_energies = np.concatenate([free_energies_src, tar_energies])
    all_entropies = np.concatenate([entropies_src, tar_entropies])
    plt.xlim(np.min(all_energies)*0.95, np.max(all_energies)*1.05)
    plt.ylim(np.min(all_entropies)*0.95, np.max(all_entropies)*1.05)
    
    # 网格和背景
    plt.grid(alpha=0.2, ls=':')
    plt.gca().set_facecolor('#f0f8ff')
    
    # 象限标注
    # 在图中添加象限标签
    plt.text(src_energy_mean+1.5, src_entropy_mean*1.5, 'Q1: High-E High-S', fontsize=10, color='darkred')
    plt.text(src_energy_mean-3.5, src_entropy_mean*1.5, 'Q2: Low-E High-S', fontsize=10, color='darkred')
    plt.text(src_energy_mean-3.5, src_entropy_mean*0.5, 'Q3: Low-E Low-S', fontsize=10, color='darkred')
    plt.text(src_energy_mean+1.5, src_entropy_mean*0.5, 'Q4: High-E Low-S', fontsize=10, color='darkred')
    
    # 保存结果
    os.makedirs(os.path.join("./visualization", visual_path), exist_ok=True)
    plt.savefig(os.path.join("./visualization", visual_path, f"EnergyEntropy_{args.idt}.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()
