import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np

class CalibratedPseudoLabels:
    def __init__(self, n_classes, n_bins=10, alpha=0.9, gamma=1.0, device='cuda'):
        """
        校准伪标签类
        
        Args:
            n_classes: 类别数量
            n_bins: 置信度分箱数量
            alpha: EMA衰减因子
            gamma: 校准强度控制因子
            device: 设备
        """
        self.n_classes = n_classes
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        
        # 初始化EMA统计量
        self.ema_acc = torch.zeros(n_classes, n_bins, device=device)  # 估计的准确率
        self.ema_conf = torch.zeros(n_classes, n_bins, device=device)  # 平均置信度
        self.bin_counts = torch.zeros(n_classes, n_bins, device=device)  # 每个bin的样本计数
        self.class_counts = torch.zeros(n_classes, device=device)  # 每个类别的样本计数
        
        # 初始化Δ向量
        self.delta = torch.zeros(n_classes, device=device)
        
    def _get_bin_index(self, confidence):
        """根据置信度确定bin索引"""
        # 将[0, 1]区间分成n_bins个等宽区间
        bin_idx = torch.floor(confidence * self.n_bins).long()
        # 处理边界情况：置信度为1.0时应该落在最后一个bin
        bin_idx = torch.clamp(bin_idx, 0, self.n_bins - 1)
        return bin_idx
    
    def update_stats(self, logits):
        """
        更新EMA统计量
        
        Args:
            logits: 模型输出的logits, shape (batch_size, n_classes)
        """
        probs = F.softmax(logits, dim=1)
        confidences, pred_classes = torch.max(probs, dim=1)
        
        batch_size = logits.shape[0]
        
        for i in range(batch_size):
            conf = confidences[i].item()
            pred_class = pred_classes[i].item()
            
            # 确定bin索引
            bin_idx = self._get_bin_index(torch.tensor(conf, device=self.device))
            
            # 更新计数
            self.bin_counts[pred_class, bin_idx] += 1
            self.class_counts[pred_class] += 1
            
            # 更新EMA统计量
            # 假设伪标签正确，所以准确率观测值为1
            acc_update = 1.0
            conf_update = conf
            
            # 使用EMA更新
            if self.bin_counts[pred_class, bin_idx] == 1:  # 第一次遇到这个bin
                self.ema_acc[pred_class, bin_idx] = acc_update
                self.ema_conf[pred_class, bin_idx] = conf_update
            else:
                self.ema_acc[pred_class, bin_idx] = (
                    self.alpha * self.ema_acc[pred_class, bin_idx] + 
                    (1 - self.alpha) * acc_update
                )
                self.ema_conf[pred_class, bin_idx] = (
                    self.alpha * self.ema_conf[pred_class, bin_idx] + 
                    (1 - self.alpha) * conf_update
                )
    
    def compute_delta(self):
        """计算校准误差向量Δ"""
        # 重置delta
        self.delta = torch.zeros(self.n_classes, device=self.device)
        
        for class_idx in range(self.n_classes):
            if self.class_counts[class_idx] == 0:
                continue
                
            class_delta = 0.0
            for bin_idx in range(self.n_bins):
                if self.bin_counts[class_idx, bin_idx] > 0:
                    # 计算权重p_i,b
                    p_ib = self.bin_counts[class_idx, bin_idx] / self.class_counts[class_idx]
                    
                    # 计算该bin的校准误差贡献
                    acc_est = self.ema_acc[class_idx, bin_idx]
                    conf_est = self.ema_conf[class_idx, bin_idx]
                    bin_delta = p_ib * (acc_est - conf_est)
                    
                    class_delta += bin_delta
            
            self.delta[class_idx] = class_delta
    
    def calibrate_probs(self, logits):
        """
        校准logits对应的概率
        
        Args:
            logits: 模型输出的logits, shape (batch_size, n_classes)
            
        Returns:
            calibrated_probs: 校准后的概率, shape (batch_size, n_classes)
        """
        # 更新统计量
        self.update_stats(logits)
        
        # 计算当前的Δ向量
        self.compute_delta()
        
        # 获取原始概率
        original_probs = F.softmax(logits, dim=1)
        
        # 应用校准：y_tilde = y_hat + γ * Δ
        calibrated_probs = original_probs + self.gamma * self.delta.unsqueeze(0)
        
        # 确保概率非负（裁剪负值）
        calibrated_probs = torch.clamp(calibrated_probs, min=0.0)
        
        # 重新归一化，使得每行和为1
        calibrated_probs = calibrated_probs / calibrated_probs.sum(dim=1, keepdim=True)
        
        return calibrated_probs
    
    def get_calibration_info(self):
        """获取校准信息，用于调试"""
        return {
            'delta': self.delta.cpu().numpy(),
            'class_counts': self.class_counts.cpu().numpy(),
            'ema_acc': self.ema_acc.cpu().numpy(),
            'ema_conf': self.ema_conf.cpu().numpy()
        }

