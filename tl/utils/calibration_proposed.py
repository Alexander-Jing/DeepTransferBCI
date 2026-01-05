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


class DynamicThresholdSelector:
    def __init__(self, num_classes, base_threshold=0.7, momentum=0.9, min_threshold=0.4):
        """
        初始化动态阈值选择器。
        
        Args:
            num_classes (int): 分类总数 (e.g., 4 for MI-EEG)
            base_threshold (float): 基准阈值 (你之前设置的固定阈值，如 0.6 或 0.7)
            momentum (float): 移动平均的动量 (0-1), 越大则历史信息权重越大，更新越平滑。
            min_threshold (float): 阈值下限，防止对某个极其困难的类别门槛降得过低引入噪声。
        """
        self.num_classes = num_classes
        self.base_threshold = base_threshold
        self.momentum = momentum
        self.min_threshold = min_threshold
        
        # 初始化每个类别的平均置信度 (初始化为基准阈值，避免冷启动时波动过大)
        # 使用 CPU tensor 存储状态，避免占用 GPU 显存
        self.class_avg_conf = torch.ones(num_classes) * base_threshold

    def check_sample(self, logits):
        """
        处理单个样本，判断是否应该存入 Buffer。
        
        Args:
            logits (torch.Tensor): 模型的原始输出 (Logits), shape应为 (1, C) 或 (C,)
        
        Returns:
            is_selected (bool): 是否通过筛选
            pseudo_label (int): 预测的伪标签
            confidence (float): 该样本的置信度
            current_thresh (float): 该样本应用的具体阈值 (用于记录或调试)
        """
        # 1. 预处理输入，确保不计算梯度
        with torch.no_grad():
            if logits.dim() == 2:
                logits = logits.squeeze(0) # 变成 (C,)
            
            # 计算概率和预测结果
            probs = torch.softmax(logits, dim=0)
            confidence, pseudo_label = torch.max(probs, dim=0)
            
            # 转为 Python标量，方便计算和返回
            conf_val = confidence.item()
            label_idx = pseudo_label.item()
            
            # 2. 更新该类别的平均置信度 (EMA)
            # 公式: new_avg = m * old_avg + (1-m) * current_conf
            old_avg = self.class_avg_conf[label_idx].item()
            new_avg = self.momentum * old_avg + (1 - self.momentum) * conf_val
            self.class_avg_conf[label_idx] = new_avg
            
            # 3. 计算动态阈值
            # 逻辑: 如果当前类别的平均置信度(new_avg)比所有类别的最大值(max_avg)低，
            # 说明这个类很难，我们要降低它的门槛。
            max_avg_conf = self.class_avg_conf.max()
            
            # 归一化因子 (加一个极小值防止除零)
            normalize_factor = max_avg_conf.item() + 1e-6
            
            # 核心公式: Threshold_c = Base * (Avg_c / Max_Avg_All)
            dynamic_threshold = self.base_threshold * (new_avg / normalize_factor)
            
            # 截断: 确保阈值不低于下限，也不超过1.0
            dynamic_threshold = max(dynamic_threshold, self.min_threshold)
            dynamic_threshold = min(dynamic_threshold, 1.0)
            
            # 4. 判断是否选中
            is_selected = conf_val >= dynamic_threshold
            
            return is_selected, label_idx, conf_val, dynamic_threshold

    def reset(self):
        """重置状态 (例如在切换新的Subject时使用)"""
        self.class_avg_conf = torch.ones(self.num_classes) * self.base_threshold