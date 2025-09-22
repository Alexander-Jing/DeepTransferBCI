import torch

class AdaptiveLRScheduler:
    def __init__(self, optimizer, base_lr=0.001, min_lr=1e-6, momentum=0.9):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.momentum = momentum  # 平滑熵变化的动量因子
        self.prev_entropy = None  # 保存上一轮熵值
        self.ema_entropy = None   # 指数移动平均熵（平滑噪声）
    
    def entropy_computation(self, logits):
        probs = logits.softmax(1)
        entropy = -probs * torch.log(probs + 1e-6)
        entropy = entropy.sum(1)
        return entropy.mean()

    def update_lr_entropy(self, logits):
        
        current_entropy = self.entropy_computation(logits)
        
        if self.prev_entropy is None:
            self.prev_entropy = current_entropy
            self.ema_entropy = current_entropy
            return

        # 指数移动平均平滑熵值（减少波动）
        self.ema_entropy = self.momentum * self.ema_entropy + (1 - self.momentum) * current_entropy
        
        # 计算熵变化率（相对上一轮）
        entropy_change = abs(self.ema_entropy - self.prev_entropy) / (self.prev_entropy + 1e-10)
        
        # 动态调整学习率：熵上升 → 降学习率；熵下降 → 升学习率
        new_lr = self.base_lr * (1 - entropy_change) + self.min_lr
        new_lr = max(self.min_lr, min(5*self.base_lr, new_lr))  # 限制范围
        
        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # 保存当前熵供下一轮使用
        self.prev_entropy = self.ema_entropy

        return new_lr
    
class AdaptiveLRScheduler_1:
    def __init__(self, optimizer, base_lr=0.001, min_lr=1e-6, momentum=0.9):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.momentum = momentum  # 平滑熵变化的动量因子
        self.prev_entropy = None  # 保存上一轮熵值
        self.ema_entropy = None   # 指数移动平均熵（平滑噪声）
    
    def entropy_computation(self, logits):
        probs = logits.softmax(1)
        entropy = -probs * torch.log(probs + 1e-6)
        entropy = entropy.sum(1)
        return entropy.mean()
    
    def _entropy_samples_normalized(self, logits):
        probs = logits.softmax(dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1)
        C = logits.size(1)  # categories
        C_tensor = torch.tensor(C, dtype=torch.float, device=logits.device)
        max_entropy = torch.log(C_tensor)

        if C == 1:
            normalized_entropy = torch.zeros_like(entropy)
        else:
            # normilize the entropy
            normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def update_lr_entropy(self, logits):
        
        current_entropy = self._entropy_samples_normalized(logits)
        
        entropy_avg = torch.mean(current_entropy)

        new_lr = self.base_lr * (1 - entropy_avg) + self.min_lr
        
        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        return new_lr
