import torch
import torch.nn as nn
import torch.nn.functional as F


def _entropy(logits):
    probs = logits.softmax(1)
    entropy = -probs * torch.log(probs + 1e-6)
    entropy = entropy.sum(1)
    return entropy.mean()

def _entropy_samples(logits):
    probs = logits.softmax(dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1)
    return entropy

def _energy_samples(logits):
    energy = -torch.logsumexp(logits,dim=1)
    return energy

def _mdr(logits):
    probs = logits.softmax(1)
    msoftmax = probs.mean(dim=0)
    MDR_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
    return MDR_loss

def _kl_loss(logits):
    probs = logits.softmax(1)
    msoftmax = probs.mean(dim=0)
    num_classes = logits.size(1)
    uniform = torch.ones_like(msoftmax) * (1.0 / num_classes)
    kl_loss = torch.sum(msoftmax * (torch.log(msoftmax + 1e-8) - torch.log(uniform + 1e-8)))
    
    return kl_loss

def _kl_loss_samples(logits):

    probs = logits.softmax(1)  # [batch_size, num_classes]
    
    num_classes = logits.size(1)
    uniform = torch.ones_like(probs) * (1.0 / num_classes)  # [batch_size, num_classes]
    
    kl_per_sample = probs * (torch.log(probs + 1e-8) - torch.log(uniform + 1e-8))
    kl_per_sample = torch.sum(kl_per_sample, dim=1)  # [batch_size]
    
    return kl_per_sample

def uniform_kl_loss(logits):
    """
    Calculate KL divergence between predicted distribution and uniform distribution per sample
    Args:
        logits: Raw model outputs, shape [batch_size, num_classes]
    Returns:
        kl_loss: Scalar loss value
    """
    # Get number of classes from logits dimension
    num_classes = logits.size(1)
    # Create uniform distribution vector [1/K, 1/K, ...]
    uniform = torch.ones_like(logits) * (1.0 / num_classes)  # Shape [batch_size, num_classes]
    # Compute predicted probability distribution (softmax normalized)
    probs = F.softmax(logits, dim=1)  # Shape [batch_size, num_classes]
    # Calculate KL divergence: KL(probs || uniform)
    kl_per_sample = probs * (torch.log(probs + 1e-8) - torch.log(uniform + 1e-8))  # Element-wise computation
    # Sum KL values across classes per sample
    kl_per_sample = torch.sum(kl_per_sample, dim=1)  # Shape [batch_size]
    # Average KL loss over batch
    kl_loss = torch.mean(kl_per_sample)
    
    return kl_loss

def softmax_entropy(x, x_ema):  # -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def symmetric_entropy_loss(p_student, p_ema):
    """
    Symmetric Entropy Loss between student and teacher distributions.
    
    Formula: L = 0.5 * [ -Σ(p_ema * log(p_student)) - Σ(p_student * log(p_ema)) ]
    
    Args:
        p_student: Probability distribution from student model (shape: [batch, classes])
        p_ema: Probability distribution from EMA teacher model (shape: [batch, classes])
    
    Returns:
        Symmetric loss value (scalar tensor)
    """
    # Term1: Cross-entropy of student relative to EMA targets
    term1 = -torch.sum(p_ema * torch.log(p_student + 1e-10), dim=1)  # 1e-10 for numerical stability
    
    # Term2: Cross-entropy of EMA relative to student targets
    term2 = -torch.sum(p_student * torch.log(p_ema + 1e-10), dim=1)
    
    # Symmetric combination: average of both terms
    loss_per_sample = 0.5 * (term1 + term2)
    
    # Batch-level mean reduction
    batch_loss = loss_per_sample.mean()
    
    return batch_loss






def contrastive_loss_samples(logits, thr=0.4, temperature=0.07):
    """
    实现图片中的对比损失函数公式
    参数:
        logits: 模型输出的logits [batch_size, num_classes]
        thr: 置信度阈值，用于过滤低置信度样本
        temperature: 对比损失温度参数
    """
    # 1. 计算每个样本的预测概率和伪标签
    prob_all, pseudo_labels = torch.max(torch.softmax(logits, dim=-1), dim=-1)
    
    # 2. 应用置信度阈值过滤
    conf_indices = prob_all >= thr
    if not torch.any(conf_indices):
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    # 3. 筛选高置信度样本
    conf_logits = logits[conf_indices]
    conf_labels = pseudo_labels[conf_indices]
    batch_size = conf_logits.size(0)
    
    # 4. 计算logits间的余弦相似度矩阵
    sim_matrix = F.cosine_similarity(
        conf_logits.unsqueeze(1),  # [batch_size, 1, num_classes]
        conf_logits.unsqueeze(0),  # [1, batch_size, num_classes]
        dim=-1
    )
    
    # 5. 创建正负样本掩码
    # 正样本：相同伪标签且非自身
    pos_mask = (conf_labels.unsqueeze(0) == conf_labels.unsqueeze(1)) & \
               (~torch.eye(batch_size, dtype=torch.bool, device=logits.device))
    
    # 负样本：不同伪标签
    neg_mask = conf_labels.unsqueeze(0) != conf_labels.unsqueeze(1)
    
    # 6. 计算每个样本的正样本数量
    pos_counts = pos_mask.sum(dim=1).float()  # [batch_size]
    valid_samples = pos_counts > 0  # 排除没有正样本的样本
    
    # 7. 初始化损失
    total_loss = torch.tensor(0.0, device=logits.device)
    valid_count = 0
    
    # 8. 遍历每个样本计算损失
    for i in range(batch_size):
        if not valid_samples[i]:
            continue  # 跳过没有正样本的样本
            
        # 获取当前样本的正样本索引
        pos_indices = torch.where(pos_mask[i])[0]
        
        # 计算分子：正样本的指数相似度之和
        numerator = torch.sum(torch.exp(sim_matrix[i, pos_indices] / temperature))
        
        # 获取当前样本的负样本索引
        neg_indices = torch.where(neg_mask[i])[0]
        
        # 计算分母：负样本的指数相似度之和
        denominator = torch.sum(torch.exp(sim_matrix[i, neg_indices] / temperature))
        
        # 计算损失项：-log(分子/分母)
        loss_term = -torch.log(numerator / denominator)
        
        # 除以正样本数量（|pos(i)|）
        loss_term /= pos_counts[i]
        
        # 累加损失
        total_loss += loss_term
        valid_count += 1
    
    # 9. 计算平均损失
    if valid_count > 0:
        return total_loss / valid_count
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

def contrastive_loss_samplefeatures(logits, features, thr=0.0, temperature=0.07):
    """
    使用样本特征计算相似度的对比损失函数
    
    参数:
        logits: 模型输出的logits [batch_size, num_classes]
        features: 样本的特征向量 [batch_size, feature_dim]
        thr: 置信度阈值，用于过滤低置信度样本
        temperature: 对比损失温度参数
    """
    # 1. 计算每个样本的预测概率和伪标签 (使用logits)
    prob_all, pseudo_labels = torch.max(torch.softmax(logits, dim=-1), dim=-1)
    
    # 2. 应用置信度阈值过滤
    conf_indices = prob_all >= thr
    if not torch.any(conf_indices):
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    # 3. 筛选高置信度样本的特征和标签
    conf_features = features[conf_indices]  # [batch_size, feature_dim]
    conf_labels = pseudo_labels[conf_indices]  # [batch_size]
    batch_size = conf_features.size(0)
    
    # 4. 计算特征间的余弦相似度矩阵 (使用features)
    sim_matrix = F.cosine_similarity(
        conf_features.unsqueeze(1),  # [batch_size, 1, feature_dim]
        conf_features.unsqueeze(0),  # [1, batch_size, feature_dim]
        dim=-1
    )
    
    # 5. 创建正负样本掩码
    # 正样本：相同伪标签且非自身
    pos_mask = (conf_labels.unsqueeze(0) == conf_labels.unsqueeze(1)) & \
               (~torch.eye(batch_size, dtype=torch.bool, device=logits.device))
    
    # 负样本：不同伪标签
    neg_mask = conf_labels.unsqueeze(0) != conf_labels.unsqueeze(1)
    
    # 6. 计算每个样本的正样本数量
    pos_counts = pos_mask.sum(dim=1).float()  # [batch_size]
    valid_samples = pos_counts > 0  # 排除没有正样本的样本
    
    # 7. 初始化损失
    total_loss = torch.tensor(0.0, device=logits.device)
    valid_count = 0
    
    # 8. 遍历每个样本计算损失
    for i in range(batch_size):
        if not valid_samples[i]:
            continue  # 跳过没有正样本的样本
            
        # 获取当前样本的正样本索引
        pos_indices = torch.where(pos_mask[i])[0]
        
        # 计算分子：正样本的特征相似度指数和
        numerator = torch.sum(torch.exp(sim_matrix[i, pos_indices] / temperature))
        
        # 获取当前样本的负样本索引
        neg_indices = torch.where(neg_mask[i])[0]
        
        # 计算分母：负样本的特征相似度指数和
        denominator = torch.sum(torch.exp(sim_matrix[i, neg_indices] / temperature))
        
        # 计算损失项：-log(分子/分母)
        loss_term = -torch.log(numerator / denominator)
        
        # 除以正样本数量（|pos(i)|）
        loss_term /= pos_counts[i]
        
        # 累加损失
        total_loss += loss_term
        valid_count += 1
    
    # 9. 计算平均损失
    if valid_count > 0:
        return total_loss / valid_count
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
def contrastive_loss_samples_selection(logits, ratio=0.75, temperature=0.07):
    """
    实现基于能量和熵的对比损失函数
    参数:
        logits: 模型输出的logits [batch_size, num_classes]
        ratio: 用于筛选权重样本的阈值百分比（比例）
        temperature: 对比损失温度参数
    """
    batch_size = logits.size(0)
    
    # 1. 计算每个样本的能量和熵
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    entropy = _entropy_samples(logits)
    
    # 2. 计算样本权重: S_i * log(1 + exp(E_i))
    weights = entropy * torch.log(1 + torch.exp(energy))
    
    # 3. 应用权重阈值筛选样本
    k = max(1, int(batch_size * ratio))  # 至少选择1个样本
    _, conf_indices = torch.topk(weights, k, largest=False, sorted=True)
    
    # 4. 获取高权重样本的logits和伪标签
    probs = torch.softmax(logits, dim=1)
    _, pseudo_labels = torch.max(probs, dim=1)
    conf_logits = logits[conf_indices]
    conf_labels = pseudo_labels[conf_indices]
    k = conf_logits.size(0)  # 实际选中的样本数量
    
    # 5. 计算logits间的余弦相似度矩阵
    sim_matrix = F.cosine_similarity(
        conf_logits.unsqueeze(1),  # [k, 1, num_classes]
        conf_logits.unsqueeze(0),  # [1, k, num_classes]
        dim=-1
    )
    
    # 6. 创建正负样本掩码
    # 正样本：相同伪标签且非自身
    pos_mask = (conf_labels.unsqueeze(0) == conf_labels.unsqueeze(1)) & \
               (~torch.eye(k, dtype=torch.bool, device=logits.device))
    
    # 负样本：不同伪标签
    neg_mask = conf_labels.unsqueeze(0) != conf_labels.unsqueeze(1)
    
    # 7. 计算每个样本的正样本数量
    pos_counts = pos_mask.sum(dim=1).float()  # [k]
    valid_samples = pos_counts > 0  # 排除没有正样本的样本
    
    # 8. 初始化损失
    total_loss = torch.tensor(0.0, device=logits.device)
    valid_count = 0
    
    # 9. 遍历每个样本计算损失
    for i in range(k):
        if not valid_samples[i]:
            continue  # 跳过没有正样本的样本
            
        # 获取当前样本的正样本索引
        pos_indices = torch.where(pos_mask[i])[0]
        
        # 计算分子：正样本的指数相似度之和
        numerator = torch.sum(torch.exp(sim_matrix[i, pos_indices] / temperature))
        
        # 获取当前样本的负样本索引
        neg_indices = torch.where(neg_mask[i])[0]
        
        # 计算分母：负样本的指数相似度之和
        denominator = torch.sum(torch.exp(sim_matrix[i, neg_indices] / temperature))
        
        # 计算损失项：-log(分子/分母)
        loss_term = -torch.log(numerator / (denominator + 1e-8))  # 添加小量避免除零
        
        # 除以正样本数量（|pos(i)|）
        loss_term /= pos_counts[i]
        
        # 累加损失
        total_loss += loss_term
        valid_count += 1
    
    # 10. 计算平均损失
    if valid_count > 0:
        return total_loss / valid_count
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)


def contrastive_loss_samples_selection_dropout(logits, probs_dropout, ratio=0.75, temperature=0.07):
    """
    实现基于能量和熵的对比损失函数
    参数:
        logits: 模型输出的logits [batch_size, num_classes]
        ratio: 用于筛选权重样本的阈值百分比（比例）
        temperature: 对比损失温度参数
    """
    batch_size = logits.size(0)
    
    # 1. 计算每个样本的能量和熵
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    entropy = _entropy_samples(logits)
    
    # 2. 计算样本权重: S_i * log(1 + exp(E_i))
    weights = entropy * torch.log(1 + torch.exp(energy))
    
    # 3. 应用权重阈值筛选样本
    k = max(1, int(batch_size * ratio))  # 至少选择1个样本
    _, conf_indices = torch.topk(weights, k, largest=False, sorted=True)
    
    # 4. 获取高权重样本的logits和伪标签
    _, pseudo_labels = torch.max(probs_dropout.detach(), dim=1)
    conf_logits = logits[conf_indices]
    conf_labels = pseudo_labels[conf_indices]
    k = conf_logits.size(0)  # 实际选中的样本数量
    
    # 5. 计算logits间的余弦相似度矩阵
    sim_matrix = F.cosine_similarity(
        conf_logits.unsqueeze(1),  # [k, 1, num_classes]
        conf_logits.unsqueeze(0),  # [1, k, num_classes]
        dim=-1
    )
    
    # 6. 创建正负样本掩码
    # 正样本：相同伪标签且非自身
    pos_mask = (conf_labels.unsqueeze(0) == conf_labels.unsqueeze(1)) & \
               (~torch.eye(k, dtype=torch.bool, device=logits.device))
    
    # 负样本：不同伪标签
    neg_mask = conf_labels.unsqueeze(0) != conf_labels.unsqueeze(1)
    
    # 7. 计算每个样本的正样本数量
    pos_counts = pos_mask.sum(dim=1).float()  # [k]
    valid_samples = pos_counts > 0  # 排除没有正样本的样本
    
    # 8. 初始化损失
    total_loss = torch.tensor(0.0, device=logits.device)
    valid_count = 0
    
    # 9. 遍历每个样本计算损失
    for i in range(k):
        if not valid_samples[i]:
            continue  # 跳过没有正样本的样本
            
        # 获取当前样本的正样本索引
        pos_indices = torch.where(pos_mask[i])[0]
        
        # 计算分子：正样本的指数相似度之和
        numerator = torch.sum(torch.exp(sim_matrix[i, pos_indices] / temperature))
        
        # 获取当前样本的负样本索引
        neg_indices = torch.where(neg_mask[i])[0]
        
        # 计算分母：负样本的指数相似度之和
        denominator = torch.sum(torch.exp(sim_matrix[i, neg_indices] / temperature))
        
        # 计算损失项：-log(分子/分母)
        loss_term = -torch.log(numerator / (denominator + 1e-8))  # 添加小量避免除零
        
        # 除以正样本数量（|pos(i)|）
        loss_term /= pos_counts[i]
        
        # 累加损失
        total_loss += loss_term
        valid_count += 1
    
    # 10. 计算平均损失
    if valid_count > 0:
        return total_loss / valid_count
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

def contrastive_loss_samples_selection_1(logits, ratio=0.75, temperature=0.07):
    """
    实现基于熵的对比损失函数
    参数:
        logits: 模型输出的logits [batch_size, num_classes]
        ratio: 用于筛选权重样本的阈值百分比（比例）
        temperature: 对比损失温度参数
    """
    batch_size = logits.size(0)
    
    # 1. 计算每个样本的熵
    entropy = _entropy_samples(logits)
    
    # 2. 计算样本权重: S_i
    weights = entropy
    
    # 3. 应用权重阈值筛选样本
    k = max(1, int(batch_size * ratio))  # 至少选择1个样本
    _, conf_indices = torch.topk(weights, k, largest=False, sorted=True)
    
    # 4. 获取高权重样本的logits和伪标签
    probs = torch.softmax(logits, dim=1)
    _, pseudo_labels = torch.max(probs, dim=1)
    conf_logits = logits[conf_indices]
    conf_labels = pseudo_labels[conf_indices]
    k = conf_logits.size(0)  # 实际选中的样本数量
    
    # 5. 计算logits间的余弦相似度矩阵
    sim_matrix = F.cosine_similarity(
        conf_logits.unsqueeze(1),  # [k, 1, num_classes]
        conf_logits.unsqueeze(0),  # [1, k, num_classes]
        dim=-1
    )
    
    # 6. 创建正负样本掩码
    # 正样本：相同伪标签且非自身
    pos_mask = (conf_labels.unsqueeze(0) == conf_labels.unsqueeze(1)) & \
               (~torch.eye(k, dtype=torch.bool, device=logits.device))
    
    # 负样本：不同伪标签
    neg_mask = conf_labels.unsqueeze(0) != conf_labels.unsqueeze(1)
    
    # 7. 计算每个样本的正样本数量
    pos_counts = pos_mask.sum(dim=1).float()  # [k]
    valid_samples = pos_counts > 0  # 排除没有正样本的样本
    
    # 8. 初始化损失
    total_loss = torch.tensor(0.0, device=logits.device)
    valid_count = 0
    
    # 9. 遍历每个样本计算损失
    for i in range(k):
        if not valid_samples[i]:
            continue  # 跳过没有正样本的样本
            
        # 获取当前样本的正样本索引
        pos_indices = torch.where(pos_mask[i])[0]
        
        # 计算分子：正样本的指数相似度之和
        numerator = torch.sum(torch.exp(sim_matrix[i, pos_indices] / temperature))
        
        # 获取当前样本的负样本索引
        neg_indices = torch.where(neg_mask[i])[0]
        
        # 计算分母：负样本的指数相似度之和
        denominator = torch.sum(torch.exp(sim_matrix[i, neg_indices] / temperature))
        
        # 计算损失项：-log(分子/分母)
        loss_term = -torch.log(numerator / (denominator + 1e-8))  # 添加小量避免除零
        
        # 除以正样本数量（|pos(i)|）
        loss_term /= pos_counts[i]
        
        # 累加损失
        total_loss += loss_term
        valid_count += 1
    
    # 10. 计算平均损失
    if valid_count > 0:
        return total_loss / valid_count
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    

def contrastive_loss_samples_selection_2(logits, ratio=0.75, temperature=0.07):
    """
    实现基于能量的对比损失函数
    参数:
        logits: 模型输出的logits [batch_size, num_classes]
        ratio: 用于筛选权重样本的阈值百分比（比例）
        temperature: 对比损失温度参数
    """
    batch_size = logits.size(0)
    
    # 1. 计算每个样本的熵
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    
    # 2. 计算样本权重: S_i
    weights = energy
    
    # 3. 应用权重阈值筛选样本
    k = max(1, int(batch_size * ratio))  # 至少选择1个样本
    _, conf_indices = torch.topk(weights, k, largest=False, sorted=True)
    
    # 4. 获取高权重样本的logits和伪标签
    probs = torch.softmax(logits, dim=1)
    _, pseudo_labels = torch.max(probs, dim=1)
    conf_logits = logits[conf_indices]
    conf_labels = pseudo_labels[conf_indices]
    k = conf_logits.size(0)  # 实际选中的样本数量
    
    # 5. 计算logits间的余弦相似度矩阵
    sim_matrix = F.cosine_similarity(
        conf_logits.unsqueeze(1),  # [k, 1, num_classes]
        conf_logits.unsqueeze(0),  # [1, k, num_classes]
        dim=-1
    )
    
    # 6. 创建正负样本掩码
    # 正样本：相同伪标签且非自身
    pos_mask = (conf_labels.unsqueeze(0) == conf_labels.unsqueeze(1)) & \
               (~torch.eye(k, dtype=torch.bool, device=logits.device))
    
    # 负样本：不同伪标签
    neg_mask = conf_labels.unsqueeze(0) != conf_labels.unsqueeze(1)
    
    # 7. 计算每个样本的正样本数量
    pos_counts = pos_mask.sum(dim=1).float()  # [k]
    valid_samples = pos_counts > 0  # 排除没有正样本的样本
    
    # 8. 初始化损失
    total_loss = torch.tensor(0.0, device=logits.device)
    valid_count = 0
    
    # 9. 遍历每个样本计算损失
    for i in range(k):
        if not valid_samples[i]:
            continue  # 跳过没有正样本的样本
            
        # 获取当前样本的正样本索引
        pos_indices = torch.where(pos_mask[i])[0]
        
        # 计算分子：正样本的指数相似度之和
        numerator = torch.sum(torch.exp(sim_matrix[i, pos_indices] / temperature))
        
        # 获取当前样本的负样本索引
        neg_indices = torch.where(neg_mask[i])[0]
        
        # 计算分母：负样本的指数相似度之和
        denominator = torch.sum(torch.exp(sim_matrix[i, neg_indices] / temperature))
        
        # 计算损失项：-log(分子/分母)
        loss_term = -torch.log(numerator / (denominator + 1e-8))  # 添加小量避免除零
        
        # 除以正样本数量（|pos(i)|）
        loss_term /= pos_counts[i]
        
        # 累加损失
        total_loss += loss_term
        valid_count += 1
    
    # 10. 计算平均损失
    if valid_count > 0:
        return total_loss / valid_count
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)


def contrastive_loss_samples_weighted(logits, thr=0.4, temperature=0.07):
    """
    实现包含低置信度样本的对比损失函数
    参数:
        logits: 模型输出的logits [batch_size, num_classes]
        thr: 置信度阈值，用于区分高/低置信度样本
        temperature: 对比损失温度参数
    """
    # 1. 计算每个样本的预测概率和伪标签
    probs = torch.softmax(logits, dim=-1)
    prob_all, pseudo_labels = torch.max(probs, dim=-1)
    num_classes = logits.size(-1)
    device = logits.device
    
    # 2. 分离高置信度和低置信度样本
    high_conf_mask = prob_all >= thr
    low_conf_mask = prob_all < thr
    
    high_logits = logits[high_conf_mask]
    high_labels = pseudo_labels[high_conf_mask]
    n_high = high_logits.size(0)
    
    low_logits = logits[low_conf_mask]
    low_probs = probs[low_conf_mask]
    n_low = low_logits.size(0)
    
    # 3. 计算高置信度样本的对比损失（原逻辑）
    high_loss = torch.tensor(0.0, device=device)
    n_valid_high = 0
    
    if n_high > 0:
        # 计算高置信度样本间的相似度
        sim_matrix_high = F.cosine_similarity(
            high_logits.unsqueeze(1), 
            high_logits.unsqueeze(0), 
            dim=-1
        )
        
        # 创建正负样本掩码
        pos_mask_high = (high_labels.unsqueeze(0) == high_labels.unsqueeze(1)) & \
                       (~torch.eye(n_high, dtype=torch.bool, device=device))
        neg_mask_high = high_labels.unsqueeze(0) != high_labels.unsqueeze(1)
        pos_counts_high = pos_mask_high.sum(dim=1).float()
        valid_high_samples = pos_counts_high > 0
        
        # 遍历每个高置信度样本计算损失
        for i in range(n_high):
            if not valid_high_samples[i]:
                continue
                
            pos_idx = torch.where(pos_mask_high[i])[0]
            neg_idx = torch.where(neg_mask_high[i])[0]
            
            numerator = torch.sum(torch.exp(sim_matrix_high[i, pos_idx] / temperature))
            denominator = torch.sum(torch.exp(sim_matrix_high[i, neg_idx] / temperature))
            
            loss_term = -torch.log(numerator / denominator)
            loss_term /= pos_counts_high[i]
            
            high_loss += loss_term
            n_valid_high += 1
    
    # 4. 计算低置信度样本的对比损失（新逻辑）
    low_loss = torch.tensor(0.0, device=device)
    n_valid_low = 0
    
    if n_low > 0 and n_high > 0:
        # 计算低置信度样本与高置信度样本间的相似度
        sim_matrix_low_high = F.cosine_similarity(
            low_logits.unsqueeze(1),  # [n_low, 1, D]
            high_logits.unsqueeze(0),  # [1, n_high, D]
            dim=-1
        )  # [n_low, n_high]
        
        # 遍历每个低置信度样本
        for i in range(n_low):
            sample_loss = torch.tensor(0.0, device=device)
            valid_categories = 0
            
            # 遍历每个可能的类别
            for c in range(num_classes):
                class_prob = low_probs[i, c]
                if class_prob < 1e-6:  # 忽略概率极小的类别
                    continue
                    
                # 获取当前类别的正负样本
                pos_idx = torch.where(high_labels == c)[0]
                neg_idx = torch.where(high_labels != c)[0]
                
                if len(pos_idx) == 0:
                    continue
                
                # 计算当前类别的对比损失
                pos_sim = sim_matrix_low_high[i, pos_idx]
                neg_sim = sim_matrix_low_high[i, neg_idx]
                
                numerator = torch.sum(torch.exp(pos_sim / temperature))
                denominator = torch.sum(torch.exp(neg_sim / temperature))
                
                # 避免分母过小导致数值不稳定
                if denominator < 1e-8:
                    continue
                
                loss_term = -torch.log(numerator / denominator)
                sample_loss += class_prob * loss_term
                valid_categories += 1
            
            if valid_categories > 0:
                # 标准化损失：平均权重总和应等于1
                sample_loss = sample_loss / (class_prob.sum() if valid_categories == 1 else valid_categories)
                low_loss += sample_loss
                n_valid_low += 1
    
    # 5. 合并两部分损失
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_valid = 0
    
    if n_valid_high > 0:
        high_loss_avg = high_loss / n_valid_high
        total_loss = total_loss + high_loss_avg
        total_valid += 1
    
    if n_valid_low > 0:
        low_loss_avg = low_loss / n_valid_low
        total_loss = total_loss + low_loss_avg
        total_valid += 1
    
    # 6. 计算最终平均损失
    if total_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    elif total_valid == 2:
        return total_loss / 2  # 两个损失分量等权平均
    else:
        return total_loss  # 只有一个分量有效



def _marginal_entropy(logits):
    probs = logits.softmax(1)
    marginal_probs = probs.mean(0)
    # return uniform loss
    return -(marginal_probs * (marginal_probs + 1e-6).log()).sum()


def _neg_mutual_information(logits):
    return _entropy(logits) - _marginal_entropy(logits)


def _neg_weighted_mutual_information(logits, lambda_info):
    return lambda_info * _entropy(logits) - _marginal_entropy(logits)


def _neg_weighted_mutual_information_on_marginal(logits, lambda_info):
    return _entropy(logits) - lambda_info * _marginal_entropy(logits)


def _calibrated_entropy(logits, gamma=1.0):
    """
    Computes the Calibrated Entropy Test-time Adaptation loss
    
    Args:
        logits (torch.Tensor): Raw model outputs with shape (batch_size, num_classes)
        gamma (float): Calibration hyperparameter controlling sensitivity to confidence difference
    
    Returns:
        torch.Tensor: Calibrated entropy loss scalar
    """
    # Compute class probabilities from logits
    probs = logits.softmax(1)  # Shape [B, C]
    
    # Calculate standard entropy per sample
    entropy_elements = -probs * torch.log(probs + 1e-6)  # Avoid log(0) with small epsilon
    sample_entropy = entropy_elements.sum(1)  # Shape [B]
    
    # Get highest (q_j) and second highest (q_k) probabilities
    top2_probs = torch.topk(probs, k=2, dim=1).values  # Top-2 probabilities [B, 2]
    q_j = top2_probs[:, 0]  # Max probability [B]
    q_k = top2_probs[:, 1]  # Second max probability [B]
    
    # Compute per-sample calibration factor
    delta = q_j - q_k  # Prediction confidence gap [B]
    calibration_factor = 1 + delta ** gamma  # Eq.(3): 1 + (q_j - q_k)^γ
    
    # Apply calibration and compute final loss
    calibrated_entropy = calibration_factor * sample_entropy  # [B]
    loss = calibrated_entropy.mean()  # Batch mean reduction
    
    return loss

def _weighted_lcs(input, cls_weight, thr=0.):
    classwise_virtual_logits = torch.matmul(cls_weight, cls_weight.T)
    #classwise_logit_dir = classwise_virtual_logits / classwise_virtual_logits.norm(dim=-1)

    # simple pseudo-labeling
    prob_all, indices_all = input.softmax(-1).max(-1)
    virtual_logits = classwise_virtual_logits[indices_all]
    conf_indices = prob_all>=thr
    loss = 1 - F.cosine_similarity(input, virtual_logits, dim=-1)
    loss = (prob_all.detach() - thr).exp() * loss
    loss = loss[conf_indices]

    # option = 0
    # if option == 0:
    #     loss = prob_all.detach().exp() * loss
    # else:
    #     loss = (prob_all.detach()-thr).exp() * loss
    # loss = loss[conf_indices]

    return loss

def _weighted_lcs_cons(input, cls_weight, thr=0., temperature=0.07):
    # 计算类别原型向量间的相似度矩阵
    classwise_virtual_logits = torch.matmul(cls_weight, cls_weight.T)
    
    # 获取预测概率和类别索引
    prob_all, indices_all = input.softmax(-1).max(-1)
    
    # 获取正样本（自身类别）
    pos_logits = classwise_virtual_logits[indices_all]
    
    # 创建负样本掩码（排除自身类别）
    batch_size = input.size(0)
    num_classes = cls_weight.size(0)
    mask = torch.ones(batch_size, num_classes, dtype=torch.bool, device=input.device)
    mask.scatter_(1, indices_all.unsqueeze(1), False)
    
    # 获取负样本（其他类别）
    neg_logits = classwise_virtual_logits.unsqueeze(0).expand(batch_size, -1, -1)
    neg_logits = neg_logits[mask].view(batch_size, num_classes-1, -1)
    
    # 计算正样本相似度
    pos_sim = F.cosine_similarity(input.unsqueeze(1), pos_logits.unsqueeze(1), dim=-1).squeeze(1)
    
    # 计算负样本相似度
    neg_sim = F.cosine_similarity(input.unsqueeze(1), neg_logits, dim=-1)
    
    # 计算对比损失
    numerator = torch.exp(pos_sim / temperature)
    denominator = numerator + torch.sum(torch.exp(neg_sim / temperature), dim=1)
    loss = -torch.log(numerator / denominator)
    
    # 应用置信度加权
    loss = (prob_all.detach() - thr).exp() * loss
    
    # 应用置信度阈值过滤
    conf_indices = prob_all >= thr
    loss = loss[conf_indices]
    
    return loss


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    return kl_div 


def TSD_loss(logits, features, prototypes, ratio, temperature=2):
    
    features = F.normalize(features, dim=1)
    batch_size = logits.size(0)
    
    # 1. 计算每个样本的能量和熵
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    entropy = _entropy_samples(logits)
    
    # 2. 计算样本权重: S_i * log(1 + exp(E_i))
    weights = entropy * torch.log(1 + torch.exp(energy))
    
    # 3. 应用权重阈值筛选样本
    k = max(1, int(batch_size * ratio))  # 至少选择1个样本
    _, conf_indices = torch.topk(weights, k, largest=False, sorted=True)
    
    # 4. 获取高权重样本的logits和伪标签
    conf_logits = logits[conf_indices]
    conf_features = features[conf_indices]
    
    # 5. calcuate the pesudo label based on prototypes
    dist = conf_features @ prototypes.T / temperature # [batch_size, num_classes] distance matrix

    # calcuate the distillation loss
    loss = softmax_kl_loss(conf_logits.detach(), dist).sum(1).mean(0)

    return loss


def Weighted_TSD_loss(logits, features, prototypes, ratio, temperature=2):
    
    features = F.normalize(features, dim=1)
    batch_size = logits.size(0)
    
    # 1. 计算每个样本的能量和熵
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    entropy = _entropy_samples(logits)
    
    # 2. 计算样本权重: S_i * log(1 + exp(E_i))
    weights = entropy * torch.log(1 + torch.exp(energy))
    
    # 3. 应用权重阈值筛选样本
    k = max(1, int(batch_size * ratio))  # 至少选择1个样本
    _, conf_indices = torch.topk(weights, k, largest=False, sorted=True)
    
    # 4. 获取高权重样本的logits和伪标签
    conf_logits = logits[conf_indices]
    conf_features = features[conf_indices]
    
    # 5. calcuate the pesudo label based on prototypes
    dist = conf_features @ prototypes.T / temperature # [batch_size, num_classes] distance matrix

    # 6. calcuate the distillation loss
    sample_losses = softmax_kl_loss(conf_logits.detach(), dist)  # 返回形状应为[k]或[k, num_classes]
    
    conf_weights = _entropy_samples(dist)
    # 7. weighted loss
    if sample_losses.dim() > 1:
        sample_losses = sample_losses.sum(1)
    
    # weighted loss
    weighted_loss = (conf_weights * sample_losses).sum() / (conf_weights.sum() + 1e-8)
    
    return weighted_loss


class MarginalEntropy(torch.nn.Module):
    def forward(self, logits):
        return _marginal_entropy(logits)


class Entropy(torch.nn.Module):
    def forward(self, logits):
        return _entropy(logits)


class NegMutualInformation(torch.nn.Module):
    def forward(self, logits):
        return _neg_mutual_information(logits)


class NegWeightedMutualInformation(torch.nn.Module):
    def __init__(self, lambda_info):
        super().__init__()
        self.lambda_info = lambda_info

    def forward(self, logits):
        return _neg_weighted_mutual_information(logits, self.lambda_info)


class NegWeightedMutualInformation_on_marginal(torch.nn.Module):
    def __init__(self, lambda_info):
        super().__init__()
        self.lambda_info = lambda_info

    def forward(self, logits):
        return _neg_weighted_mutual_information_on_marginal(logits, self.lambda_info)
    

class SelectiveSoftplusEnergyAlignment(nn.Module):
    def __init__(self, ratio=0.5, temp=1.0):
        super().__init__()
        self.ratio = ratio    # Proportion of low-energy samples to select
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation

    def forward(self, logits):
        """
        Args:
            logits: Model output tensor with shape [batch_size, num_classes]
        Returns:
            Alignment loss scalar (retains gradient for backpropagation)
        """
        # 1. Compute energy scores: lower values indicate higher prediction confidence
        energy = -self.temp * torch.logsumexp(logits / self.temp, dim=1)  # [batch_size]
        
        # 2. Sort energies (detached to prevent gradient flow through sorting)
        sorted_energy, _ = energy.detach().sort()
        
        # 3. Select top-k low-energy samples as source domain proxy
        num_chunks = int(1 / self.ratio)
        low_energy_chunk = torch.chunk(sorted_energy, num_chunks)[0]  # First ratio% samples
        src_energy_approx = low_energy_chunk.mean()  # Reference energy level
        
        # 4. Compute deviation from reference and apply Softplus
        diff = energy - src_energy_approx  # Gradient-preserving difference
        loss = self.softplus(diff).mean()  # Aggregate batch loss
        
        return loss
    
class MemorySoftplusEnergyAlignment(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, temp=1.0):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2


    def forward(self, logits, preds_of_pre_source_data):
        """
        Args:
            logits: Model output tensor with shape [batch_size, num_classes]
        Returns:
            Alignment loss scalar (retains gradient for backpropagation)
        """
        # Compute energy scores: lower values indicate higher prediction confidence
        energy = -self.temp * torch.logsumexp(logits / self.temp, dim=1)  # [batch_size]
        # the presudo source energy
        energy_preds_of_pre_source_data = -self.temp * torch.logsumexp(preds_of_pre_source_data / self.temp, dim=1)  # [batch_size]
        src_energy_approx = energy_preds_of_pre_source_data.detach()  # Reference energy level
        
        # Compute deviation from reference and apply Softplus
        diff = energy - src_energy_approx.mean() # Gradient-preserving difference
        loss = self.softplus(diff).mean()  # Aggregate batch loss
        
        # with the entropy loss
        loss_sum = self.lambda_1 * _entropy(logits) + self.lambda_2 * loss
        
        return  loss_sum
    
class MemorySoftplusEnergyWeightedAlignment(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, temp=1.0, epsilon=1e-8):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.epsilon = epsilon

    def forward(self, logits, preds_of_pre_source_data):
        """
        Args:
            logits: Model output tensor with shape [batch_size, num_classes]
        Returns:
            Alignment loss scalar (retains gradient for backpropagation)
        """
        # Compute energy scores: lower values indicate higher prediction confidence
        energy = -self.temp * torch.logsumexp(logits / self.temp, dim=1)  # [batch_size]
        # the presudo source energy
        energy_preds_of_pre_source_data = -self.temp * torch.logsumexp(preds_of_pre_source_data / self.temp, dim=1)  # [batch_size]
        src_energy_approx = energy_preds_of_pre_source_data.detach().mean()  # Reference energy level
        
        # Compute deviation from reference and apply Softplus
        diff = energy - src_energy_approx # Gradient-preserving difference

        # calculate the entropy
        probs = logits.softmax(1)
        entropy = -probs * torch.log(probs + 1e-6)
        entropy = entropy.sum(1)  # [batch_size]

        # normalize the weight
        weights = 1 / (torch.abs(diff.detach()) + self.epsilon)  # [batch_size]
        weights = weights * len(weights) / weights.sum()  # [batch_size]

        # weighted sum of loss
        weighted_entropy = weights * entropy  # [batch_size]
        entropy_loss = weighted_entropy.mean()

        loss = entropy_loss
        
        return  loss

class MemorySoftplusEnergyWeightedAlignmentMDR(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, temp=1.0, epsilon=1e-8):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.epsilon = epsilon

    def forward(self, logits, preds_of_pre_source_data):
        """
        Args:
            logits: Model output tensor with shape [batch_size, num_classes]
        Returns:
            Alignment loss scalar (retains gradient for backpropagation)
        """
        # Compute energy scores: lower values indicate higher prediction confidence
        energy = -self.temp * torch.logsumexp(logits / self.temp, dim=1)  # [batch_size]
        # the presudo source energy
        energy_preds_of_pre_source_data = -self.temp * torch.logsumexp(preds_of_pre_source_data / self.temp, dim=1)  # [batch_size]
        src_energy_approx = energy_preds_of_pre_source_data.detach().mean()  # Reference energy level
        
        # Compute deviation from reference and apply Softplus
        diff = energy - src_energy_approx # Gradient-preserving difference

        # calculate the entropy
        probs = logits.softmax(1)
        entropy = -probs * torch.log(probs + 1e-6)
        entropy = entropy.sum(1)  # [batch_size]

        # normalize the weight
        weights = 1 / (torch.abs(diff.detach()) + self.epsilon)  # [batch_size]
        weights = weights * len(weights) / weights.sum()  # [batch_size]

        # weighted sum of loss
        weighted_entropy = weights * entropy  # [batch_size]
        entropy_loss = weighted_entropy.mean()

        loss = self.lambda_1 * entropy_loss + self.lambda_2 * _mdr(logits)
        
        return  loss

class MemorySoftplusEnergyFeatureWeightedAlignment(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, temp=1.0, epsilon=1e-8):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1  # Future use weight parameter
        self.lambda_2 = lambda_2  # Future use weight parameter
        self.epsilon = epsilon   # Numerical stability constant

    def forward(self, logits, features, class_centers, missing_classes_flag):
        """
        Args:
            logits: Model predictions [batch_size, num_classes]
            features: Input data features [batch_size, feature_dim]
            class_centers: Class prototype features [num_classes, feature_dim]
        
        Returns:
            Weighted alignment loss scalar (with preserved gradients)
        """
        if missing_classes_flag:
            return torch.tensor(0.0, requires_grad=True)

        # Compute scaled probabilities with temperature, the entropy and presudo labels
        scaled_probs = F.softmax(logits / self.temp, dim=1)
        entropy = -torch.sum(scaled_probs * torch.log(scaled_probs + self.epsilon), dim=1)
        pseudo_labels = torch.argmax(scaled_probs, dim=1)  # [batch_size]
        
        # Retrieve corresponding class centers for each sample
        class_centers = class_centers[pseudo_labels]  # [batch_size, feature_dim]
        
        # Compute cosine similarity between features and class centers
        # Normalize vectors to unit length for cosine calculation
        normed_features = F.normalize(features, p=2, dim=1)
        normed_centers = F.normalize(class_centers, p=2, dim=1)
        # Compute cosine similarities [batch_size]
        cos_similarities = torch.sum(normed_features * normed_centers, dim=1)
        # Map similarity values from [-1, 1] to [0, 1] for weighting
        alignment_weights = (cos_similarities + 1) / 2
        
        # Apply softplus to entropy and weight by alignment confidence
        weighted_entropy = self.softplus(entropy) * alignment_weights
        
        # Compute final loss as mean across batch
        loss = torch.mean(weighted_entropy)
        
        return loss
class MemorySoftplusEnergyThresholdWeightedAlignment(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, temp=1.0, epsilon=1e-8, threshold=0.5):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.epsilon = epsilon
        self.threshold = threshold

    def forward(self, logits, preds_of_pre_source_data):
        """
        Args:
            logits: Model output tensor with shape [batch_size, num_classes]
            preds_of_pre_source_data: Predictions from memory model [batch_size, num_classes]
        Returns:
            Alignment loss scalar (retains gradient for backpropagation)
        """
        # Compute energy scores: lower values indicate higher prediction confidence
        energy = -self.temp * torch.logsumexp(logits / self.temp, dim=1)  # [batch_size]
        energy_preds_of_pre_source_data = -self.temp * torch.logsumexp(preds_of_pre_source_data / self.temp, dim=1)  # [batch_size]
        src_energy_approx = energy_preds_of_pre_source_data.detach().mean()  # Reference energy level
        
        # Compute entropy to measure prediction uncertainty
        probs = logits.softmax(1)
        entropy = -probs * torch.log(probs + 1e-6)
        entropy = entropy.sum(1)  # [batch_size]
        
        # Create mask for low-entropy samples (entropy < threshold)
        low_entropy_mask = entropy < self.threshold
        
        # If no samples meet criteria, return zero loss with preserved gradients
        if not torch.any(low_entropy_mask):
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Compute energy difference only for low-entropy samples
        low_entropy_energy = energy[low_entropy_mask]
        diff = low_entropy_energy - src_energy_approx  # [low_entropy_batch_size]
        
        # Calculate weights based on inverse magnitude of energy difference
        # Detach weights to prevent second-order gradients
        weights = 1 / (torch.abs(diff.detach()) + self.epsilon)  # [low_entropy_batch_size]
        # Normalize weights to maintain scale consistency
        weights = weights * len(weights) / weights.sum()  # [low_entropy_batch_size]

        # Extract entropy values for low-entropy samples
        low_entropy_entropy = entropy[low_entropy_mask]  # [low_entropy_batch_size]
        
        # Apply calculated weights to entropy values
        weighted_entropy = weights * low_entropy_entropy
        # Compute mean loss across selected samples
        entropy_loss = weighted_entropy.mean()

        return entropy_loss

class MemorySoftplusEnergyRatioSortedAlignment(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, temp=1.0, epsilon=1e-8, ratio=0.5):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.epsilon = epsilon
        self.ratio = ratio    # Ratio of low-entropy samples to select

    def forward(self, logits, preds_of_pre_source_data):
        """
        Args:
            logits: Model output tensor [batch_size, num_classes]
            preds_of_pre_source_data: Memory predictions [batch_size, num_classes]
        Returns:
            Alignment loss scalar
        """
        # Compute energy scores
        energy = -self.temp * torch.logsumexp(logits / self.temp, dim=1)
        energy_preds = -self.temp * torch.logsumexp(preds_of_pre_source_data / self.temp, dim=1)
        src_energy_approx = energy_preds.detach().mean()  # Reference energy
        
        # Compute entropy per sample
        probs = logits.softmax(1)
        entropy = -(probs * torch.log(probs + 1e-6)).sum(1)  # [batch_size]
        batch_size = entropy.size(0)
        
        # Determine number of samples to select based on ratio
        k = max(1, int(batch_size * self.ratio))  # At least 1 sample
        if k >= batch_size:
            selected_indices = torch.arange(batch_size)
        else:
            # Select indices with smallest entropy
            _, selected_indices = torch.topk(entropy, k=k, largest=False)
        
        # Compute energy difference for selected samples
        selected_energy = energy[selected_indices]
        diff = selected_energy - src_energy_approx
        
        # Calculate adaptive weights
        weights = 1 / (torch.abs(diff.detach()) + self.epsilon)
        weights = weights * k / weights.sum()  # Normalize weights
        
        # Apply weights to selected entropy values
        selected_entropy = entropy[selected_indices]
        weighted_entropy = weights * selected_entropy
        
        return weighted_entropy.mean()

class PresudoLabelMemorySoftplusEnergyAlignment(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

    def forward(self, logits, preds_of_pre_source_data):
        """
        Args:
            logits: Model output tensor with shape [batch_size, num_classes]
        Returns:
            Alignment loss scalar (retains gradient for backpropagation)
        """
        # Compute energy scores: lower values indicate higher prediction confidence
        energy = -self.temp * torch.logsumexp(logits / self.temp, dim=1)  # [batch_size]
        # the presudo source energy
        energy_preds_of_pre_source_data = -self.temp * torch.logsumexp(preds_of_pre_source_data / self.temp, dim=1)  # [batch_size]
        src_energy_approx = energy_preds_of_pre_source_data.detach()  # Reference energy level
        
        # Compute deviation from reference and apply Softplus
        diff = energy - src_energy_approx.mean() # Gradient-preserving difference
        loss = self.softplus(diff).mean()  # Aggregate batch loss
        
        # with the entropy loss
        loss_sum = self.lambda_1 * _entropy(logits) + self.lambda_2 * loss + self.lambda_3 * _mdr(logits)
        
        return  loss_sum
    

    
class CE_MDR(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, temp=1.0):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2


    def forward(self, logits):
        
        loss_sum = self.lambda_1 * _entropy(logits) + self.lambda_2 * _mdr(logits)
        
        return  loss_sum

class CE_KL(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, temp=1.0):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2


    def forward(self, logits):
        
        loss_sum = self.lambda_1 * _entropy(logits) + self.lambda_2 * _kl_loss(logits)
        
        return  loss_sum

class CaliE_MDR(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, temp=1.0):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2


    def forward(self, logits):
        
        loss_sum = self.lambda_1 * _calibrated_entropy(logits,gamma=5) + self.lambda_2 * _mdr(logits)
        
        return  loss_sum
    
class CaliE_KL(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, temp=1.0):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2


    def forward(self, logits):
        
        loss_sum = self.lambda_1 * _calibrated_entropy(logits,gamma=5) + self.lambda_2 * _kl_loss(logits)
        
        return  loss_sum
class CaliE_UKL(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, temp=1.0):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2


    def forward(self, logits):
        
        loss_sum = self.lambda_1 * _calibrated_entropy(logits,gamma=5) + self.lambda_2 * uniform_kl_loss(logits)
        
        return  loss_sum
    
class CaliE_MDR_lcs(nn.Module):
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits, cls_weight):
        
        loss_sum = self.lambda_1 * _calibrated_entropy(logits,gamma=5) + self.lambda_2 * _mdr(logits) + self.lambda_3 * _weighted_lcs(logits, cls_weight, thr=0.4).mean(0)
        
        return  loss_sum
    
class CaliE_MDR_lcs_cons(nn.Module):
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits, cls_weight):
        
        loss_sum = self.lambda_1 * _entropy(logits) + self.lambda_2 * _mdr(logits) + self.lambda_3 * _weighted_lcs_cons(logits, cls_weight, thr=0.4).mean(0)
        
        return  loss_sum
    
class CaliE_MDR_lcs_selection(nn.Module):
    def __init__(self, thr=0.4, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.thr = thr  # Original selection ratio parameter (retained but not used for max_prob selection)

    def forward(self, logits, cls_weight):
        """
        Compute loss only for samples where max softmax probability > self.thr
        
        Args:
            logits: Unnormalized model predictions [batch_size, num_classes]
            cls_weight: Class weights (if applicable) [batch_size] or [batch_size, ...]
            
        Returns:
            Weighted combination of losses for selected samples
        """
        # Compute softmax probabilities along class dimension
        probs = F.softmax(logits, dim=-1)
        
        # Get maximum prediction probability for each sample
        max_probs, _ = torch.max(probs, dim=-1)
        
        # Create boolean mask: True where max probability > self.thr
        selection_mask = max_probs > self.thr
        
        # Handle case with no qualifying samples
        if not torch.any(selection_mask):
            # Return scalar 0 tensor to maintain gradient flow
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Apply selection mask to logits
        selected_logits = logits[selection_mask]
        
        # Compute loss components using ONLY selected samples
        ce_loss = _calibrated_entropy(selected_logits, gamma=5)
        mdr_loss = _mdr(selected_logits)
        wlcs_loss = _weighted_lcs(selected_logits, cls_weight, thr=self.thr).mean(0)
        
        # Combine losses with weighting coefficients
        loss_sum = (
            self.lambda_1 * ce_loss + 
            self.lambda_2 * mdr_loss + 
            self.lambda_3 * wlcs_loss
        )
        
        return loss_sum

class CaliE_MDR_lcs_ConsSamples(nn.Module):
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits, cls_weight):
        
        cons_loss = contrastive_loss_samples(logits, thr=0.4, temperature=self.temp)

        loss_sum = self.lambda_1 * _calibrated_entropy(logits,gamma=5) + self.lambda_2 * _mdr(logits) + self.lambda_3 * cons_loss
        
        return  loss_sum

class CE_KL_lcs_ConsSamples(nn.Module):
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits, cls_weight):
        
        cons_loss = contrastive_loss_samples(logits, thr=0.0, temperature=self.temp)

        loss_sum = self.lambda_1 * _entropy(logits) + self.lambda_2 * _kl_loss(logits) + self.lambda_3 * cons_loss
        
        return  loss_sum

class CE_KL_lcs_ConsSamples_selection(nn.Module):
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits, cls_weight):
        
        cons_loss = contrastive_loss_samples_selection(logits, ratio=self.ratio, temperature=self.temp)

        loss_sum = self.lambda_1 * _entropy(logits) + self.lambda_2 * _kl_loss(logits) + self.lambda_3 * cons_loss
        
        return  loss_sum


class CaliE_MDR_lcs_ConsSamplesFea(nn.Module):
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits, feature):
        
        cons_loss = contrastive_loss_samplefeatures(logits, feature, thr=0.4, temperature=self.temp)

        loss_sum = self.lambda_1 * _calibrated_entropy(logits,gamma=5) + self.lambda_2 * _mdr(logits) + self.lambda_3 * cons_loss
        
        return  loss_sum


class CE_KL_ConsSamplesFea(nn.Module):
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits, feature):

        cons_loss = contrastive_loss_samplefeatures(logits, feature, thr=0.4, temperature=self.temp)
        
        loss_sum = self.lambda_1 * _entropy(logits) + self.lambda_2 * _kl_loss(logits) + self.lambda_3 * cons_loss
        
        return  loss_sum


class EnergyEntropy_selected(nn.Module):
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy+mdr

    def forward(self, logits):
        # 1. Compute energy and entropy for all samples
        energy = -self.temp * torch.logsumexp(logits / self.temp, dim=1)
        entropy = _entropy_samples(logits)
        
        # 2. Compute sample weights: S_i * log(1 + exp(E_i))
        weights = entropy * torch.log(1 + torch.exp(energy))
        
        # 3. Sort and select top ratio samples with smallest weights
        k = int(self.ratio * logits.size(0))
        _, indices = torch.topk(weights, k, largest=False)
        selected_logits = logits[indices]
        
        # 4. Compute losses for selected samples
        entropy_loss = _entropy_samples(selected_logits).mean()  # Entropy of selected samples
        mdr_loss = _mdr(selected_logits)  # MDR loss on selected samples
        
        # 5. Compute energy loss for remaining samples
        remaining_indices = torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)
        remaining_indices[indices] = False
        remaining_energy = energy[remaining_indices]
        energy_loss = self.softplus(remaining_energy).mean()
        
        # 6. Combine all losses
        loss_sum = (
            self.lambda_1 * entropy_loss + 
            self.lambda_2 * energy_loss + 
            self.lambda_3 * mdr_loss
        )
        
        return loss_sum
    
class EnergyEntropy_selected_all(nn.Module):
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy+mdr

    def forward(self, logits):
        # 1. Compute energy and entropy for all samples
        energy = -self.temp * torch.logsumexp(logits / self.temp, dim=1)
        entropy = _entropy_samples(logits)
        
        # 2. Compute sample weights: S_i * log(1 + exp(E_i))
        weights = entropy * torch.log(1 + torch.exp(energy))
        
        # 3. Sort and select top ratio samples with smallest weights
        k = int(self.ratio * logits.size(0))
        _, indices = torch.topk(weights, k, largest=False)
        selected_logits = logits[indices]
        
        # 4. Compute losses for selected samples
        entropy_loss = _entropy_samples(selected_logits).mean()  # Entropy of selected samples
        mdr_loss = _mdr(selected_logits)  # MDR loss on selected samples
        
        # 5. Compute energy loss for all samples
        energy_loss = self.softplus(energy).mean()
        
        # 6. Combine all losses
        loss_sum = (
            self.lambda_1 * entropy_loss + 
            self.lambda_2 * energy_loss + 
            self.lambda_3 * mdr_loss
        )
        
        return loss_sum
    

class EnergyEntropy_selected_energy(nn.Module):
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy+mdr

    def forward(self, logits):
        # 1. Compute energy and entropy for all samples
        energy = -self.temp * torch.logsumexp(logits / self.temp, dim=1)
        entropy = _entropy_samples(logits)
        
        # 2. Compute sample weights: S_i * log(1 + exp(E_i))
        weights = entropy * torch.log(1 + torch.exp(energy))
        
        # 3. Sort and select top ratio samples with smallest weights
        k = int(self.ratio * logits.size(0))
        _, indices = torch.topk(weights, k, largest=False)
        selected_logits = logits[indices]
        
        # 4. Compute losses for selected samples
        entropy_loss = _entropy_samples(selected_logits).mean()  # Entropy of selected samples
        mdr_loss = _mdr(selected_logits)  # MDR loss on selected samples
        energy_loss = self.softplus(_energy_samples(selected_logits)).mean()
        
        # 6. Combine all losses
        loss_sum = (
            self.lambda_1 * entropy_loss + 
            self.lambda_2 * energy_loss + 
            self.lambda_3 * mdr_loss
        )
        
        return loss_sum
    

class EnergyEntropy_selected_align(nn.Module):
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy+mdr

    def forward(self, logits):
        # 1. Compute energy and entropy for all samples
        energy = -self.temp * torch.logsumexp(logits / self.temp, dim=1)
        entropy = _entropy_samples(logits)
        # entropy_ = _entropy(logits)
        
        # 2. Compute sample weights: S_i * log(1 + exp(E_i))
        weights = entropy * torch.log(1 + torch.exp(energy))
        
        # 3. Sort and select top ratio samples with smallest weights
        k = int(self.ratio * logits.size(0))
        _, indices = torch.topk(weights, k, largest=False)
        selected_logits = logits[indices]
        selected_energy = energy[indices]
        
        # 4. Compute losses for selected samples
        entropy_loss = _entropy_samples(selected_logits).mean()  # Entropy of selected samples
        mdr_loss = _mdr(selected_logits)  # MDR loss on selected samples
        
        # 5. Compute energy loss for left samples
        mask = torch.ones_like(energy, dtype=torch.bool)
        mask[indices] = False
        remaining_energy = energy[mask]
        energy_diff = selected_energy.detach().mean() - remaining_energy.mean()
        energy_align_loss = self.softplus(energy_diff)

        
        # 6. Combine all losses
        loss_sum = (
            self.lambda_1 * entropy_loss + 
            self.lambda_2 * energy_align_loss + 
            self.lambda_3 * mdr_loss
        )
        
        return loss_sum
    

class PresudoLabelEMA(nn.Module):
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy+mdr

    def forward(self, logits, logits_ema):

        presudolabel_loss = softmax_entropy(logits, logits_ema).mean(0)
        
        loss_sum = self.lambda_1 * presudolabel_loss + self.lambda_2 * _mdr(logits)

        return loss_sum


class PresudoLabelEMA_selection(nn.Module):
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, threshold=0.5):
        super().__init__()
        self.temp = temp
        self.threshold = threshold  # 置信度阈值
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio

    def forward(self, logits, logits_ema):
        
        # se_loss = F.cross_entropy(logits, pseudo_labels, reduction='none')
        se_loss = softmax_entropy(logits, logits_ema)
        
        prob_ema = logits_ema.softmax(1)
        conf_mask = (prob_ema.max(dim=1)[0] > self.threshold)  # 高置信度样本掩码
        se_loss = se_loss[conf_mask].mean() if conf_mask.any() else 0.0
        mdr_loss = _mdr(logits[conf_mask]) if conf_mask.any() else 0.0

        loss_sum = self.lambda_1 * se_loss + self.lambda_2 * mdr_loss
        return loss_sum

class PresudoLabelEMA_symmetric(nn.Module):
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp          # Temperature parameter for probability sharpening
        self.softplus = nn.Softplus()  # Softplus activation (currently unused)
        self.lambda_1 = lambda_1  # Weight for symmetric entropy loss
        self.lambda_2 = lambda_2  # Weight for MDR regularization
        self.lambda_3 = lambda_3  # Reserved weight parameter
        self.ratio = ratio        # Ratio of samples to select for entropy + MDR

    def forward(self, logits, logits_ema):
        # Compute probability distributions with temperature scaling
        p_student = F.softmax(logits / self.temp, dim=1)
        p_ema = F.softmax(logits_ema / self.temp, dim=1)
        
        # Calculate symmetric entropy loss
        symmetric_loss = symmetric_entropy_loss(p_student, p_ema)
        
        # Combine losses with weighting coefficients
        loss_sum = self.lambda_1 * symmetric_loss + self.lambda_2 * _mdr(logits)
        return loss_sum

class PresudoLabelEMA_energy(nn.Module):
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy+mdr

    def forward(self, logits, logits_ema):

        # 1. Compute energy and entropy for all samples
        energy = -self.temp * torch.logsumexp(logits / self.temp, dim=1)
        entropy = _entropy_samples(logits)    
        
        # 2. Compute sample weights: S_i * log(1 + exp(E_i))
        weights = entropy * torch.log(1 + torch.exp(energy))
        
        # 3. Sort and select top ratio samples with smallest weights
        k = int(self.ratio * logits.size(0))
        _, indices = torch.topk(weights, k, largest=False)
        
        # 4. calcuate the loss for the selected samples
        presudolabel_loss = softmax_entropy(logits[indices], logits_ema[indices]).mean()
        
        loss_sum = self.lambda_1 * presudolabel_loss + self.lambda_2 * _energy_samples(logits[indices]).mean()

        return loss_sum

"""class PresudoLabelEMA_energy(nn.Module):
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy+mdr

    def forward(self, logits, logits_ema):

        # 1. Compute energy and entropy for all samples
        energy = -self.temp * torch.logsumexp(logits / self.temp, dim=1)
        entropy = _entropy_samples(logits)    
        
        # 2. Compute sample weights: S_i * log(1 + exp(E_i))
        weights = entropy * torch.log(1 + torch.exp(energy))
        
        # 3. Sort and select top ratio samples with smallest weights
        k = int(self.ratio * logits.size(0))
        _, indices = torch.topk(weights, k, largest=False)
        
        # 4. calcuate the loss for the selected samples
        presudolabel_loss = softmax_entropy(logits[indices], logits_ema[indices]).mean()
        
        loss_sum = self.lambda_1 * presudolabel_loss + self.lambda_2 * _mdr(logits[indices])

        return loss_sum"""
    
class PresudoLabelEMA_lcs(nn.Module):
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits, logits_ema, cls_weight):

        presudolabel_loss = softmax_entropy(logits, logits_ema).mean(0)
        
        loss_sum = self.lambda_1 * presudolabel_loss + self.lambda_2 * _mdr(logits) + self.lambda_3 * _weighted_lcs(logits, cls_weight, thr=0.4).mean(0)

        return loss_sum

class PresudoLabelEMA_SampleCons(nn.Module):
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits, logits_ema):

        presudolabel_loss = softmax_entropy(logits, logits_ema).mean(0)
        
        loss_sum = self.lambda_1 * presudolabel_loss + self.lambda_2 * _mdr(logits) + self.lambda_3 * contrastive_loss_samples(logits, thr=0.4, temperature=self.temp)

        return loss_sum

class EntropyMDREMA_lcs(nn.Module):
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits, logits_ema, cls_weight):
        
        loss_sum = self.lambda_1 * _calibrated_entropy(logits,gamma=5) + self.lambda_2 * _mdr(logits) + self.lambda_3 * _weighted_lcs(logits, cls_weight, thr=0.4).mean(0)

        return loss_sum

class ConsSamples(nn.Module):
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits):
        
        cons_loss = contrastive_loss_samples(logits, thr=0.66, temperature=self.temp)
        
        loss_sum = self.lambda_1 * cons_loss
        
        return  loss_sum

class ConsSamples_lcs(nn.Module):
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits, cls_weight):
        
        cons_loss = contrastive_loss_samples(logits, thr=0.4, temperature=self.temp)
        lcs_loss = _weighted_lcs(logits, cls_weight, thr=0.4).mean(0)

        loss_sum = self.lambda_1 * cons_loss + self.lambda_2 * lcs_loss
        
        return  loss_sum

class ConsSamples_weighted(nn.Module):
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits):
        
        cons_loss = contrastive_loss_samples_weighted(logits, thr=0.4, temperature=self.temp)

        loss_sum = self.lambda_1 * cons_loss
        
        return  loss_sum

class ConsSamples_selection(nn.Module):
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits):
        
        cons_loss = contrastive_loss_samples_selection(logits, ratio=self.ratio, temperature=self.temp)

        loss_sum = self.lambda_1 * cons_loss
        
        return  loss_sum

class ConsSamples_selection_two_stage(nn.Module):
    # special version for two stage model updating
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits):
        
        cons_loss = contrastive_loss_samples_selection(logits, ratio=self.ratio, temperature=self.temp)

        loss_sum = self.lambda_3 * cons_loss
        
        return  loss_sum

class ConsSamples_selection_1(nn.Module):
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits):
        
        cons_loss = contrastive_loss_samples_selection_1(logits, ratio=self.ratio, temperature=self.temp)

        loss_sum = self.lambda_1 * cons_loss
        
        return  loss_sum
    
class ConsSamples_selection_2(nn.Module):
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits):
        
        cons_loss = contrastive_loss_samples_selection_2(logits, ratio=self.ratio, temperature=self.temp)

        loss_sum = self.lambda_1 * cons_loss
        
        return  loss_sum
    

class ConsSamples_selection_dropout(nn.Module):
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits, logits_dropout):
        
        cons_loss = contrastive_loss_samples_selection_dropout(logits, logits_dropout, ratio=self.ratio, temperature=self.temp)

        loss_sum = self.lambda_1 * cons_loss
        
        return  loss_sum
    


class ConsSamples_selection_distillation(nn.Module):
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits, features, prototypes):
        
        cons_loss = contrastive_loss_samples_selection(logits, ratio=self.ratio, temperature=self.temp)
        dis_loss = TSD_loss(logits, features, prototypes, self.ratio, temperature=self.temp)


        loss_sum = self.lambda_1 * cons_loss + self.lambda_2 * dis_loss
        
        return  loss_sum


class Weighted_ConsSamples_selection_distillation(nn.Module):
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs

    def forward(self, logits, features, prototypes):
        
        cons_loss = contrastive_loss_samples_selection(logits, ratio=self.ratio, temperature=self.temp)
        # en_loss = _neg_weighted_mutual_information_on_marginal(logits, 1.0)
        dis_loss = Weighted_TSD_loss(logits, features, prototypes, self.ratio, temperature=self.temp)


        loss_sum = self.lambda_1 * cons_loss + self.lambda_2 * dis_loss
        
        return  loss_sum