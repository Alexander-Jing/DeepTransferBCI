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

def _entropy_samples_normalized(logits):
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


def contrastive_loss_samples_selection_modified(logits, ratio=0.75, temperature=0.07, weight_type='entropy_energy'):
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
    if weight_type == 'entropy_energy':
        weights = entropy * torch.log(1 + torch.exp(energy))
    elif weight_type == 'entropy':
        weights = entropy
    elif weight_type == 'energy':
        weights = torch.log(1 + torch.exp(energy))
    else:
        raise ValueError(f"Unsupported weight_type: {weight_type}")
    
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
        
        # 计算分子：正样本的指数相似度
        numerator = torch.exp(sim_matrix[i, pos_indices] / temperature)
        
        # 获取当前样本的负样本索引
        neg_indices = torch.where(neg_mask[i])[0]
        
        # 计算分母：负样本的指数相似度之和
        denominator = torch.sum(torch.exp(sim_matrix[i, neg_indices] / temperature))
        
        # 计算损失项：-sum(log(分子/分母))
        loss_term = -torch.sum(torch.log(numerator / (denominator + 1e-8)))  # 添加小量避免除零
        
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
    

def contrastive_loss_samples_selection_modified_feas(logits, feas, ratio=0.75, temperature=0.07, weight_type='entropy_energy'):
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
    if weight_type == 'entropy_energy':
        weights = entropy * torch.log(1 + torch.exp(energy))
    elif weight_type == 'entropy':
        weights = entropy
    elif weight_type == 'energy':
        weights = torch.log(1 + torch.exp(energy))
    else:
        raise ValueError(f"Unsupported weight_type: {weight_type}")
    
    # 3. 应用权重阈值筛选样本
    k = max(1, int(batch_size * ratio))  # 至少选择1个样本
    _, conf_indices = torch.topk(weights, k, largest=False, sorted=True)
    
    # 4. 获取高权重样本的logits和伪标签
    probs = torch.softmax(logits, dim=1)
    _, pseudo_labels = torch.max(probs, dim=1)
    conf_logits = logits[conf_indices]
    conf_feas = feas[conf_indices]
    conf_labels = pseudo_labels[conf_indices]
    k = conf_logits.size(0)  # 实际选中的样本数量
    
    # 5. 计算logits间的余弦相似度矩阵
    sim_matrix = F.cosine_similarity(
        conf_feas.unsqueeze(1),  # [k, 1, num_classes]
        conf_feas.unsqueeze(0),  # [1, k, num_classes]
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
        
        # 计算分子：正样本的指数相似度
        numerator = torch.exp(sim_matrix[i, pos_indices] / temperature)
        
        # 获取当前样本的负样本索引
        neg_indices = torch.where(neg_mask[i])[0]
        
        # 计算分母：负样本的指数相似度之和
        denominator = torch.sum(torch.exp(sim_matrix[i, neg_indices] / temperature))
        
        # 计算损失项：-sum(log(分子/分母))
        loss_term = -torch.sum(torch.log(numerator / (denominator + 1e-8)))  # 添加小量避免除零
        
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


def contrastive_loss_samples_selection_review_2_2(logits, review_data_logits, ratio=0.75, ratio_review=0.75, temperature=0.07, weight_type='entropy_energy'):
    """
    实现基于能量和熵的对比损失函数，加入 review_data_logits 样本
    参数:
        logits: 模型输出的logits [batch_size, num_classes]
        review_data_logits: 额外的样本logits [review_batch_size, num_classes]
        ratio: 用于筛选权重样本的阈值百分比（比例）
        ratio_review: 用于筛选review数据中每个类别的样本比例
        temperature: 对比损失温度参数
    """
    batch_size = logits.size(0)
    
    # 1. 计算每个样本的能量和熵
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    entropy = _entropy_samples(logits)
    
    # 2. 计算样本权重: S_i * log(1 + exp(E_i))
    if weight_type == 'entropy':
        weights = entropy
    elif weight_type == 'energy':
        weights = torch.log(1 + torch.exp(energy))
    elif weight_type == 'entropy_energy':
        weights = entropy * torch.log(1 + torch.exp(energy))
    else:
        raise ValueError(f"Unsupported weight_type: {weight_type}")

    # 3. 应用权重阈值筛选样本
    k = max(1, int(batch_size * ratio))  # 至少选择1个样本
    _, conf_indices = torch.topk(weights, k, largest=False, sorted=True)
    
    # 4. 获取高权重样本的logits和伪标签
    probs = torch.softmax(logits, dim=1)
    _, pseudo_labels = torch.max(probs, dim=1)
    conf_logits = logits[conf_indices]
    conf_labels = pseudo_labels[conf_indices]
    
    # 5. 对review_data_logits进行同样的权重计算和筛选
    review_energy = -temperature * torch.logsumexp(review_data_logits / temperature, dim=1)
    review_entropy = _entropy_samples(review_data_logits)
    if weight_type == 'entropy':
        review_weights = review_entropy
    elif weight_type == 'energy':
        review_weights = torch.log(1 + torch.exp(review_energy))
    elif weight_type == 'entropy_energy':
        review_weights = review_entropy * torch.log(1 + torch.exp(review_energy))
    else:
        raise ValueError(f"Unsupported weight_type: {weight_type}")
    
    review_probs = torch.softmax(review_data_logits, dim=1)
    _, review_pseudo_labels = torch.max(review_probs, dim=1)
    
    # 6. 对review数据按类别筛选：每个类别选择ratio_review比例的权重最小样本
    unique_classes = torch.unique(review_pseudo_labels)
    selected_review_logits = []
    selected_review_labels = []
    
    for cls in unique_classes:
        # 获取当前类别的样本索引
        cls_mask = (review_pseudo_labels == cls)
        cls_indices = torch.where(cls_mask)[0]
        
        if len(cls_indices) == 0:
            continue
            
        # 获取当前类别的权重
        cls_weights = review_weights[cls_indices]
        
        # 计算当前类别需要选择的样本数量
        k_review = max(1, int(len(cls_indices) * ratio_review))
        
        # 选择权重最小的k_review个样本（最确定的样本）
        _, cls_conf_indices = torch.topk(cls_weights, k_review, largest=False, sorted=True)
        
        # 获取选中的样本
        selected_cls_logits = review_data_logits[cls_indices[cls_conf_indices]]
        selected_cls_labels = review_pseudo_labels[cls_indices[cls_conf_indices]]
        
        selected_review_logits.append(selected_cls_logits)
        selected_review_labels.append(selected_cls_labels)
    
    # 7. 合并选中的review样本
    if selected_review_logits:
        selected_review_logits = torch.cat(selected_review_logits, dim=0)
        selected_review_labels = torch.cat(selected_review_labels, dim=0)
    else:
        selected_review_logits = torch.tensor([], device=logits.device)
        selected_review_labels = torch.tensor([], device=logits.device, dtype=torch.long)
    
    # 8. 将conf_logits和筛选后的review_data_logits合并
    extended_logits = torch.cat([conf_logits, selected_review_logits], dim=0)
    extended_labels = torch.cat([conf_labels, selected_review_labels], dim=0)
    extended_k = extended_logits.size(0)  # 扩展后的样本数量
    
    if extended_k == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    # 9. 计算扩展样本集的余弦相似度矩阵
    sim_matrix = F.cosine_similarity(
        extended_logits.unsqueeze(1),  # [extended_k, 1, num_classes]
        extended_logits.unsqueeze(0),  # [1, extended_k, num_classes]
        dim=-1
    )
    
    # 10. 创建正负样本掩码
    # 正样本：相同伪标签且非自身
    pos_mask = (extended_labels.unsqueeze(0) == extended_labels.unsqueeze(1)) & \
               (~torch.eye(extended_k, dtype=torch.bool, device=logits.device))
    
    # 负样本：不同伪标签
    neg_mask = extended_labels.unsqueeze(0) != extended_labels.unsqueeze(1)
    
    # 11. 计算每个样本的正样本数量
    pos_counts = pos_mask.sum(dim=1).float()  # [extended_k]
    valid_samples = pos_counts > 0  # 排除没有正样本的样本
    
    # 12. 初始化损失
    total_loss = torch.tensor(0.0, device=logits.device)
    valid_count = 0
    
    # 13. 遍历每个样本计算损失（遍历extended_logits的样本）
    for i in range(extended_logits.size(0)):  # 遍历extended_logits的样本
        if i >= extended_k or not valid_samples[i]:
            continue  # 跳过索引超出或没有正样本的样本
            
        # 获取当前样本的正样本索引
        pos_indices = torch.where(pos_mask[i])[0]
        
        # 计算分子：正样本的指数相似度
        numerator = torch.exp(sim_matrix[i, pos_indices] / temperature)
        
        # 获取当前样本的负样本索引
        neg_indices = torch.where(neg_mask[i])[0]
        
        # 计算分母：负样本的指数相似度之和
        denominator = torch.sum(torch.exp(sim_matrix[i, neg_indices] / temperature))
        
        # 计算损失项：-sum(log(分子/分母))
        loss_term = -torch.sum(torch.log(numerator / (denominator + 1e-8)))  # 添加小量避免除零
        
        # 除以正样本数量（|pos(i)|）
        loss_term /= pos_counts[i]
        
        # 累加损失
        total_loss += loss_term
        valid_count += 1
    
    # 14. 计算平均损失
    if valid_count > 0:
        return total_loss / valid_count
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)


def contrastive_loss_samples_selection_review_2_2_feas(logits, review_data_logits, feas, review_feas, ratio=0.75, ratio_review=0.75, temperature=0.07, weight_type='entropy_energy'):
    """
    实现基于能量和熵的对比损失函数，加入 review_data_logits 样本
    参数:
        logits: 模型输出的logits [batch_size, num_classes]
        review_data_logits: 额外的样本logits [review_batch_size, num_classes]
        ratio: 用于筛选权重样本的阈值百分比（比例）
        ratio_review: 用于筛选review数据中每个类别的样本比例
        temperature: 对比损失温度参数
    """
    batch_size = logits.size(0)
    
    # 1. 计算每个样本的能量和熵
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    entropy = _entropy_samples(logits)
    
    # 2. 计算样本权重: S_i * log(1 + exp(E_i))
    if weight_type == 'entropy':
        weights = entropy
    elif weight_type == 'energy':
        weights = torch.log(1 + torch.exp(energy))
    elif weight_type == 'entropy_energy':
        weights = entropy * torch.log(1 + torch.exp(energy))
    else:
        raise ValueError(f"Unsupported weight_type: {weight_type}")

    # 3. 应用权重阈值筛选样本
    k = max(1, int(batch_size * ratio))  # 至少选择1个样本
    _, conf_indices = torch.topk(weights, k, largest=False, sorted=True)
    
    # 4. 获取高权重样本的logits和伪标签
    probs = torch.softmax(logits, dim=1)
    _, pseudo_labels = torch.max(probs, dim=1)
    conf_logits = logits[conf_indices]
    conf_labels = pseudo_labels[conf_indices]
    conf_feas = feas[conf_indices]
    
    # 5. 对review_data_logits进行同样的权重计算和筛选
    review_energy = -temperature * torch.logsumexp(review_data_logits / temperature, dim=1)
    review_entropy = _entropy_samples(review_data_logits)
    if weight_type == 'entropy':
        review_weights = review_entropy
    elif weight_type == 'energy':
        review_weights = torch.log(1 + torch.exp(review_energy))
    elif weight_type == 'entropy_energy':
        review_weights = review_entropy * torch.log(1 + torch.exp(review_energy))
    else:
        raise ValueError(f"Unsupported weight_type: {weight_type}")
    
    review_probs = torch.softmax(review_data_logits, dim=1)
    _, review_pseudo_labels = torch.max(review_probs, dim=1)
    
    # 6. 对review数据按类别筛选：每个类别选择ratio_review比例的权重最小样本
    unique_classes = torch.unique(review_pseudo_labels)
    selected_review_logits = []
    selected_review_feas = []
    selected_review_labels = []
    
    for cls in unique_classes:
        # 获取当前类别的样本索引
        cls_mask = (review_pseudo_labels == cls)
        cls_indices = torch.where(cls_mask)[0]
        
        if len(cls_indices) == 0:
            continue
            
        # 获取当前类别的权重
        cls_weights = review_weights[cls_indices]
        
        # 计算当前类别需要选择的样本数量
        k_review = max(1, int(len(cls_indices) * ratio_review))
        
        # 选择权重最小的k_review个样本（最确定的样本）
        _, cls_conf_indices = torch.topk(cls_weights, k_review, largest=False, sorted=True)
        
        # 获取选中的样本
        selected_cls_logits = review_data_logits[cls_indices[cls_conf_indices]]
        selected_cls_feas = review_feas[cls_indices[cls_conf_indices]]
        selected_cls_labels = review_pseudo_labels[cls_indices[cls_conf_indices]]
        
        selected_review_logits.append(selected_cls_logits)
        selected_review_labels.append(selected_cls_labels)
        selected_review_feas.append(selected_cls_feas)
    
    # 7. 合并选中的review样本
    if selected_review_logits:
        selected_review_logits = torch.cat(selected_review_logits, dim=0)
        selected_review_feas = torch.cat(selected_review_feas, dim=0)
        selected_review_labels = torch.cat(selected_review_labels, dim=0)
    else:
        selected_review_logits = torch.tensor([], device=logits.device)
        selected_review_labels = torch.tensor([], device=logits.device, dtype=torch.long)
    
    # 8. 将conf_logits和筛选后的review_data_logits合并
    extended_logits = torch.cat([conf_logits, selected_review_logits], dim=0)
    extended_feas =  torch.cat([conf_feas, selected_review_feas], dim=0)
    extended_labels = torch.cat([conf_labels, selected_review_labels], dim=0)
    extended_k = extended_logits.size(0)  # 扩展后的样本数量
    
    if extended_k == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    # 9. 计算扩展样本集的余弦相似度矩阵
    sim_matrix = F.cosine_similarity(
        extended_feas.unsqueeze(1),  # [extended_k, 1, num_classes]
        extended_feas.unsqueeze(0),  # [1, extended_k, num_classes]
        dim=-1
    )
    
    # 10. 创建正负样本掩码
    # 正样本：相同伪标签且非自身
    pos_mask = (extended_labels.unsqueeze(0) == extended_labels.unsqueeze(1)) & \
               (~torch.eye(extended_k, dtype=torch.bool, device=logits.device))
    
    # 负样本：不同伪标签
    neg_mask = extended_labels.unsqueeze(0) != extended_labels.unsqueeze(1)
    
    # 11. 计算每个样本的正样本数量
    pos_counts = pos_mask.sum(dim=1).float()  # [extended_k]
    valid_samples = pos_counts > 0  # 排除没有正样本的样本
    
    # 12. 初始化损失
    total_loss = torch.tensor(0.0, device=logits.device)
    valid_count = 0
    
    # 13. 遍历每个样本计算损失（遍历extended_logits的样本）
    for i in range(extended_feas.size(0)):  # 遍历extended_logits的样本
        if i >= extended_k or not valid_samples[i]:
            continue  # 跳过索引超出或没有正样本的样本
            
        # 获取当前样本的正样本索引
        pos_indices = torch.where(pos_mask[i])[0]
        
        # 计算分子：正样本的指数相似度
        numerator = torch.exp(sim_matrix[i, pos_indices] / temperature)
        
        # 获取当前样本的负样本索引
        neg_indices = torch.where(neg_mask[i])[0]
        
        # 计算分母：负样本的指数相似度之和
        denominator = torch.sum(torch.exp(sim_matrix[i, neg_indices] / temperature))
        
        # 计算损失项：-sum(log(分子/分母))
        loss_term = -torch.sum(torch.log(numerator / (denominator + 1e-8)))  # 添加小量避免除零
        
        # 除以正样本数量（|pos(i)|）
        loss_term /= pos_counts[i]
        
        # 累加损失
        total_loss += loss_term
        valid_count += 1
    
    # 14. 计算平均损失
    if valid_count > 0:
        return total_loss / valid_count
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)






    


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



class CE_KL_review_weighted_10(nn.Module):

    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, confidence_threshold=0.6, num_classes=4, entropy_threshold=0.5, ratio=0.75, weight_type='entropy', thre_alpha=1.0, gate_type='mean'):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.scale = scale
        self.confidence_threshold = confidence_threshold  # 置信度阈值
        self.num_classes = num_classes
        self.entropy_threshold = entropy_threshold
        self.ratio = ratio
        self.weight_type = weight_type
        self.thre_alpha = thre_alpha
        self.gate_type = gate_type

    def forward(self, logits, preds_of_data_review, review_data_class, review_data_logits, current_threshold, current_threshold_std):

        # 判断memory_bank是否为空
        if not review_data_logits.shape[0] == 0:

            class_counts = torch.bincount(review_data_class, minlength=self.num_classes)
            # 计算每个样本的权重：基于其类别的出现频率
            epsilon = 1e-6  # 小常数防止除零
            
            # 为每个样本创建权重：权重 = 1 / 该类别的出现次数
            sample_weights = 1.0 / (class_counts[review_data_class].float() + epsilon)
            
            # 可选：对权重进行归一化，使得权重和为1
            sample_weights = sample_weights / sample_weights.sum()
            
            # 计算不带权重的CE loss（使用reduction='none'得到每个样本的损失）
            ce_loss_per_sample = F.cross_entropy(
                preds_of_data_review, 
                review_data_class, 
                reduction='none'
            )
            # 手动应用样本权重
            weighted_ce_loss = (ce_loss_per_sample * sample_weights).sum()
            loss_sum = (self.lambda_1 * _entropy(logits) + 
                self.lambda_2 * _kl_loss(logits) + 
                self.lambda_3 * weighted_ce_loss)  
        else:
            loss_sum = (self.lambda_1 * _entropy(logits) + 
                    self.lambda_2 * _kl_loss(logits))
        
        return loss_sum


class ConsSamples_selection_two_stage_weighted_4_1_review_4_4_3(nn.Module):
    # special version for two stage model updating
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, entropy_threshold=0.5, ratio_reivew=0.25, weight_type='entropy_energy', thre_alpha=1.0, loss_weight_type='sigmoid',gate_type='mean'):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs
        self.scale = scale
        self.entropy_threshold = entropy_threshold  # Threshold for entropy_avg
        self.ratio_review = ratio_reivew
        self.weight_type = weight_type
        self.thre_alpha = thre_alpha
        self.loss_weight_type = loss_weight_type
        self.gate_type = gate_type

    def forward(self, logits, logits_initial, review_data_logits, current_threshold, current_threshold_std):
        
        # 计算当前mini-batch的均值entropy
        entropy_normalized = _entropy_samples_normalized(logits_initial)
        if self.gate_type in ['mean']:
            entropy_avg = torch.mean(entropy_normalized)
        elif self.gate_type in ['median']:
            entropy_avg = torch.median(entropy_normalized)

        # 归一化当前的阈值
        C = logits.size(1)  # categories
        C_tensor = torch.tensor(C, dtype=torch.float, device=logits.device)
        max_entropy = torch.log(C_tensor)
        _threshold = current_threshold / max_entropy + self.thre_alpha * current_threshold_std / max_entropy

        # 权重设置
        if self.loss_weight_type in ['sigmoid']:    
            # Transform input for weight calculation
            transformed_input = self.scale * (2 * entropy_avg - 1)  # map to [-scale, scale]
            # Constrain output to [0,1] using Sigmoid
            weight_ = torch.sigmoid(-transformed_input / self.temp)
        elif self.loss_weight_type in ['linear']:
            weight_ = 1.0 - entropy_avg
        elif self.loss_weight_type in ['buffer_sigmoid']:
            weight_ = torch.sigmoid(-(entropy_avg - current_threshold/max_entropy)*self.scale / self.temp)
        elif self.loss_weight_type in ['gate']:
            if entropy_avg < _threshold:
                weight_ = 1.0
            else:
                weight_ = 0.0
        elif self.loss_weight_type in ['constant_0']:
            weight_ = 0.0
        else:
            weight_ = 1.0

        # 计算对比损失函数
        if not review_data_logits.shape[0] == 0:
            cons_loss = contrastive_loss_samples_selection_review_2_2(logits, review_data_logits, ratio=self.ratio, ratio_review=self.ratio_review, temperature=self.temp, weight_type=self.weight_type)
        else:
            cons_loss = contrastive_loss_samples_selection_modified(logits, ratio=self.ratio, temperature=self.temp, weight_type=self.weight_type)
    
        # Compute final loss
        loss_sum = self.lambda_3 * weight_ * cons_loss

        return loss_sum
    

class ConsSamples_selection_two_stage_weighted_4_1_review_4_4_4(nn.Module):
    # special version for two stage model updating
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, entropy_threshold=0.5, ratio_reivew=0.25, weight_type='entropy_energy', thre_alpha=1.0, loss_weight_type='sigmoid',gate_type='mean'):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs
        self.scale = scale
        self.entropy_threshold = entropy_threshold  # Threshold for entropy_avg
        self.ratio_review = ratio_reivew
        self.weight_type = weight_type
        self.thre_alpha = thre_alpha
        self.loss_weight_type = loss_weight_type
        self.gate_type = gate_type

    def forward(self, logits, logits_initial, review_data_logits, current_threshold, current_threshold_std):
        
        # 计算当前mini-batch的均值entropy
        entropy_normalized = _entropy_samples_normalized(logits_initial)
        if self.gate_type in ['mean']:
            entropy_avg = torch.mean(entropy_normalized)
        elif self.gate_type in ['median']:
            entropy_avg = torch.median(entropy_normalized)

        # 归一化当前的阈值
        C = logits.size(1)  # categories
        C_tensor = torch.tensor(C, dtype=torch.float, device=logits.device)
        max_entropy = torch.log(C_tensor)
        _threshold = current_threshold / max_entropy + self.thre_alpha * current_threshold_std / max_entropy

        # 权重设置
        if self.loss_weight_type in ['sigmoid']:    
            # Transform input for weight calculation
            transformed_input = self.scale * (2 * entropy_avg - 1)  # map to [-scale, scale]
            # Constrain output to [0,1] using Sigmoid
            weight_ = torch.sigmoid(-transformed_input / self.temp)
        elif self.loss_weight_type in ['linear']:
            weight_ = 1.0 - entropy_avg
        elif self.loss_weight_type in ['buffer_sigmoid']:
            weight_ = torch.sigmoid(-(entropy_avg - current_threshold/max_entropy)*self.scale / self.temp)
        elif self.loss_weight_type in ['gate']:
            if entropy_avg < _threshold:
                weight_ = 1.0
            else:
                weight_ = 0.0
        elif self.loss_weight_type in ['constant_0']:
            weight_ = 0.0
        else:
            weight_ = 1.0

        
        if self.loss_weight_type in ['no_buffer']:
            cons_loss = contrastive_loss_samples_selection_modified(logits, ratio=self.ratio, temperature=self.temp, weight_type=self.weight_type)
        elif self.loss_weight_type in ['buffer_only']:
            cons_loss = contrastive_loss_samples_selection_modified(review_data_logits, ratio=self.ratio_review, temperature=self.temp, weight_type=self.weight_type)
        else:
            # 计算对比损失函数
            if not review_data_logits.shape[0] == 0:
                cons_loss = contrastive_loss_samples_selection_review_2_2(logits, review_data_logits, ratio=self.ratio, ratio_review=self.ratio_review, temperature=self.temp, weight_type=self.weight_type)
            else:
                cons_loss = contrastive_loss_samples_selection_modified(logits, ratio=self.ratio, temperature=self.temp, weight_type=self.weight_type)

        # Compute final loss
        loss_sum = weight_ * cons_loss

        return loss_sum


class ConsSamples_selection_two_stage_weighted_4_1_review_4_4_4_feas(nn.Module):
    # special version for two stage model updating
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, entropy_threshold=0.5, ratio_reivew=0.25, weight_type='entropy_energy', thre_alpha=1.0, loss_weight_type='sigmoid',gate_type='mean'):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs
        self.scale = scale
        self.entropy_threshold = entropy_threshold  # Threshold for entropy_avg
        self.ratio_review = ratio_reivew
        self.weight_type = weight_type
        self.thre_alpha = thre_alpha
        self.loss_weight_type = loss_weight_type
        self.gate_type = gate_type

    def forward(self, logits, logits_initial, review_data_logits, feas, review_feas, current_threshold, current_threshold_std):
        
        # 计算当前mini-batch的均值entropy
        entropy_normalized = _entropy_samples_normalized(logits_initial)
        if self.gate_type in ['mean']:
            entropy_avg = torch.mean(entropy_normalized)
        elif self.gate_type in ['median']:
            entropy_avg = torch.median(entropy_normalized)

        # 归一化当前的阈值
        C = logits.size(1)  # categories
        C_tensor = torch.tensor(C, dtype=torch.float, device=logits.device)
        max_entropy = torch.log(C_tensor)
        _threshold = current_threshold / max_entropy + self.thre_alpha * current_threshold_std / max_entropy

        # 权重设置
        if self.loss_weight_type in ['sigmoid']:    
            # Transform input for weight calculation
            transformed_input = self.scale * (2 * entropy_avg - 1)  # map to [-scale, scale]
            # Constrain output to [0,1] using Sigmoid
            weight_ = torch.sigmoid(-transformed_input / self.temp)
        elif self.loss_weight_type in ['linear']:
            weight_ = 1.0 - entropy_avg
        elif self.loss_weight_type in ['buffer_sigmoid']:
            weight_ = torch.sigmoid(-(entropy_avg - current_threshold/max_entropy)*self.scale / self.temp)
        elif self.loss_weight_type in ['gate']:
            if entropy_avg < _threshold:
                weight_ = 1.0
            else:
                weight_ = 0.0
        elif self.loss_weight_type in ['constant_0']:
            weight_ = 0.0
        else:
            weight_ = 1.0

        
        if self.loss_weight_type in ['no_buffer']:
            cons_loss = contrastive_loss_samples_selection_modified_feas(logits, feas, ratio=self.ratio, temperature=self.temp, weight_type=self.weight_type)
        else:
            # 计算对比损失函数
            if not review_data_logits.shape[0] == 0:
                cons_loss = contrastive_loss_samples_selection_review_2_2_feas(logits, review_data_logits, feas, review_feas, ratio=self.ratio, ratio_review=self.ratio_review, temperature=self.temp, weight_type=self.weight_type)
            else:
                cons_loss = contrastive_loss_samples_selection_modified_feas(logits, feas, ratio=self.ratio, temperature=self.temp, weight_type=self.weight_type)

        # Compute final loss
        loss_sum = weight_ * cons_loss

        return loss_sum


class ConsSamples_selection_two_stage_weighted_4_1_modified(nn.Module):
    # special version for two stage model updating
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, weight_type='entropy_energy'):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs
        self.scale = scale
        self.weight_type = weight_type

    def forward(self, logits, logits_initial):
        
        batch_size = logits_initial.size(0)
        entropy_normalized = _entropy_samples_normalized(logits_initial)
        entropy_avg = torch.mean(entropy_normalized)
        
        cons_loss = contrastive_loss_samples_selection_modified(logits, ratio=self.ratio, temperature=self.temp, weight_type=self.weight_type)

        transformed_input = self.scale * (2 * entropy_avg - 1)  # map to [-scale, scale]
        
        # 通过 Sigmoid 约束输出到 [0,1]
        weight_ = torch.sigmoid(-transformed_input / self.temp)

        loss_sum = self.lambda_3 * weight_ * cons_loss
        
        return  loss_sum
    
class ConsSamples_selection_two_stage_weighted_4_1_modified_2(nn.Module):
    # special version for two stage model updating
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, weight_type='entropy_energy'):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs
        self.scale = scale
        self.weight_type = weight_type

    def forward(self, logits, logits_initial):
        
        batch_size = logits_initial.size(0)
        entropy_normalized = _entropy_samples_normalized(logits_initial)
        entropy_avg = torch.mean(entropy_normalized)
        
        cons_loss = contrastive_loss_samples_selection_modified(logits, ratio=self.ratio, temperature=self.temp, weight_type=self.weight_type)

        transformed_input = self.scale * (2 * entropy_avg - 1)  # map to [-scale, scale]
        
        # 通过 Sigmoid 约束输出到 [0,1]
        weight_ = torch.sigmoid(-transformed_input)

        loss_sum = self.lambda_3 * weight_ * cons_loss
        
        return  loss_sum

class ConsSamples_selection_two_stage_weighted_4_1_modified_feas(nn.Module):
    # special version for two stage model updating
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, weight_type='entropy_energy'):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs
        self.scale = scale
        self.weight_type = weight_type

    def forward(self, logits, feas, logits_initial):
        
        batch_size = logits_initial.size(0)
        entropy_normalized = _entropy_samples_normalized(logits_initial)
        entropy_avg = torch.mean(entropy_normalized)
        
        cons_loss = contrastive_loss_samples_selection_modified_feas(logits, feas, ratio=self.ratio, temperature=self.temp, weight_type=self.weight_type)

        transformed_input = self.scale * (2 * entropy_avg - 1)  # map to [-scale, scale]
        
        # 通过 Sigmoid 约束输出到 [0,1]
        weight_ = torch.sigmoid(-transformed_input)

        loss_sum = self.lambda_3 * weight_ * cons_loss
        
        return  loss_sum

class ConsSamples_selection_two_stage_weighted_4_1_modified_1(nn.Module):
    # special version for two stage model updating
    def __init__(self,  ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, weight_type='entropy_energy', loss_weight_type='sigmoid'):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ratio = ratio  # Ratio of samples to select for entropy++lcs
        self.scale = scale
        self.weight_type = weight_type
        self.loss_weight_type = loss_weight_type

    def forward(self, logits, logits_initial):
        
        batch_size = logits_initial.size(0)
        entropy_normalized = _entropy_samples_normalized(logits_initial)
        entropy_avg = torch.mean(entropy_normalized)
        
        cons_loss = contrastive_loss_samples_selection_modified(logits, ratio=self.ratio, temperature=self.temp, weight_type=self.weight_type)

        if self.loss_weight_type in ['sigmoid']:
            transformed_input = self.scale * (2 * entropy_avg - 1)  # map to [-scale, scale]
            # 通过 Sigmoid 约束输出到 [0,1]
            weight_ = torch.sigmoid(-transformed_input / self.temp)
        elif self.loss_weight_type in ['constant_0']:
            weight_ = 0.0
        else:
            weight_ = 1.0

        loss_sum = self.lambda_3 * weight_ * cons_loss
        
        return  loss_sum


class CE_KL_review_weighted_10_constrastive(nn.Module):

    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, confidence_threshold=0.6, num_classes=4, entropy_threshold=0.5, ratio=0.75, weight_type='entropy', thre_alpha=1.0, gate_type='mean', ratio_review=0.25):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.scale = scale
        self.confidence_threshold = confidence_threshold  # 置信度阈值
        self.num_classes = num_classes
        self.entropy_threshold = entropy_threshold
        self.ratio = ratio
        self.weight_type = weight_type
        self.thre_alpha = thre_alpha
        self.gate_type = gate_type
        self.ratio_review = ratio_review

    def forward(self, logits, preds_of_data_review, review_data_class, review_data_logits, current_threshold, current_threshold_std):

        # 判断memory_bank是否为空
        if not review_data_logits.shape[0] == 0:

            class_counts = torch.bincount(review_data_class, minlength=self.num_classes)
            # 计算每个样本的权重：基于其类别的出现频率
            epsilon = 1e-6  # 小常数防止除零
            
            # 为每个样本创建权重：权重 = 1 / 该类别的出现次数
            sample_weights = 1.0 / (class_counts[review_data_class].float() + epsilon)
            
            # 可选：对权重进行归一化，使得权重和为1
            sample_weights = sample_weights / sample_weights.sum()
            
            # 计算不带权重的CE loss（使用reduction='none'得到每个样本的损失）
            ce_loss_per_sample = F.cross_entropy(
                preds_of_data_review, 
                review_data_class, 
                reduction='none'
            )
            # 手动应用样本权重
            weighted_ce_loss = (ce_loss_per_sample * sample_weights).sum()

            cons_loss = contrastive_loss_samples_selection_review_2_2(logits, preds_of_data_review, ratio=self.ratio, ratio_review=self.ratio_review, temperature=self.temp, weight_type=self.weight_type)

            loss_sum = (self.lambda_1 * _entropy(logits) + 
                self.lambda_2 * _kl_loss(logits) + 
                self.lambda_3 * weighted_ce_loss + cons_loss)  
        else:
            loss_sum = (self.lambda_1 * _entropy(logits) + 
                    self.lambda_2 * _kl_loss(logits))
        
        return loss_sum
    
class CE_KL_review_weighted_10_constrastive_visual(nn.Module):

    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, confidence_threshold=0.6, num_classes=4, entropy_threshold=0.5, ratio=0.75, weight_type='entropy', thre_alpha=1.0, gate_type='mean', ratio_review=0.25):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.scale = scale
        self.confidence_threshold = confidence_threshold  # 置信度阈值
        self.num_classes = num_classes
        self.entropy_threshold = entropy_threshold
        self.ratio = ratio
        self.weight_type = weight_type
        self.thre_alpha = thre_alpha
        self.gate_type = gate_type
        self.ratio_review = ratio_review

    def forward(self, logits, preds_of_data_review, review_data_class, review_data_logits, current_threshold, current_threshold_std):

        # 判断memory_bank是否为空
        if not review_data_logits.shape[0] == 0:

            class_counts = torch.bincount(review_data_class, minlength=self.num_classes)
            # 计算每个样本的权重：基于其类别的出现频率
            epsilon = 1e-6  # 小常数防止除零
            
            # 为每个样本创建权重：权重 = 1 / 该类别的出现次数
            sample_weights = 1.0 / (class_counts[review_data_class].float() + epsilon)
            
            # 可选：对权重进行归一化，使得权重和为1
            sample_weights = sample_weights / sample_weights.sum()
            
            # 计算不带权重的CE loss（使用reduction='none'得到每个样本的损失）
            ce_loss_per_sample = F.cross_entropy(
                preds_of_data_review, 
                review_data_class, 
                reduction='none'
            )
            # 手动应用样本权重
            weighted_ce_loss = (ce_loss_per_sample * sample_weights).sum()

            cons_loss = contrastive_loss_samples_selection_review_2_2(logits, preds_of_data_review, ratio=self.ratio, ratio_review=self.ratio_review, temperature=self.temp, weight_type=self.weight_type)

            loss_sum = (self.lambda_1 * _entropy(logits) + 
                self.lambda_2 * _kl_loss(logits) + 
                self.lambda_3 * weighted_ce_loss + cons_loss)  
            
            loss_sum_1 = (self.lambda_1 * _entropy(logits) + 
                self.lambda_2 * _kl_loss(logits) + 
                self.lambda_3 * weighted_ce_loss)  
            loss_sum_2 = cons_loss
        else:
            loss_sum = (self.lambda_1 * _entropy(logits) + 
                    self.lambda_2 * _kl_loss(logits))
        
        return loss_sum, loss_sum_1, loss_sum_2


class CE_KL_review_weighted_10_constrastive_visual_PCGrad(nn.Module):

    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, confidence_threshold=0.6, num_classes=4, entropy_threshold=0.5, ratio=0.75, weight_type='entropy', thre_alpha=1.0, gate_type='mean', ratio_review=0.25):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.scale = scale
        self.confidence_threshold = confidence_threshold  # 置信度阈值
        self.num_classes = num_classes
        self.entropy_threshold = entropy_threshold
        self.ratio = ratio
        self.weight_type = weight_type
        self.thre_alpha = thre_alpha
        self.gate_type = gate_type
        self.ratio_review = ratio_review

    def forward(self, logits, preds_of_data_review, review_data_class, review_data_logits, current_threshold, current_threshold_std):

        # 判断memory_bank是否为空
        if not review_data_logits.shape[0] == 0:

            class_counts = torch.bincount(review_data_class, minlength=self.num_classes)
            # 计算每个样本的权重：基于其类别的出现频率
            epsilon = 1e-6  # 小常数防止除零
            
            # 为每个样本创建权重：权重 = 1 / 该类别的出现次数
            sample_weights = 1.0 / (class_counts[review_data_class].float() + epsilon)
            
            # 可选：对权重进行归一化，使得权重和为1
            sample_weights = sample_weights / sample_weights.sum()
            
            # 计算不带权重的CE loss（使用reduction='none'得到每个样本的损失）
            ce_loss_per_sample = F.cross_entropy(
                preds_of_data_review, 
                review_data_class, 
                reduction='none'
            )
            # 手动应用样本权重
            weighted_ce_loss = (ce_loss_per_sample * sample_weights).sum()

            cons_loss = contrastive_loss_samples_selection_review_2_2(logits, preds_of_data_review, ratio=self.ratio, ratio_review=self.ratio_review, temperature=self.temp, weight_type=self.weight_type)

            loss_sum_1 = self.lambda_1 * (_entropy(logits) + 
                _kl_loss(logits) + 
                weighted_ce_loss)  
            loss_sum_2 = self.lambda_2 * cons_loss

            loss_sum = loss_sum_1 + loss_sum_2
        else:
            loss_sum = (self.lambda_1 * _entropy(logits) + 
                    self.lambda_2 * _kl_loss(logits))
        
        return loss_sum, loss_sum_1, loss_sum_2


class CE_KL_review_weighted_10_constrastive_fea(nn.Module):

    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, confidence_threshold=0.6, num_classes=4, entropy_threshold=0.5, ratio=0.75, weight_type='entropy', thre_alpha=1.0, gate_type='mean', ratio_review=0.25):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.scale = scale
        self.confidence_threshold = confidence_threshold  # 置信度阈值
        self.num_classes = num_classes
        self.entropy_threshold = entropy_threshold
        self.ratio = ratio
        self.weight_type = weight_type
        self.thre_alpha = thre_alpha
        self.gate_type = gate_type
        self.ratio_review = ratio_review

    def forward(self, logits, preds_of_data_review, review_data_class, review_data_logits, feas, review_feas, current_threshold, current_threshold_std):

        # 判断memory_bank是否为空
        if not review_data_logits.shape[0] == 0:

            class_counts = torch.bincount(review_data_class, minlength=self.num_classes)
            # 计算每个样本的权重：基于其类别的出现频率
            epsilon = 1e-6  # 小常数防止除零
            
            # 为每个样本创建权重：权重 = 1 / 该类别的出现次数
            sample_weights = 1.0 / (class_counts[review_data_class].float() + epsilon)
            
            # 可选：对权重进行归一化，使得权重和为1
            sample_weights = sample_weights / sample_weights.sum()
            
            # 计算不带权重的CE loss（使用reduction='none'得到每个样本的损失）
            ce_loss_per_sample = F.cross_entropy(
                preds_of_data_review, 
                review_data_class, 
                reduction='none'
            )
            # 手动应用样本权重
            weighted_ce_loss = (ce_loss_per_sample * sample_weights).sum()

            cons_loss = contrastive_loss_samples_selection_review_2_2_feas(logits, review_data_logits, feas, review_feas, ratio=self.ratio, ratio_review=self.ratio_review, temperature=self.temp, weight_type=self.weight_type)

            loss_sum = (self.lambda_1 * _entropy(logits) + 
                self.lambda_2 * _kl_loss(logits) + 
                self.lambda_3 * weighted_ce_loss + cons_loss)  
        else:
            loss_sum = (self.lambda_1 * _entropy(logits) + 
                    self.lambda_2 * _kl_loss(logits))
        
        return loss_sum


class review_constrastive_loss(nn.Module):

    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, confidence_threshold=0.6, num_classes=4, entropy_threshold=0.5, ratio=0.75, weight_type='entropy', thre_alpha=1.0, gate_type='mean', ratio_review=0.25):
        super().__init__()
        self.temp = temp      # Temperature scaling factor
        self.softplus = nn.Softplus()  # Activation function for loss calculation
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.scale = scale
        self.confidence_threshold = confidence_threshold  # 置信度阈值
        self.num_classes = num_classes
        self.entropy_threshold = entropy_threshold
        self.ratio = ratio
        self.weight_type = weight_type
        self.thre_alpha = thre_alpha
        self.gate_type = gate_type
        self.ratio_review = ratio_review

    def forward(self, logits, preds_of_data_review, review_data_class, review_data_logits, feas, review_feas, current_threshold, current_threshold_std):

        # 判断memory_bank是否为空
        if not review_data_logits.shape[0] == 0:

            cons_loss = contrastive_loss_samples_selection_review_2_2(logits, preds_of_data_review, ratio=self.ratio, ratio_review=self.ratio_review, temperature=self.temp, weight_type=self.weight_type)

            loss_sum = cons_loss  
        else:
            loss_sum = contrastive_loss_samples_selection_modified(logits, ratio=self.ratio, temperature=self.temp, weight_type=self.weight_type)

        
        return loss_sum
    
