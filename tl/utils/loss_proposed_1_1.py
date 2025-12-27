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



    
def contrastive_loss_samples_selection(logits, ratio=0.75, temperature=0.07, weight_type='entropy_energy'):
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


    

def contrastive_loss_samples_selection_review_2_1(logits, review_data_logits, ratio=0.75, ratio_review=0.75, temperature=0.07, weight_type='entropy_energy'):
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
        
        # 选择权重最小的k_review个样本（最不确定的样本）
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
    
    # 14. 计算平均损失
    if valid_count > 0:
        return total_loss / valid_count
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)



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
    



    

    

class CE_KL_review_weighted_3(nn.Module):

    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, confidence_threshold=0.6, num_classes=4, entropy_threshold=0.5):
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

    def forward(self, logits, preds_of_data_review, review_data_class, review_data_logits):
        # 计算每个类别的频率（基于当前batch）
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
            
            # 进一步计算当前样本的权重
            entropy_normalized = _entropy_samples_normalized(logits.detach().clone())
            entropy_avg = torch.mean(entropy_normalized)
            # transformed_input = self.scale * (2 * entropy_avg - 1)  # map to [-scale, scale]
            # weight_ = torch.sigmoid(-transformed_input / self.temp)
            weight_ = 1.0 if entropy_avg < self.entropy_threshold else 0.0
        else:
            weighted_ce_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            weight_ = torch.tensor(0.0, device=logits.device, requires_grad=True)

        # 这里假设 _entropy 和 _kl_loss 是您定义的其他损失函数
        loss_sum = (self.lambda_1 * _entropy(logits) + 
                   self.lambda_2 * _kl_loss(logits) + 
                   self.lambda_3 * weight_ * weighted_ce_loss)
        
        return loss_sum


class CE_KL_review_weighted_6(nn.Module):

    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, confidence_threshold=0.6, num_classes=4, entropy_threshold=0.5):
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

    def forward(self, logits, preds_of_data_review, review_data_class, review_data_logits):
        # 计算每个类别的频率（基于当前batch）
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
            
            # 进一步计算当前样本的权重
            entropy_normalized = _entropy_samples_normalized(logits.detach().clone())
            entropy_avg = torch.mean(entropy_normalized)
            # transformed_input = self.scale * (2 * entropy_avg - 1)  # map to [-scale, scale]
            # weight_ = torch.sigmoid(-transformed_input / self.temp)
            weight_ = 1.0 if entropy_avg < self.entropy_threshold else 0.0
        else:
            weighted_ce_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            weight_ = torch.tensor(0.0, device=logits.device, requires_grad=True)

        # 这里假设 _entropy 和 _kl_loss 是您定义的其他损失函数
        loss_sum = ((1.0-weight_) * (self.lambda_1 * _entropy(logits) + 
                   self.lambda_2 * _kl_loss(logits)) + 
                   self.lambda_3 * weight_ * weighted_ce_loss)
        
        return loss_sum


class CE_KL_review_weighted_7(nn.Module):

    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, confidence_threshold=0.6, num_classes=4, entropy_threshold=0.5, ratio=0.75, weight_type='entropy'):
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

    def forward(self, logits, preds_of_data_review, review_data_class, review_data_logits):
        
        # 判断memory_bank是否为空
        if not review_data_logits.shape[0] == 0:
            
            # 进一步计算当前样本的权重
            entropy_normalized = _entropy_samples_normalized(logits.detach().clone())
            entropy_avg = torch.mean(entropy_normalized)
            
            # 判断当前的 entropy_avg 是否小于阈值
            if entropy_avg < self.entropy_threshold:
                # 小于阈值，使用当前的mini-batch的数据和过往的memory_bank数据计算损失函数
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
                # 如果大于阈值，则使用当前mini-batch的数据计算损失函数
                # 选择熵较低的样本
                if self.weight_type == 'entropy':
                    entropy = _entropy_samples(logits)
                    weights = entropy
                else:
                    raise ValueError(f"Unsupported weight_type: {self.weight_type}")
                k = max(1, int(logits.size(0) * self.ratio))  # 至少选择1个样本
                _, conf_indices = torch.topk(weights, k, largest=False, sorted=True)
                # 低熵样本的logits用于计算损失
                conf_logits = logits[conf_indices]
                loss_sum = (self.lambda_1 * _entropy(conf_logits) + 
                   self.lambda_2 * _kl_loss(conf_logits))        
        # 空的话直接使用当前mini-batch的数据计算损失函数（一般情况下不会使用到）    
        else:
            loss_sum = (self.lambda_1 * _entropy(logits) + 
                    self.lambda_2 * _kl_loss(logits))
        
        return loss_sum


class CE_KL_review_weighted_8(nn.Module):

    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, confidence_threshold=0.6, num_classes=4, entropy_threshold=0.5, ratio=0.75, weight_type='entropy', thre_alpha=1.0):
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

    def forward(self, logits, preds_of_data_review, review_data_class, review_data_logits, current_threshold):
        
        # 归一化当前的阈值
        C = logits.size(1)  # categories
        C_tensor = torch.tensor(C, dtype=torch.float, device=logits.device)
        max_entropy = torch.log(C_tensor)
        current_threshold = self.thre_alpha * current_threshold / max_entropy

        # 判断memory_bank是否为空
        if not review_data_logits.shape[0] == 0:
            
            # 进一步计算当前样本的权重
            entropy_normalized = _entropy_samples_normalized(logits.detach().clone())
            entropy_avg = torch.mean(entropy_normalized)
            
            # 判断当前的 entropy_avg 是否小于阈值current_threshold
            if entropy_avg < current_threshold:
                # 小于阈值，使用当前的mini-batch的数据和过往的memory_bank数据计算损失函数
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
                # 如果大于阈值，则使用当前mini-batch的数据计算损失函数
                # 选择熵较低的样本
                if self.weight_type == 'entropy':
                    entropy = _entropy_samples(logits)
                    weights = entropy
                else:
                    raise ValueError(f"Unsupported weight_type: {self.weight_type}")
                k = max(1, int(logits.size(0) * self.ratio))  # 至少选择1个样本
                _, conf_indices = torch.topk(weights, k, largest=False, sorted=True)
                # 低熵样本的logits用于计算损失
                conf_logits = logits[conf_indices]
                loss_sum = (self.lambda_1 * _entropy(conf_logits) + 
                   self.lambda_2 * _kl_loss(conf_logits))        
        # 空的话直接使用当前mini-batch的数据计算损失函数（一般情况下不会使用到）    
        else:
            loss_sum = (self.lambda_1 * _entropy(logits) + 
                    self.lambda_2 * _kl_loss(logits))
        
        return loss_sum
    

class CE_KL_review_weighted_9(nn.Module):

    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, confidence_threshold=0.6, num_classes=4, entropy_threshold=0.5, ratio=0.75, weight_type='entropy', thre_alpha=1.0):
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

    def forward(self, logits, preds_of_data_review, review_data_class, review_data_logits, current_threshold, current_threshold_std):
        
        # 归一化当前的阈值
        C = logits.size(1)  # categories
        C_tensor = torch.tensor(C, dtype=torch.float, device=logits.device)
        max_entropy = torch.log(C_tensor)
        _threshold = current_threshold / max_entropy + self.thre_alpha * current_threshold_std / max_entropy

        # 判断memory_bank是否为空
        if not review_data_logits.shape[0] == 0:
            
            # 进一步计算当前样本的权重
            entropy_normalized = _entropy_samples_normalized(logits.detach().clone())
            entropy_avg = torch.mean(entropy_normalized)
            
            # 判断当前的 entropy_avg 是否小于阈值_threshold
            if entropy_avg < _threshold:
                # 小于阈值，使用当前的mini-batch的数据和过往的memory_bank数据计算损失函数
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
                # 如果大于阈值，则使用当前mini-batch的数据计算损失函数
                # 选择熵较低的样本
                if self.weight_type == 'entropy':
                    entropy = _entropy_samples(logits)
                    weights = entropy
                else:
                    raise ValueError(f"Unsupported weight_type: {self.weight_type}")
                k = max(1, int(logits.size(0) * self.ratio))  # 至少选择1个样本
                _, conf_indices = torch.topk(weights, k, largest=False, sorted=True)
                # 低熵样本的logits用于计算损失
                conf_logits = logits[conf_indices]
                loss_sum = (self.lambda_1 * _entropy(conf_logits) + 
                   self.lambda_2 * _kl_loss(conf_logits))        
        # 空的话直接使用当前mini-batch的数据计算损失函数（一般情况下不会使用到）    
        else:
            loss_sum = (self.lambda_1 * _entropy(logits) + 
                    self.lambda_2 * _kl_loss(logits))
        
        return loss_sum

class CE_KL_review_3(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, confidence_threshold=0.6, num_classes=4, entropy_threshold=0.5):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.scale = scale
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        self.entropy_threshold = entropy_threshold

    def forward(self, logits, preds_of_data_review, review_data_class, review_data_logits):
        if not review_data_logits.shape[0] == 0:
            # 计算不带权重的CE loss
            ce_loss = F.cross_entropy(
                preds_of_data_review, 
                review_data_class
            )
            entropy_normalized = _entropy_samples_normalized(logits.detach().clone())
            entropy_avg = torch.mean(entropy_normalized)
            weight_ = 1.0 if entropy_avg < self.entropy_threshold else 0.0
        else:
            ce_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            weight_ = torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss_sum = (self.lambda_1 * _entropy(logits) + 
                   self.lambda_2 * _kl_loss(logits) + 
                   self.lambda_3 * weight_ * ce_loss)
        
        return loss_sum


class CE_KL_review_4(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, confidence_threshold=0.6, num_classes=4, entropy_threshold=0.5):
        super().__init__()
        self.temp = temp
        self.softplus = nn.Softplus()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.scale = scale
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        self.entropy_threshold = entropy_threshold

    def forward(self, logits, preds_of_data_review, review_data_class, review_data_logits):
        if not review_data_logits.shape[0] == 0:
            # 计算不带权重的CE loss
            ce_loss = F.cross_entropy(
                preds_of_data_review, 
                review_data_class
            )
            entropy_normalized = _entropy_samples_normalized(logits.detach().clone())
            entropy_avg = torch.mean(entropy_normalized)
            weight_ = 1.0 if entropy_avg < self.entropy_threshold else 0.0
        else:
            ce_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            weight_ = torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss_sum = ((1-weight_) * (self.lambda_1 * _entropy(logits) + 
                   self.lambda_2 * _kl_loss(logits)) + 
                   self.lambda_3 * weight_ * ce_loss)
        
        return loss_sum


class ConsSamples_selection_two_stage_weighted_4_1_review_4_1(nn.Module):
    # special version for two stage model updating
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, entropy_threshold=0.5, ratio_reivew=0.25, weight_type='entropy_energy'):
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

    def forward(self, logits, logits_initial, review_data_logits):
        batch_size = logits_initial.size(0)
        entropy_normalized = _entropy_samples_normalized(logits_initial)
        entropy_avg = torch.mean(entropy_normalized)

        # Determine weight_ce based on entropy_avg
        weight_ce = 1.0 if entropy_avg < self.entropy_threshold else 0.0

        # Combine logits based on weight_ce
        if weight_ce == 1.0:
            # Compute contrastive loss
            if not review_data_logits.shape[0] == 0:
                cons_loss = contrastive_loss_samples_selection_review_2_1(logits, review_data_logits, ratio=self.ratio, ratio_review=self.ratio_review, temperature=self.temp, weight_type=self.weight_type)
            else:
                cons_loss = contrastive_loss_samples_selection(logits, ratio=self.ratio, temperature=self.temp, weight_type=self.weight_type)
        else:
            cons_loss = contrastive_loss_samples_selection(logits, ratio=self.ratio, temperature=self.temp, weight_type=self.weight_type)
        
        # Transform input for weight calculation
        transformed_input = self.scale * (2 * entropy_avg - 1)  # map to [-scale, scale]

        # Constrain output to [0,1] using Sigmoid
        weight_ = torch.sigmoid(-transformed_input / self.temp)

        # Compute final loss
        loss_sum = self.lambda_3 * weight_ * cons_loss

        return loss_sum


class ConsSamples_selection_two_stage_weighted_4_1_review_4_2(nn.Module):
    # special version for two stage model updating
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, entropy_threshold=0.5, ratio_reivew=0.25, weight_type='entropy_energy'):
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

    def forward(self, logits, logits_initial, review_data_logits):
        
        batch_size = logits_initial.size(0)
        entropy_normalized = _entropy_samples_normalized(logits_initial)
        entropy_avg = torch.mean(entropy_normalized)

        # 判断是否需要进行阈值更新
        if entropy_avg < self.entropy_threshold:
            # 计算对比损失函数
            if not review_data_logits.shape[0] == 0:
                cons_loss = contrastive_loss_samples_selection_review_2_1(logits, review_data_logits, ratio=self.ratio, ratio_review=self.ratio_review, temperature=self.temp, weight_type=self.weight_type)
            else:
                cons_loss = contrastive_loss_samples_selection(logits, ratio=self.ratio, temperature=self.temp, weight_type=self.weight_type)
        else:
            cons_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        # Transform input for weight calculation
        transformed_input = self.scale * (2 * entropy_avg - 1)  # map to [-scale, scale]

        # Constrain output to [0,1] using Sigmoid
        weight_ = torch.sigmoid(-transformed_input / self.temp)

        # Compute final loss
        loss_sum = self.lambda_3 * weight_ * cons_loss

        return loss_sum
    
class ConsSamples_selection_two_stage_weighted_4_1_review_4_3(nn.Module):
    # special version for two stage model updating
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, entropy_threshold=0.5, ratio_reivew=0.25, weight_type='entropy_energy', thre_alpha=1.0):
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

    def forward(self, logits, logits_initial, review_data_logits, current_threshold):
        
        entropy_normalized = _entropy_samples_normalized(logits_initial)
        entropy_avg = torch.mean(entropy_normalized)

        # 归一化当前的阈值
        C = logits.size(1)  # categories
        C_tensor = torch.tensor(C, dtype=torch.float, device=logits.device)
        max_entropy = torch.log(C_tensor)
        current_threshold = self.thre_alpha * current_threshold / max_entropy

        # 判断是否需要进行阈值更新
        if entropy_avg < current_threshold:
            # 计算对比损失函数
            if not review_data_logits.shape[0] == 0:
                cons_loss = contrastive_loss_samples_selection_review_2_1(logits, review_data_logits, ratio=self.ratio, ratio_review=self.ratio_review, temperature=self.temp, weight_type=self.weight_type)
            else:
                cons_loss = contrastive_loss_samples_selection(logits, ratio=self.ratio, temperature=self.temp, weight_type=self.weight_type)
        else:
            cons_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        # Transform input for weight calculation
        transformed_input = self.scale * (2 * entropy_avg - 1)  # map to [-scale, scale]

        # Constrain output to [0,1] using Sigmoid
        weight_ = torch.sigmoid(-transformed_input / self.temp)

        # Compute final loss
        loss_sum = self.lambda_3 * weight_ * cons_loss

        return loss_sum
    

class ConsSamples_selection_two_stage_weighted_4_1_review_4_4(nn.Module):
    # special version for two stage model updating
    def __init__(self, ratio=0.5, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, temp=1.0, scale=5, entropy_threshold=0.5, ratio_reivew=0.25, weight_type='entropy_energy', thre_alpha=1.0):
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

    def forward(self, logits, logits_initial, review_data_logits, current_threshold, current_threshold_std):
        
        entropy_normalized = _entropy_samples_normalized(logits_initial)
        entropy_avg = torch.mean(entropy_normalized)

        # 归一化当前的阈值
        C = logits.size(1)  # categories
        C_tensor = torch.tensor(C, dtype=torch.float, device=logits.device)
        max_entropy = torch.log(C_tensor)
        _threshold = current_threshold / max_entropy + self.thre_alpha * current_threshold_std / max_entropy

        # 判断是否需要进行阈值更新
        if entropy_avg < _threshold:
            # 计算对比损失函数
            if not review_data_logits.shape[0] == 0:
                cons_loss = contrastive_loss_samples_selection_review_2_1(logits, review_data_logits, ratio=self.ratio, ratio_review=self.ratio_review, temperature=self.temp, weight_type=self.weight_type)
            else:
                cons_loss = contrastive_loss_samples_selection(logits, ratio=self.ratio, temperature=self.temp, weight_type=self.weight_type)
        else:
            cons_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        # Transform input for weight calculation
        transformed_input = self.scale * (2 * entropy_avg - 1)  # map to [-scale, scale]

        # Constrain output to [0,1] using Sigmoid
        weight_ = torch.sigmoid(-transformed_input / self.temp)

        # Compute final loss
        loss_sum = self.lambda_3 * weight_ * cons_loss

        return loss_sum