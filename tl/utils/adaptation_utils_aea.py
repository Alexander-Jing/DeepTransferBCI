import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn import linear_model

def define_optimizer(meta_conf, params, lr=1e-3):
    """Set up optimizer for adaptation."""
    weight_decay = meta_conf.weight_decay if hasattr(meta_conf, "weight_decay") else 0

    if not hasattr(meta_conf, "optimizer") or meta_conf.optimizer == "SGD":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=meta_conf.momentum_SGD if hasattr(meta_conf, "momentum_SGD") else 0.9,
            dampening=meta_conf.dampening if hasattr(meta_conf, "dampening") else 0,
            weight_decay=weight_decay,
            nesterov=meta_conf.nesterov if hasattr(meta_conf, "nesterov") else True,
        )
    elif meta_conf.optimizer == "Adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=(meta_conf.beta if hasattr(meta_conf, "beta") else 0.9, 0.999),
            weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError


"""loss-related functions."""


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def softmax_entropy_decompose(x):
    """Decomposed Entropy of softmax distribution from logits."""
    weighted_energys = -(x.softmax(1) * x).sum(1)
    neg_free_energys = -1 * energy(x)
    return weighted_energys, neg_free_energys


def teacher_student_softmax_entropy(
    x: torch.Tensor, x_ema: torch.Tensor
) -> torch.Tensor:
    """Cross entropy between the teacher and student predictions."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


def entropy(input):
    bs = input.size(0)
    ent = -input * torch.log(input + 1e-5)
    ent = torch.sum(ent, dim=1)
    return ent

#################### Energy-related ###############################
def energy(input, temp=1.0):
    bs = input.size(0)
    energy = -temp * torch.logsumexp(input/temp, dim=1)
    return energy

def adaptive_energy(input, temp=1.0):
    bs = input.size(0)
    weight = 1/softmax_entropy(input).exp()
    #weight = -1 * softmax_entropy(input)
    weight = weight.detach()
    energy_values = energy(input, temp=temp)
    return weight * energy_values

# def energy_alignment_with_sample(input, temp=1.0):
#     energy = -temp * torch.logsumexp(input / temp, dim=1)
#     diff = -1
#     loss = nn.ReLU()(diff)
#     return loss

def oracle_energy_alignment(input, src_energy, temp=1.0):
    energy = -temp * torch.logsumexp(input / temp, dim=1)
    tar_energy = energy.detach().mean(0)
    diff = energy - src_energy
    loss = nn.ReLU()(diff)
    return loss, tar_energy

def sample_selective_energy_alignment(input, ratio=0.5, temp=1.0):
    num_chunk = int(1/ratio)

    energy = -temp * torch.logsumexp(input / temp, dim=1)
    values, indices = energy.detach().sort()
    tar_energy = values.mean()

    src_energy_approx = torch.chunk(values, num_chunk)[0].mean()
    diff = energy - src_energy_approx
    loss = nn.ReLU()(diff)
    return loss, src_energy_approx, tar_energy

def sample_selective_softplus_energy_alignment(input, ratio=0.5, temp=1.0):
    num_chunk = int(1/ratio)

    energy = -temp * torch.logsumexp(input / temp, dim=1)
    values, indices = energy.detach().sort()
    tar_energy = values.mean()

    src_energy_approx = torch.chunk(values, num_chunk)[0].mean()
    diff = energy - src_energy_approx
    loss = nn.Softplus()(diff)
    return loss


def energy_distribution_matching(input, ratio=0.5, temp=1.0):
    num_chunk = int(1/ratio)

    energy = -temp * torch.logsumexp(input / temp, dim=1)
    values, indices = energy.detach().sort()
    tar_energy = values.mean()
    tar_mean, tar_std = energy.mean(), energy.std()

    src_energy = torch.chunk(values, num_chunk)[0]
    src_mean, src_std = src_energy.mean(), src_energy.std()

    diff = energy - src_mean
    loss = nn.Softplus()(diff).mean()
    # loss = torch.abs(tar_mean-src_mean)
    return loss

############ Energy-adjusted Entropy Minimization (EEM) ############
def energy_adj_ent_min(input, energy_weight, temp=1.0):
    bs = input.size(0)
    ent = softmax_entropy(input)
    eng = energy(input, temp=temp)
    return ent + energy_weight * eng

def logit_similarity(input, cls_weight, thr=0.):
    classwise_virtual_logits = torch.matmul(cls_weight, cls_weight.T)
    #classwise_logit_dir = classwise_virtual_logits / classwise_virtual_logits.norm(dim=-1)

    # simple pseudo-labeling
    prob_all, indices_all = input.softmax(-1).max(-1)
    virtual_logits = classwise_virtual_logits[indices_all]
    conf_indices = prob_all>=thr
    loss = 1 - F.cosine_similarity(input[conf_indices], virtual_logits[conf_indices], dim=-1)
    return loss

def weighted_lcs(input, cls_weight, thr=0.):
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

def covariance(features):
    assert len(features.size()) == 2, "TODO: multi-dimensional feature map covariance"
    n = features.shape[0]
    tmp = torch.ones((1, n), device=features.device) @ features
    cov = (features.t() @ features - (tmp.t() @ tmp) / n) / (n - 1)
    return cov


def coral(cs, ct):
    d = cs.shape[0]
    loss = (cs - ct).pow(2).sum() / (4.0 * d**2)
    return loss


def linear_mmd(ms, mt):
    loss = (ms - mt).pow(2).mean()
    return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, device, epsilon=0.1, reduction=True):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).cpu(), 1
        )
        targets = targets.to(self.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class HLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super().__init__()
        self.temp_factor = temp_factor

    def forward(self, x):

        softmax = F.softmax(x / self.temp_factor, dim=1)
        entropy = -softmax * torch.log(softmax + 1e-6)
        b = entropy.mean()

        return b


## from https://github.com/thuml/TransCal.git
def get_weight(source_train_feature, target_feature, source_val_feature):
    """
    :param source_train_feature: shape [n_tr, d], features from training set
    :param target_feature: shape [n_t, d], features from test set
    :param source_val_feature: shape [n_v, d], features from validation set

    :return:
    """
    # print("-"*30 + "get_weight" + '-'*30)
    n_tr, d = source_train_feature.shape
    n_t, _d = target_feature.shape
    # n_v, _d = source_val_feature.shape
    # print("n_tr: ", n_tr, "n_v: ", n_v, "n_t: ", n_t, "d: ", d)

    if n_tr < n_t:
        sample_index = np.random.choice(n_tr,  n_t, replace=True)
        source_train_feature = source_train_feature[sample_index]
        sample_num = n_t
    elif n_tr > n_t:
        sample_index = np.random.choice(n_t, n_tr, replace=True)
        target_feature = target_feature[sample_index]
        sample_num = n_tr

    combine_feature = np.concatenate((source_train_feature, target_feature))
    combine_label = np.asarray([1] * sample_num + [0] * sample_num, dtype=np.int32)
    domain_classifier = linear_model.LogisticRegression()
    domain_classifier.fit(combine_feature, combine_label)
    domain_out = domain_classifier.predict_proba(source_val_feature)
    weight = domain_out[:, :1] / domain_out[:, 1:]
    return weight

