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
    


