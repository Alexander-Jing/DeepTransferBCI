import mne
import numpy as np
from pyriemann.utils.mean import mean_riemann
from scipy import linalg
import torch
from torch.nn import functional as F
from copy import deepcopy
import torch.nn as nn
import torch.jit



class TTAMethod(nn.Module):
    def __init__(self, model: nn.Module, config: dict, info: mne.Info):
        super(TTAMethod, self).__init__()
        self.model = model
        self.config = config
        self.info = info
        #self.device = self.model.device
        self.device = next(self.model.parameters()).device
        self.input_buffer = None
        self.buffer_length = self.config.get("buffer_length")
        self.buffer_counter = 0

        self.configure_model()
        self.params, param_names = self.collect_params()
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None
        self.print_amount_trainable_params()

    def forward(self, x):

        if x.shape[0] == 1:  # Only single-sample test-time adaptation allowed
            # add sample to buffer, replace the oldest sample if buffer is full
            if self.input_buffer is None:
                self.input_buffer = x
            elif self.input_buffer.shape[0] < self.buffer_length:
                self.input_buffer = torch.cat([self.input_buffer, x], dim=0)
            else:
                self.input_buffer = torch.cat([self.input_buffer[1:], x], dim=0)

            # update the model if the complete buffer has changed
            if self.buffer_counter == (self.buffer_length - 1):
                outputs = self.forward_and_adapt(self.input_buffer)
                outputs = outputs[-1].unsqueeze(0)
            else:
                outputs = self.forward_sliding_window(self.input_buffer)
                outputs = outputs[-1].unsqueeze(0)

            # increase counter
            self.buffer_counter += 1
            self.buffer_counter %= self.buffer_length

        else:
            outputs = self.forward_and_adapt(x)

        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def forward_sliding_window(self, x):
        return self.model(x)

    def configure_model(self):
        raise NotImplementedError

    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def setup_optimizer(self):
        if self.config["optimizer"] == 'Adam':
            return torch.optim.Adam(self.params,
                                    lr=self.config["optimizer_kwargs"]["lr"],
                                    betas=(self.config["optimizer_kwargs"]["beta"], 0.999),
                                    weight_decay=self.config["optimizer_kwargs"]["weight_decay"])
        else:
            raise NotImplementedError

    def print_amount_trainable_params(self):
        trainable = sum(p.numel() for p in self.params) if len(self.params) > 0 else 0
        total = sum(p.numel() for p in self.model.parameters())
        print(f"#Trainable/total parameters: {trainable}/{total}")

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_states, optimizer_state


class OnlineAlignment(TTAMethod):
    def __init__(self, model: nn.Module, config: dict, info: mne.Info):
        super(OnlineAlignment, self).__init__(model, config, info)

    def forward_sliding_window(self, x):
        return self.forward_and_adapt(x)

    @torch.no_grad()
    def forward_and_adapt(self, x):
        x_aligned = self.align_data(
            x, self.config.get("alignment"),
            self.config.get("averaging_method", "equal"),
            self.config.get("alpha", None))
        outputs = self.model(x_aligned)
        return outputs

    @staticmethod
    def align_data(x, alignment, averaging_method: str, alpha: float = None):
        n_trials = x.shape[0]
        weights = OnlineAlignment._calculate_weights(n_trials, averaging_method, alpha)
        covmats = torch.matmul(x, x.transpose(1, 2)).detach().cpu().numpy()
        if alignment == "euclidean":
            R = np.average(covmats, axis=0, weights=weights)
        elif alignment == "riemann":
            R = mean_riemann(covmats, sample_weight=weights)
        else:
            raise NotImplementedError
        R_op = linalg.inv(linalg.sqrtm(R))
        x_aligned = torch.matmul(
            torch.tensor(R_op, dtype=torch.float32, device=x.device), x)
        return x_aligned

    @staticmethod
    def _calculate_weights(n_trials: int, averaging_method: str, alpha: float = None):
        if averaging_method == "equal":
            weights = None
        elif averaging_method == "linear":
            weights = np.arange(1, n_trials + 1) / n_trials
        elif averaging_method == "ema":
            assert alpha is not None
            if n_trials == 1:
                weights = np.array([1.])
            else:
                first, last = (1 - alpha) ** (n_trials - 1), alpha
                if n_trials >= 3:
                    weights = [alpha * ((1 - alpha) ** i) for i in reversed(
                        range(1, n_trials - 1))]
                    weights = [first] + weights + [last]
                else:
                    weights = [first, last]
                weights = np.array(weights)
        else:
            raise NotImplementedError

        return weights

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)


class AlphaBatchNorm(nn.Module):
    """ Use the source statistics as a prior on the target statistics """

    @staticmethod
    def find_bns(parent, alpha):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            if isinstance(child, nn.BatchNorm2d):
                module = AlphaBatchNorm(child, alpha)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(AlphaBatchNorm.find_bns(child, alpha))

        return replace_mods

    @staticmethod
    def adapt_model(model, alpha):
        replace_mods = AlphaBatchNorm.find_bns(model, alpha)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, alpha):
        assert alpha >= 0 and alpha <= 1

        super().__init__()
        self.layer = layer
        self.layer.eval()
        self.alpha = alpha

        self.norm = nn.BatchNorm2d(self.layer.num_features, affine=False, momentum=1.0)

    def forward(self, input):
        self.norm(input)

        running_mean = ((1 - self.alpha) * self.layer.running_mean + self.alpha * self.norm.running_mean)
        running_var = ((1 - self.alpha) * self.layer.running_var + self.alpha * self.norm.running_var)

        return F.batch_norm(
            input,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0,
            self.layer.eps,
        )


class RobustBN(nn.Module):
    @staticmethod
    def find_bns(parent, alpha):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            if isinstance(child, nn.BatchNorm2d):
                module = RobustBN(child, alpha)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(RobustBN.find_bns(child, alpha))

        return replace_mods

    @staticmethod
    def adapt_model(model, alpha):
        replace_mods = RobustBN.find_bns(model, alpha)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, bn_layer: nn.BatchNorm2d, momentum):
        super(RobustBN, self).__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        if bn_layer.track_running_stats and bn_layer.running_var is not None and bn_layer.running_mean is not None:
            self.register_buffer("source_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("source_var", deepcopy(bn_layer.running_var))
            self.source_num = bn_layer.num_batches_tracked

        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)
        self.eps = bn_layer.eps

    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
        else:
            mean, var = self.source_mean.view(1, -1, 1, 1), self.source_var.view(1, -1, 1, 1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * weight + bias






class EntropyMinimization(TTAMethod):
    def __init__(self, model: nn.Module, config: dict, info: mne.Info):
        super(EntropyMinimization, self).__init__(model, config, info)

    def forward_sliding_window(self, x):
        if self.config.get("alignment", False):
            # align data
            x = OnlineAlignment.align_data(
                x, self.config.get("alignment"),
                self.config.get("averaging_method", "equal"),
                self.config.get("align_alpha", None))
        outputs = self.model(x)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[-1]
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        if self.config.get("alignment", False):
            # align data
            x = OnlineAlignment.align_data(
                x, self.config.get("alignment"),
                self.config.get("averaging_method", "equal"),
                self.config.get("align_alpha", None))

        outputs = self.model(x)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[-1]
        loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs

    def configure_model(self):
        self.model.eval()  # eval mode to avoid using dropout during test-time
        self.model.requires_grad_(True)
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)




class Norm(TTAMethod):
    def __init__(self, model: nn.Module, config: dict, info: mne.Info):
        super(Norm, self).__init__(model, config, info)

    def forward_sliding_window(self, x):
        return self.forward_and_adapt(x)

    @torch.no_grad()
    def forward_and_adapt(self, x):
        if self.config.get("alignment", False):
            x = OnlineAlignment.align_data(
                x, self.config.get("alignment"),
                self.config.get("averaging_method", "equal"),
                self.config.get("align_alpha", None))
        outputs = self.model(x)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[-1]
        return outputs

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

        if self.config.get("norm") == "norm_test":  # BN-1
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
        elif self.config.get("norm") == "norm_alpha":  # BN-0.1
            self.model = AlphaBatchNorm.adapt_model(
                self.model, alpha=self.config.get("alpha"))
        elif self.config.get("norm") == "robust_norm":  # RoTTA
            self.model = RobustBN.adapt_model(
                self.model, alpha=self.config.get("alpha"))
        else:
            raise NotImplementedError