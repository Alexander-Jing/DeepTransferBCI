import os
import math
from collections import OrderedDict
from copy import deepcopy
from typing import Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.jit
from torch.nn.utils import prune

from easydict import EasyDict as edict
import time

# from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer
from tl.utils.memory_proposed_3 import DropMemoryBank_review_8, HUS, CSTU, FIFO, OnlineBufferInstance
from tl.utils.loss_proposed_1_1 import CE_KL, CE_KL_review_weighted_10, ConsSamples_selection_two_stage_weighted_4_1_review_4_4_3, ConsSamples_selection_two_stage_weighted_4_1_modified, ConsSamples_selection_two_stage_weighted_4_1_review_4_4_4, \
    ConsSamples_selection_two_stage_weighted_4_1_review_4_4_4_feas, ConsSamples_selection_two_stage_weighted_4_1_modified_feas, CE_KL_review_weighted_10_constrastive, ConsSamples_selection_two_stage_weighted_4_1_modified_2, _entropy_samples
from tl.utils.calibration_proposed import DynamicThresholdSelector
from tl.utils.optimizer_proposed import build_optimizer

pruning_methods = {
    "l1_unstructured": prune.l1_unstructured,
    "ln_structured": prune.ln_structured,
    "random_unstructured": prune.random_unstructured,
    "random_structured": prune.random_structured
}


class proposed_TTA(nn.Module):
    def __init__(self, model, paras_optim, capacity, num_classes, bn_alpha, temp_factor,
                 update_frequency, confidence_threshold, uncertainty_threshold, prune_ratio, arch, dataset,
                 enable_robustBN, loss_name, paras_loss, EnergyAlignment, steps=1,
                 episodic=False, memory_bank_type='uhus', use_BN=False, use_buffer=True, fix_pruning_model=True,
                 pruning_strategy='l1_unstructured', pruning_module='conv', calculate_selection_mask=False,
                 category_uniform=True, record=False, metric_name='mean_probs_dropout', update_counter='each', return_type='xy', 
                 num_dropout=20, updating_type="presudo_src", batch_size_online=8, align=True, presudo_source_center=True, mt=0.9, temperature=2, calibrate_probs=False,
                 memory_type='DropMemoryBank_review', memory_review='get_memory_review'):

        super().__init__()
        self.model = model
        self.paras_optim = paras_optim

        self.bn_alpha = bn_alpha
        self.update_frequency = update_frequency
        self.update_counter = update_counter
        self.enable_robustBN = enable_robustBN

        self.arch = arch
        self.dataset = dataset
        self.prune_ratio = prune_ratio
        self.metric_name = metric_name
        self.num_classes = num_classes

        self.num_instance = 0

        self.steps = steps
        self.episodic = episodic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.return_type = return_type
        self.num_dropout = num_dropout
        self.dropout_p = 0.25
        self.updating_type = updating_type
        self.capacity = capacity
        self.align = align
        self.presudo_source_center = presudo_source_center
        self.loss_name = loss_name
        self.mt = mt 
        self.temperature = temperature
        self.EnergyAlignment = EnergyAlignment
        self.calibrate_probs = calibrate_probs
        self.memory_type = memory_type
        self.memory_review = memory_review
        self.show_run_time = False
        self.show_total_run_time = False
        
        # memory
        if self.memory_type in ['DropMemoryBank_review_8']:
            self.memory = DropMemoryBank_review_8(capacity, num_classes, confidence_threshold, uncertainty_threshold,
                                     category_uniform)
        elif self.memory_type in ['HUS']:
            self.memory = HUS(capacity, num_classes, confidence_threshold, uncertainty_threshold,
                                     category_uniform)
        elif self.memory_type in ['CSTU']:
            self.memory = CSTU(capacity, num_classes)
        elif self.memory_type in ['FIFO']:
            self.memory = FIFO(capacity, num_classes, confidence_threshold, uncertainty_threshold,
                                     category_uniform)
        
        self.memory_copy = deepcopy(self.memory)

        
        # optimizer
        if use_BN:
            self.configure_model()
        params, param_names = self.collect_params(use_bn=use_BN)
        self.optimizer = build_optimizer(params, paras_optim)
        try:
            self.optimizer.set_model(self.model)
        except:
            print("optimizer does not have set_model method")

        # loss function
        if not self.paras_optim['two_stage']:
            self.loss_fn = loss_prepare(loss_name, EnergyAlignment)
            self.loss_fn_0 = loss_prepare(loss_name="CE_KL", EnergyAlignment=self.EnergyAlignment)
        else:
            losses = loss_name.split(',')  # if two losses, use a "," to split, e.g. "CE_KL, lcs_ConsSamples"
            self.losses = losses
            self.loss_fn = loss_prepare(loss_name=self.losses[0].strip(), EnergyAlignment=EnergyAlignment)
            self.loss_fn_1 = loss_prepare(loss_name=self.losses[1].strip(), EnergyAlignment=EnergyAlignment)
            # loss for the warm-up stage
            self.loss_fn_0 = loss_prepare(loss_name="CE_KL", EnergyAlignment=self.EnergyAlignment)
            if self.losses[1].strip() in ['ConsSamples_selection_two_stage_weighted_4_1_review_4_4_4_feas']:  
                self.loss_fn_1_0 = loss_prepare(loss_name="ConsSamples_selection_two_stage_weighted_4_1_modified_feas", EnergyAlignment=self.EnergyAlignment)
            else:
                self.loss_fn_1_0 = loss_prepare(loss_name="ConsSamples_selection_two_stage_weighted_4_1_modified_2", EnergyAlignment=self.EnergyAlignment)
        
        
        self.model_state, self.optimizer_state = deepcopy(model.state_dict()), deepcopy(self.optimizer.state_dict())
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"


        self.initial_weights = {name: param.clone().detach() for name, param in model.named_parameters() if
                                param.requires_grad}

        self.confidence_threshold, self.uncertainty_threshold = confidence_threshold, uncertainty_threshold
        self.use_buffer = use_buffer
        self.calculate_selection_mask = calculate_selection_mask
        self.selection_mask = []
        self.online_buffer = OnlineBufferInstance(buffer_size=batch_size_online)
        self.batch_size_online = batch_size_online
        self.dynamic_threshold_selector = DynamicThresholdSelector(num_classes=num_classes, base_threshold=self.confidence_threshold, momentum=0.9, min_threshold=self.EnergyAlignment.min_threshold)
        self.dynamic_threshold_selector.reset()

        if record:
            self.record = {}
        else:
            self.record = None
        

    def configure_model(self):
        if self.enable_robustBN:
            print('using the robust BN mode')
            raise NotImplemented("not implement robust BN")
        else:
            print('using the training BN mode')
            for param in self.model.parameters():  # initially turn off requires_grad for all
                param.requires_grad = False
            for module in self.model.modules():
                if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = True
                    module.momentum = self.bn_alpha

                    module.weight.requires_grad_(True)
                    module.bias.requires_grad_(True)

    def collect_params(self, use_bn=True):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        model = self.model
        params = []
        names = []
        if use_bn:
            for nm, m in model.named_modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:  # weight is scale, bias is shift
                            params.append(p)
                            names.append(f"{nm}.{np}")
        else:
            params = self.model.parameters()
        
        return params, names

    def forward(self, x, sample_test_origin, sqrtRefEA):
        if isinstance(x, dict):
            x = x['img']

        if self.episodic:
            self.reset()

        # inference
        # batch data
        with torch.no_grad():
            
            self.model.eval()
            if self.return_type=='xy':
                fea, out = self.model(x)
            elif self.return_type == 'y':
                out = self.model(x)
            
            prob = torch.softmax(out, dim=1)
            energy = -self.temperature * torch.logsumexp(out / self.temperature, dim=1)
            entropy = _entropy_samples(out)
            weights = entropy * torch.log(1 + torch.exp(energy))

            pseudo_label = torch.argmax(prob, dim=1)
            pseudo_conf = torch.max(prob, dim=1)[0]

        # update memory
        update_model_flag = False
        for i, data in enumerate(sample_test_origin): 
            
            # add to the memory bank (memory buffer)
            p_l = pseudo_label[i].item()
            conf = pseudo_conf[i].item()
            uncertainty = weights[i].item()
            entropy_item = entropy[i].item()
            current_instance = edict(data=data, prediction=p_l, uncertainty=uncertainty,
                                    logit=out[i].detach().clone(), confidence=conf, time_stamp=self.num_instance)  # instance includes the feature, pesudo label, weights, and probability

            # save the instances in the memory based on confidence threshold
            if self.EnergyAlignment.buffer_selefction_type in ["confidence"]:
                if conf >= self.confidence_threshold:
                    self.memory.add_instance(current_instance)
            if self.EnergyAlignment.buffer_selefction_type in ["dynamic_confidence"]: 
                is_selected, _, _, _ = self.dynamic_threshold_selector.check_sample(out[i].detach().clone().unsqueeze(0))
                if is_selected:  # low entropy samples
                    self.memory.add_instance(current_instance)
            elif self.EnergyAlignment.buffer_selefction_type in ["entropy"]:
                C_tensor = torch.tensor(self.num_classes, dtype=torch.float, device=entropy.device)
                max_entropy = torch.log(C_tensor)
                if entropy_item/max_entropy <= self.confidence_threshold:  # low entropy samples
                    self.memory.add_instance(current_instance)
            else:
                if conf >= self.confidence_threshold:
                    self.memory.add_instance(current_instance)

    
            # add to the online memory bank for updating (mini-batch)
            if self.update_counter == 'each':
                self.num_instance += 1
                self.online_buffer.add_data(data)
            else:
                if weights[i].item() >= self.uncertainty_threshold:
                    self.num_instance += 1

            # whether to update the model
            if self.num_instance >= self.batch_size_online and self.num_instance % self.update_frequency == 0:
                update_model_flag = True


        # update model
        if update_model_flag:
            update_time_start = time.time()
            
            if not isinstance(sqrtRefEA, torch.Tensor):
                sqrtRefEA = torch.tensor(sqrtRefEA, device=self.device, dtype=torch.float32)
            else:
                sqrtRefEA = sqrtRefEA.to(self.device, non_blocking=True) # on the device
            
            for _ in range(self.steps):
                self.update_model(self.online_buffer.get_data(), sqrtRefEA)
            
            update_time_end = time.time()
            if self.show_total_run_time:
                print(f"num instance: {self.num_instance}, whole model update time: {update_time_end - update_time_start:.3f} seconds")

        # return outputs
        return fea, out

    @torch.enable_grad()
    def update_model(self, batch_data, sqrtRefEA):
        if self.show_run_time:
            load_data_time_start = time.time()


        # prepare the data for review (buffer data)
        review_data, review_data_logits, review_data_class = self.memory.get_memory()
        mean_entropy, std_entropy = 0.0, 0.0
        if self.EnergyAlignment.gate_type in ['median']:
            _, mean_entropy, std_entropy = self.memory.compute_logits_entropy_median_iqr()
        elif self.EnergyAlignment.gate_type in ['mean']:
            _, mean_entropy, std_entropy = self.memory.compute_logits_entropy()
        
        if len(review_data) == 0:
            # generate empty tensors
            C,H,W = batch_data[0].shape
            review_data = torch.empty((0, C, H, W), device=self.device)
            review_data_logits = torch.empty((0, self.num_classes), device=self.device)
            review_data_class = torch.empty((0,), dtype=torch.long, device=self.device)
        else:
            review_data = torch.stack(review_data).to(self.device, non_blocking=True)
            review_data_logits = torch.stack(review_data_logits).to(self.device, non_blocking=True)
            review_data_class = torch.tensor(review_data_class, device=self.device, dtype=torch.long)
        
        
        if self.show_run_time:
            load_data_time_end = time.time()
            print(f"num instance: {self.num_instance}, load data time: {load_data_time_end - load_data_time_start:.4f} seconds")
        
        if len(batch_data) > 0:
            if self.show_run_time:
                prepare_data_time_start = time.time()

            # prepare the data from current batch and memory
            sup_data = torch.stack(batch_data).cuda().clone()

            # use the EA for alignment
            if self.align:
                sup_data = torch.matmul(sqrtRefEA, sup_data)
                review_data = torch.matmul(sqrtRefEA, review_data)

            
            if self.show_run_time:
                prepare_data_time_end = time.time()
                print(f"num instance: {self.num_instance}, prepare data time: {prepare_data_time_end - prepare_data_time_start:.4f} seconds")

            self.model.train()
            if not self.paras_optim['two_stage']:
                
                if self.paras_optim['name'] == 'Adam':
                    
                    if self.updating_type in ["entropy_review"]:

                        if self.return_type=='xy':
                            _, preds_of_data = self.model(sup_data)
                            if not review_data.shape[0] == 0:
                                _, preds_of_data_review = self.model(review_data)
                            else:
                                preds_of_data_review = torch.empty((0, self.num_classes), device=self.device)
                        elif self.return_type == 'y':
                            preds_of_data = self.model(sup_data)

                        use_buffer_loss = self.memory.get_occupancy() >= int(self.capacity/2)

                        if self.loss_name in ["CE_KL_review_weighted_10_constrastive"]: 
                            if use_buffer_loss:
                                loss = self.loss_fn(preds_of_data, preds_of_data_review, review_data_class, review_data_logits, mean_entropy, std_entropy)
                            else:
                                loss = self.loss_fn_0(preds_of_data)
                        else:
                            loss = self.loss_fn(preds_of_data)

                        self.optimizer.zero_grad(set_to_none=True)

                        loss.backward()

                        self.optimizer.step()
                    
            else:  # two stage updating
                if self.paras_optim['name'] == 'Adam':
                    
                    if self.updating_type in ["entropy_review"]:    
                        if self.show_run_time:
                            time_start = time.time()
                        # first step
                        if self.return_type=='xy':
                            feas_of_data, preds_of_data = self.model(sup_data)
                            if not review_data.shape[0] == 0:
                                _, preds_of_data_review = self.model(review_data)
                            else:
                                preds_of_data_review = torch.empty((0, self.num_classes), device=self.device)
                        elif self.return_type == 'y':
                            preds_of_data = self.model(sup_data)
                        
                        use_buffer_loss = self.memory.get_occupancy() >= int(self.capacity/2)

                        if use_buffer_loss:

                            if self.losses[0].strip() in ["CE_KL_review_weighted_9","CE_KL_review_weighted_10", "CE_KL_review_weighted_10_constrastive"]: 
                                loss = self.loss_fn(preds_of_data, preds_of_data_review, review_data_class, review_data_logits, mean_entropy, std_entropy)
                            else:
                                loss = self.loss_fn(preds_of_data)
                        else: 
                            if self.EnergyAlignment.loss_weight_type in ['buffer_only']:
                                loss = torch.tensor(0.0, device=preds_of_data.device, requires_grad=True)
                            else:
                                loss = self.loss_fn_0(preds_of_data)
                        
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        self.optimizer.step()

                        # zero grad
                        self.optimizer.zero_grad(set_to_none=True)

                        # second step
                        if self.return_type=='xy':
                            feas_of_data_1, preds_of_data_1 = self.model(sup_data)
                            if not review_data.shape[0] == 0:
                                feas_of_data_review_1, preds_of_data_review_1 = self.model(review_data)
                            else:
                                feas_of_data_review_1 = torch.empty((0, feas_of_data_1.shape[1]), device=self.device)
                                preds_of_data_review_1 = torch.empty((0, self.num_classes), device=self.device)
                        elif self.return_type == 'y':
                            preds_of_data_1 = self.model(sup_data)
                        
                        if use_buffer_loss:
                            
                            if self.losses[1].strip() in ["ConsSamples_selection_two_stage_weighted_4_1_review_4_4_3","ConsSamples_selection_two_stage_weighted_4_1_review_4_4_4"]: 
                                loss_1 = self.loss_fn_1(preds_of_data_1, preds_of_data.clone().detach(), preds_of_data_review_1, mean_entropy, std_entropy)              
                            elif self.losses[1].strip() in ["ConsSamples_selection_two_stage_weighted_4_1_review_4_4_4_feas"]: 
                                loss_1 = self.loss_fn_1(preds_of_data_1, preds_of_data.clone().detach(), preds_of_data_review_1, feas_of_data_1, feas_of_data_review_1, mean_entropy, std_entropy)  
                            elif self.losses[1].strip() in ["ConsSamples_selection_two_stage_weighted_4_1_modified_feas"]:
                                loss_1 = self.loss_fn_1(preds_of_data_1, feas_of_data_1, preds_of_data.clone().detach())
                            else:
                                loss_1 = self.loss_fn_1(preds_of_data_1)
                        else:
                            if self.EnergyAlignment.loss_weight_type in ['buffer_only']:
                                loss_1 = torch.tensor(0.0, device=preds_of_data_1.device, requires_grad=True)
                            else:
                                loss_1 = self.loss_fn_1_0(preds_of_data_1, preds_of_data.clone().detach())
                        
                        loss_1.backward()
                        self.optimizer.step()

                        self.optimizer.zero_grad(set_to_none=True)

                        if self.show_run_time:
                            time_end = time.time()
                            print(f"num instance: {self.num_instance}, update time: {time_end - time_start:.2f} seconds")


    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.memory = deepcopy(self.memory_copy)
        self.num_instance = 0
        self.num_consistent = 0
        self.uncertainty_result = dict(uncertainty=[], domain=[])




def loss_prepare(loss_name, EnergyAlignment):
    
    if loss_name == 'CE_KL':
        return CE_KL(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, temp=EnergyAlignment.temp)
    elif loss_name == 'CE_KL_review_weighted_10':
        return CE_KL_review_weighted_10(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, confidence_threshold=EnergyAlignment.confidence_threshold, num_classes=EnergyAlignment.num_class, entropy_threshold=EnergyAlignment.entropy_threshold, ratio=EnergyAlignment.ratio, thre_alpha=EnergyAlignment.thre_alpha)
    elif loss_name == 'CE_KL_review_weighted_10_constrastive':
        return CE_KL_review_weighted_10_constrastive(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, confidence_threshold=EnergyAlignment.confidence_threshold, num_classes=EnergyAlignment.num_class, entropy_threshold=EnergyAlignment.entropy_threshold, ratio=EnergyAlignment.ratio, thre_alpha=EnergyAlignment.thre_alpha, ratio_review=EnergyAlignment.ratio_review)

    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_review_4_4_3':
        return ConsSamples_selection_two_stage_weighted_4_1_review_4_4_3(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, weight_type=EnergyAlignment.weight_type, ratio_reivew=EnergyAlignment.ratio_review, thre_alpha=EnergyAlignment.thre_alpha, loss_weight_type=EnergyAlignment.loss_weight_type, gate_type=EnergyAlignment.gate_type)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_review_4_4_4':
        return ConsSamples_selection_two_stage_weighted_4_1_review_4_4_4(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, weight_type=EnergyAlignment.weight_type, ratio_reivew=EnergyAlignment.ratio_review, thre_alpha=EnergyAlignment.thre_alpha, loss_weight_type=EnergyAlignment.loss_weight_type, gate_type=EnergyAlignment.gate_type)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_modified':
        return ConsSamples_selection_two_stage_weighted_4_1_modified(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, weight_type='entropy_energy')
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_modified_2':
        return ConsSamples_selection_two_stage_weighted_4_1_modified_2(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, weight_type='entropy_energy')
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_review_4_4_4_feas':
        return ConsSamples_selection_two_stage_weighted_4_1_review_4_4_4_feas(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, weight_type=EnergyAlignment.weight_type, ratio_reivew=EnergyAlignment.ratio_review, thre_alpha=EnergyAlignment.thre_alpha, loss_weight_type=EnergyAlignment.loss_weight_type, gate_type=EnergyAlignment.gate_type)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_modified_feas':
        return ConsSamples_selection_two_stage_weighted_4_1_modified_feas(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, weight_type='entropy_energy')
