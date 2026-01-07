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
from tl.utils.memory_proposed_2 import DropMemoryBank, DropMemoryBank_review, DropMemoryBank_review_1, OnlineBuffer, OnlineBufferInstance
from tl.utils.memory_proposed_3 import DropMemoryBank_review_2, DropMemoryBank_review_3, DropMemoryBank_review_4, DropMemoryBank_review_5, DropMemoryBank_review_6, DropMemoryBank_review_7, DropMemoryBank_review_8
from tl.utils.loss_proposed_1 import MemorySoftplusEnergyAlignment, CE_MDR, PresudoLabelMemorySoftplusEnergyAlignment, MemorySoftplusEnergyWeightedAlignment, \
    MemorySoftplusEnergyRatioSortedAlignment, MemorySoftplusEnergyFeatureWeightedAlignment, MemorySoftplusEnergyWeightedAlignmentMDR, \
        CaliE_MDR, CaliE_UKL, CE_KL, CaliE_KL, EnergyEntropy_selected, EnergyEntropy_selected_all, EnergyEntropy_selected_align, PresudoLabelEMA, PresudoLabelEMA_selection, \
        PresudoLabelEMA_energy, PresudoLabelEMA_symmetric, PresudoLabelEMA_lcs, EntropyMDREMA_lcs, CaliE_MDR_lcs, CaliE_MDR_lcs_selection, CaliE_MDR_lcs_cons, CaliE_MDR_lcs_ConsSamples, \
        CaliE_MDR_lcs_ConsSamplesFea, CE_KL_ConsSamplesFea, PresudoLabelEMA_SampleCons, ConsSamples_lcs, ConsSamples_weighted, ConsSamples, ConsSamples_selection, ConsSamples_selection_1, \
        ConsSamples_selection_2, ConsSamples_selection_dropout, _entropy_samples, ConsSamples_selection_distillation, Weighted_ConsSamples_selection_distillation, CE_KL_lcs_ConsSamples, CE_KL_lcs_ConsSamples_selection, \
        ConsSamples_selection_two_stage, ConsSamples_selection_two_stage_weighted, ConsSamples_selection_two_stage_weighted_1, ConsSamples_selection_two_stage_weighted_2, ConsSamples_selection_two_stage_weighted_3, ConsSamples_selection_two_stage_weighted_4, ConsSamples_selection_two_stage_weighted_4_1, \
            ConsSamples_selection_two_stage_weighted_4_1_double, ConsSamples_selection_two_stage_weighted_4_1_double_1, CE_KL_review, CE_KL_review_weighted, CE_KL_review_weighted_1, CE_KL_review_weighted_2, CE_KL_review_weighted_3, CE_KL_review_weighted_4, CE_KL_review_weighted_5, ConsSamples_selection_two_stage_weighted_4_1_review, \
            ConsSamples_selection_two_stage_weighted_4_1_review_1, ConsSamples_selection_two_stage_weighted_4_1_review_2, ConsSamples_selection_two_stage_weighted_4_1_review_3, CE_KL_review_weighted_3_1, ConsSamples_selection_two_stage_weighted_4_1_review_4, ConsSamples_selection_two_stage_weighted_4_1_review_4_1, CE_KL_review_weighted_3_2
from tl.utils.loss_proposed_1_1 import CE_KL_review_3, CE_KL_review_weighted_6, CE_KL_review_weighted_7, ConsSamples_selection_two_stage_weighted_4_1_review_4_2, CE_KL_review_weighted_8, ConsSamples_selection_two_stage_weighted_4_1_review_4_3, CE_KL_review_weighted_9, ConsSamples_selection_two_stage_weighted_4_1_review_4_4, ConsSamples_selection_two_stage_weighted_4_1_review_4_4_1, \
     ConsSamples_selection_two_stage_weighted_4_1_review_4_4_2, CE_KL_review_weighted_10
from tl.utils.calibration_proposed import CalibratedPseudoLabels, DynamicThresholdSelector
from tl.utils.optimizer_proposed import build_optimizer
from tl.utils.network import backbone_net
from tl.utils.adaptiveLR_proposed import AdaptiveLRScheduler, AdaptiveLRScheduler_1

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
        
        # memory
        if self.memory_type in ['DropMemoryBank_review']:
            self.memory = DropMemoryBank_review(capacity, num_classes, confidence_threshold, uncertainty_threshold,
                                     category_uniform)
        elif self.memory_type in ['DropMemoryBank_review_1']:
            self.memory = DropMemoryBank_review_1(capacity, num_classes, confidence_threshold, uncertainty_threshold,
                                     category_uniform)
        elif self.memory_type in ['DropMemoryBank_review_2']:
            self.memory = DropMemoryBank_review_2(capacity, num_classes, confidence_threshold, uncertainty_threshold,
                                     category_uniform)
        elif self.memory_type in ['DropMemoryBank_review_3']:
            self.memory = DropMemoryBank_review_3(capacity, num_classes, confidence_threshold, uncertainty_threshold,
                                     category_uniform)
        elif self.memory_type in ['DropMemoryBank_review_4']:
            self.memory = DropMemoryBank_review_4(capacity, num_classes, confidence_threshold, uncertainty_threshold,
                                     category_uniform)
        elif self.memory_type in ['DropMemoryBank_review_5']:
            self.memory = DropMemoryBank_review_5(capacity, num_classes, confidence_threshold, uncertainty_threshold,
                                     category_uniform)
        elif self.memory_type in ['DropMemoryBank_review_6']:
            self.memory = DropMemoryBank_review_6(capacity, num_classes, confidence_threshold, uncertainty_threshold,
                                     category_uniform)
        elif self.memory_type in ['DropMemoryBank_review_7']:
            self.memory = DropMemoryBank_review_7(capacity, num_classes, confidence_threshold, uncertainty_threshold,
                                     category_uniform)
        elif self.memory_type in ['DropMemoryBank_review_8']:
            self.memory = DropMemoryBank_review_8(capacity, num_classes, confidence_threshold, uncertainty_threshold,
                                     category_uniform)
        
        self.memory_copy = deepcopy(self.memory)

        # add the prototypes to the memory buffer
        if self.updating_type in ["cls_proto"]:
            cls_weight = self.model[1].fc.weight.data.clone()
            for i in range(num_classes):
                # normalization 
                normalized_weight = F.normalize(cls_weight[i], dim=0)
                # add instance
                current_instance = edict(data=normalized_weight.cpu(), prediction=i, uncertainty=0.0,
                                     confidence=1.0)  # instance includes the feature, pesudo label, weights, and probability
                self.memory.add_instance(current_instance) # add to the memory bank
            self.prototypes = cls_weight  # save the prototypes for ema based updating
        
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
        else:
            losses = loss_name.split(',')  # if two losses, use a "," to split, e.g. "CE_KL, lcs_ConsSamples"
            self.losses = losses
            self.loss_fn = loss_prepare(loss_name=losses[0].strip(), EnergyAlignment=EnergyAlignment)
            self.loss_fn_1 = loss_prepare(loss_name=losses[1].strip(), EnergyAlignment=EnergyAlignment)
            if self.losses[1].strip() in ['ConsSamples_selection_two_stage_adaptiveLR', 'ConsSamples_selection_two_stage_adaptiveLR_1','ConsSamples_selection_two_stage_adaptiveLR_2']:
                if self.losses[1].strip() == 'ConsSamples_selection_two_stage_adaptiveLR':
                    self.lr_scheduler = AdaptiveLRScheduler(self.optimizer, base_lr=self.paras_optim.lr)
                if self.losses[1].strip() == 'ConsSamples_selection_two_stage_adaptiveLR_1': 
                    self.lr_scheduler = AdaptiveLRScheduler_1(self.optimizer, base_lr=self.paras_optim.lr)
                if self.losses[1].strip() == 'ConsSamples_selection_two_stage_adaptiveLR_2': 
                    self.lr_scheduler = AdaptiveLRScheduler_1(self.optimizer, base_lr=self.paras_optim.lr)
            # need of prototypes
            if self.losses[1].strip() in ["ConsSamples_selection_two_stage_weighted_4_1_double_1"]:
                cls_weight = self.model[1].fc.weight.data.clone()
                for i in range(num_classes):
                    # normalization 
                    normalized_weight = F.normalize(cls_weight[i], dim=0)
                    # add instance
                    current_instance = edict(data=normalized_weight.cpu(), prediction=i, uncertainty=0.0,
                                        confidence=1.0)  # instance includes the feature, pesudo label, weights, and probability
                    self.memory.add_instance(current_instance) # add to the memory bank
                self.prototypes = cls_weight  # save the prototypes for ema based updating

        # state copy
        if self.updating_type in ["ema"]: 
            self.model_ema = deepcopy(self.model)
        if self.updating_type in ["entropy_ema"]: 
            self.model_ema = deepcopy(self.model)
        if self.updating_type in ["entropy_ensamble"]:
            self.model_initial = deepcopy(self.model) 
            self.initial_reference = {}
            for name, param in self.model_initial.named_parameters():
                self.initial_reference[name] = param.data.clone()

            current_state = self.model.state_dict()
            initial_state = self.model_initial.state_dict()

            # 详细比较每一个键值对
            for key in current_state:
                if not torch.allclose(current_state[key], initial_state[key]):
                    print(f"初始：状态不匹配: {key}")
        
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
        self.probs_calibrated =  CalibratedPseudoLabels(n_classes=self.num_classes, device=self.device)
        self.dynamic_threshold_selector = DynamicThresholdSelector(num_classes=num_classes, base_threshold=self.confidence_threshold, momentum=0.9, min_threshold=self.EnergyAlignment.min_threshold)
        self.dynamic_threshold_selector.reset()

        if record:
            self.record = {}
        else:
            self.record = None
        
        # dropout model 
        self.model_dropout = get_backbone_dropout(self.model, self.dropout_p)

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
            if self.updating_type in ["ema","entropy_ema"]:
                self.model_ema.eval()
                if self.return_type=='xy':
                    fea, out = self.model_ema(x)
                elif self.return_type == 'y':
                    out = self.model_ema(x)
            else:
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

            # add to the memory bank to calcuate the prototypes
            if not self.paras_optim['two_stage']:
                if self.updating_type in ["cls_proto"]:
                    p_l = pseudo_label[i].item()
                    conf = pseudo_conf[i].item()
                    uncertainty = weights[i].item()
                    _fea = fea[i].detach().clone()
                    _fea = F.normalize(_fea, dim=0)  # normalize
                    current_instance = edict(data=_fea.cpu(), prediction=p_l, uncertainty=uncertainty,
                                            confidence=conf)  # instance includes the feature, pesudo label, weights, and probability
                    self.online_buffer.add_instance(current_instance) # add to the memory bank
            else:
                if self.losses[1].strip() in ["ConsSamples_selection_two_stage_weighted_4_1_double_1"]:
                    p_l = pseudo_label[i].item()
                    conf = pseudo_conf[i].item()
                    uncertainty = weights[i].item()
                    _fea = fea[i].detach().clone()
                    _fea = F.normalize(_fea, dim=0)  # normalize
                    current_instance = edict(data=_fea.cpu(), prediction=p_l, uncertainty=uncertainty,
                                            confidence=conf)  # instance includes the feature, pesudo label, weights, and probability
                    self.online_buffer.add_instance(current_instance) # add to the memory bank
                
                if self.updating_type in ["entropy_review", "entropy_review_1", "entropy_review_2"]:
                    p_l = pseudo_label[i].item()
                    conf = pseudo_conf[i].item()
                    uncertainty = weights[i].item()
                    entropy_item = entropy[i].item()
                    current_instance = edict(data=data, prediction=p_l, uncertainty=uncertainty,
                                            logit=out[i].detach().clone(), confidence=conf, time_stamp=self.num_instance)  # instance includes the feature, pesudo label, weights, and probability
                    self.online_buffer.add_instance(current_instance) # add to the memory bank
                    # prob_item = prob[i].detach().clone()
                    # top2 = torch.topk(prob_item, 2).values

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

            

            # add to the online memory bank for updating
            if self.update_counter == 'each':
                self.num_instance += 1
                self.online_buffer.add_data(data)
            else:
                if weights[i].item() >= self.uncertainty_threshold:
                    self.num_instance += 1

            # whether to update the model
            if self.updating_type not in ["entropy_review_2"]:
                if self.num_instance >= self.batch_size_online and self.num_instance % self.update_frequency == 0:
                    update_model_flag = True
            else:
                if self.num_instance >= self.batch_size_online and self.memory.get_occupancy() >= self.num_classes*self.batch_size_online and self.num_instance % self.update_frequency == 0:
                    update_model_flag = True

        
        # update model
        if update_model_flag:
            update_time_start = time.time()
            
            for _ in range(self.steps):
                self.update_model(self.online_buffer.get_data(), sqrtRefEA)
            
            update_time_end = time.time()
            if self.show_run_time:
                print(f"num instance: {self.num_instance}, whole model update time: {update_time_end - update_time_start:.3f} seconds")

        # return outputs
        return fea, out

    @torch.enable_grad()
    def update_model(self, batch_data, sqrtRefEA):
        
        load_data_time_start = time.time()
        if not self.paras_optim['two_stage']:
            loss_fn = self.loss_fn
        else:
            loss_fn = self.loss_fn
            loss_fn_1 = self.loss_fn_1
            if self.updating_type in ["entropy_review", "entropy_review_1", "entropy_review_2"]:
                loss_fn_0 = loss_prepare(loss_name="CE_KL", EnergyAlignment=self.EnergyAlignment)
                loss_fn_1_0 = loss_prepare(loss_name="ConsSamples_selection_two_stage_weighted_4_1", EnergyAlignment=self.EnergyAlignment)

        # prepare the data from current batch and memory
        if not self.paras_optim['two_stage']:
            if self.updating_type in ["cls_proto"]:
                prototypes = deepcopy(self.memory.get_prototypes(ratio=self.EnergyAlignment.ratio))
                self.prototypes = self.mt * self.prototypes + (1-self.mt) * prototypes.cuda()
        else:
            if self.losses[1].strip() in ["ConsSamples_selection_two_stage_weighted_4_1_double_1"]:
                prototypes = deepcopy(self.memory.get_prototypes(ratio=1.0))
                self.prototypes = self.mt * self.prototypes + (1-self.mt) * prototypes.cuda()

        if self.updating_type in ["entropy_review", "entropy_review_1", "entropy_review_2"]:
            if self.memory_review in ['get_memory_review']:
                review_data, review_data_logits, review_data_class = deepcopy(self.memory.get_memory_review(self.batch_size_online))
            elif self.memory_review in ['get_memory_review_1']:
                review_data, review_data_logits, review_data_class = deepcopy(self.memory.get_memory_review_1(self.batch_size_online))
            elif self.memory_review in ['get_memory']:
                #review_data, review_data_logits, review_data_class = deepcopy(self.memory.get_memory())
                #_, mean_entropy, std_entropy = deepcopy(self.memory.compute_logits_entropy())
                review_data, review_data_logits, review_data_class = self.memory.get_memory()
                if self.EnergyAlignment.gate_type in ['median']:
                    _, mean_entropy, std_entropy = self.memory.compute_logits_entropy_median_iqr()
                elif self.EnergyAlignment.gate_type in ['mean']:
                    _, mean_entropy, std_entropy = self.memory.compute_logits_entropy()
                else:
                    _, mean_entropy, std_entropy = self.memory.compute_logits_entropy()

            if len(review_data) == 0:
                # generate empty tensors
                C,H,W = batch_data[0].shape
                review_data = torch.empty((0, C, H, W)).cuda()
                review_data_logits = torch.empty((0, self.num_classes)).cuda()
                review_data_class = torch.empty((0,), dtype=torch.long).cuda()
            else:
                review_data = torch.stack(review_data).cuda()
                review_data_logits = torch.stack(review_data_logits).cuda()
                review_data_class = torch.tensor(review_data_class).cuda()
            
            review_data, review_data_logits, review_data_class = review_data.clone(), review_data_logits.clone(), review_data_class.clone()
        
        load_data_time_end = time.time()
        if self.show_run_time:
            print(f"num instance: {self.num_instance}, load data time: {load_data_time_end - load_data_time_start:.4f} seconds")
        
        if len(batch_data) > 0:
            
            prepare_data_time_start = time.time()

            # prepare the data from current batch and memory
            sup_data = torch.stack(batch_data).cuda().clone()

            if self.updating_type in ["presudo_src"]:
                pre_source_data = torch.stack(pre_source_data)
                pre_source_data = pre_source_data.cuda(non_blocking=True)

            # use the EA for alignment
            if self.align:
                sqrtRefEA = torch.tensor(sqrtRefEA).float()
                sqrtRefEA = sqrtRefEA.cuda(non_blocking=True)
                sup_data = torch.matmul(sqrtRefEA, sup_data)
                # sup_data = sup_data.permute(1, 2, 0, 3)
                if self.updating_type in ["presudo_src"]:
                    pre_source_data = torch.matmul(sqrtRefEA, pre_source_data)
                    # pre_source_data = pre_source_data.permute(1, 2, 0, 3)
                if self.updating_type in ["entropy_review", "entropy_review_1", "entropy_review_2"]:
                   review_data = torch.matmul(sqrtRefEA, review_data)

            
            if self.updating_type in ["presudo_src"]:
                if self.return_type=='xy':
                    feas_of_data, preds_of_data = self.model(pre_source_data)
                elif self.return_type == 'y':
                    preds_of_data = self.model(pre_source_data)
                pre_source_labels = torch.tensor(pre_source_labels).cuda()
                class_centers, unique_labels, missing_classes_flag = self.compute_class_centers(feas_of_data, pre_source_labels)
                class_centers = class_centers.detach()
                
            prepare_data_time_end = time.time()
            if self.show_run_time:
                print(f"num instance: {self.num_instance}, prepare data time: {prepare_data_time_end - prepare_data_time_start:.4f} seconds")

            self.model.train()
            if not self.paras_optim['two_stage']:
                
                if self.paras_optim['name'] == 'Adam':
                    
                    if self.updating_type == "entropy":

                        if self.return_type=='xy':
                            feas_of_data, preds_of_data = self.model(sup_data)
                        elif self.return_type == 'y':
                            preds_of_data = self.model(sup_data)

                        if self.loss_name in ['ConsSamples_selection_dropout']:
                            logits_dropout = self.eval_dropout_1(sup_data)[self.metric_name]

                        if self.loss_name in ["PresudoLabelEMA_lcs", "EntropyMDREMA_lcs", "CaliE_MDR_lcs", "CaliE_MDR_lcs_selection", \
                                            "CaliE_MDR_lcs_cons", "CaliE_MDR_lcs_ConsSamples", "ConsSamples_lcs", "CE_KL_lcs_ConsSamples", "CE_KL_lcs_ConsSamples_selection"]:
                            cls_weight = self.model[1].fc.weight.data.clone()
                        
                        if self.loss_name in ["PresudoLabelEMA_lcs", "EntropyMDREMA_lcs", "CaliE_MDR_lcs", "CaliE_MDR_lcs_selection", \
                                            "CaliE_MDR_lcs_cons", "CaliE_MDR_lcs_ConsSamples", "ConsSamples_lcs", "CE_KL_lcs_ConsSamples", "CE_KL_lcs_ConsSamples_selection"]:
                            loss = loss_fn(preds_of_data, cls_weight)
                        elif self.loss_name in ["CaliE_MDR_lcs_ConsSamplesFea", "CE_KL_ConsSamplesFea"]:
                            loss = loss_fn(preds_of_data, feas_of_data)
                        elif self.loss_name in ['ConsSamples_selection_dropout']:
                            loss = loss_fn(preds_of_data, logits_dropout)
                        elif self.loss_name in ["ConsSamples_selection_distillation"]:
                            loss = loss_fn(preds_of_data, feas_of_data, self.prototypes)
                        elif self.loss_name in ["ConsSamples_selection_two_stage_weighted","ConsSamples_selection_two_stage_weighted_1","ConsSamples_selection_two_stage_weighted_2",\
                                                "ConsSamples_selection_two_stage_weighted_3","ConsSamples_selection_two_stage_weighted_4","ConsSamples_selection_two_stage_weighted_4_1"]:
                            loss = loss_fn(preds_of_data, preds_of_data.clone().detach())
                        else:
                            loss = loss_fn(preds_of_data)

                        self.optimizer.zero_grad()

                        loss.backward()

                        self.optimizer.step()
                    
                    elif self.updating_type in ["cls_proto"]:
                        
                        if self.return_type=='xy':
                            feas_of_data, preds_of_data = self.model(sup_data)
                        elif self.return_type == 'y':
                            preds_of_data = self.model(sup_data)

                        if self.loss_name in ["ConsSamples_selection_distillation", "Weighted_ConsSamples_selection_distillation"]:
                            loss = loss_fn(preds_of_data, feas_of_data, self.prototypes)
                        else:
                            loss = loss_fn(preds_of_data)

                        self.optimizer.zero_grad()

                        loss.backward()

                        self.optimizer.step()
            else:  # two stage updating
                if self.paras_optim['name'] == 'Adam':
                    if self.updating_type in ["entropy", "entropy_ensamble", "entropy_ema"]:    
                        # first step
                        if self.return_type=='xy':
                            feas_of_data, preds_of_data = self.model(sup_data)
                        elif self.return_type == 'y':
                            preds_of_data = self.model(sup_data)
                        
                        if self.losses[1].strip() in ["ConsSamples_selection_two_stage_adaptiveLR","ConsSamples_selection_two_stage_adaptiveLR_1"]:  
                            _newLR = self.lr_scheduler.update_lr_entropy(preds_of_data.clone().detach())

                        loss = loss_fn(preds_of_data)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        # zero grad
                        self.optimizer.zero_grad()

                        if self.losses[1].strip() in ["ConsSamples_selection_two_stage_adaptiveLR_2"]:
                            original_lr = self.optimizer.param_groups[0]['lr']  # save the original lr
                            _newLR = self.lr_scheduler.update_lr_entropy(preds_of_data.clone().detach())

                        # second step
                        if self.return_type=='xy':
                            feas_of_data_1, preds_of_data_1 = self.model(sup_data)
                        elif self.return_type == 'y':
                            preds_of_data_1 = self.model(sup_data)
                        
                        if self.losses[1].strip() in ["ConsSamples_selection_two_stage_weighted","ConsSamples_selection_two_stage_weighted_1","ConsSamples_selection_two_stage_weighted_2","ConsSamples_selection_two_stage_weighted_3","ConsSamples_selection_two_stage_weighted_4","ConsSamples_selection_two_stage_weighted_4_1"]: 
                            loss_1 = loss_fn_1(preds_of_data_1, preds_of_data.clone().detach())
                        elif self.losses[1].strip() in ["ConsSamples_selection_two_stage_weighted_4_1_double"]:
                            loss_1 = loss_fn_1(preds_of_data_1, preds_of_data.clone().detach(), feas_of_data_1)
                        elif self.losses[1].strip() in ["ConsSamples_selection_two_stage_weighted_4_1_double_1"]:
                            loss_1 = loss_fn_1(preds_of_data_1, preds_of_data.clone().detach(), feas_of_data_1, self.prototypes)
                        else:
                            loss_1 = loss_fn_1(preds_of_data_1)
                        loss_1.backward()
                        self.optimizer.step()

                        self.optimizer.zero_grad()

                        if self.losses[1].strip() in ["ConsSamples_selection_two_stage_adaptiveLR_2"]:
                            for param_group in self.optimizer.param_groups:
                               param_group['lr'] = original_lr
                    

                    
                    if self.updating_type in ["entropy_review"]:    
                        time_start = time.time()
                        # first step
                        if self.return_type=='xy':
                            feas_of_data, preds_of_data = self.model(sup_data)
                            if not review_data.shape[0] == 0:
                                feas_of_data_review, preds_of_data_review = self.model(review_data)
                            else:
                                feas_of_data_review = torch.empty((0, feas_of_data.shape[1])).cuda()
                                preds_of_data_review = torch.empty((0, self.num_classes)).cuda()
                        elif self.return_type == 'y':
                            preds_of_data = self.model(sup_data)
                        
                        if self.memory.get_occupancy() >= int(self.capacity/2):

                            if self.losses[0].strip() in ["CE_KL_review"]: 
                                loss = loss_fn(preds_of_data, preds_of_data_review, review_data_class)
                            elif self.losses[0].strip() in ["CE_KL_review_weighted","CE_KL_review_weighted_1","CE_KL_review_weighted_2","CE_KL_review_weighted_3", "CE_KL_review_3", 
                                                            "CE_KL_review_weighted_3_1", "CE_KL_review_weighted_3_2", "CE_KL_review_weighted_4", "CE_KL_review_weighted_5","CE_KL_review_weighted_6","CE_KL_review_weighted_7"]: 
                                loss = loss_fn(preds_of_data, preds_of_data_review, review_data_class, review_data_logits)
                            elif self.losses[0].strip() in ["CE_KL_review_weighted_8"]: 
                                loss = loss_fn(preds_of_data, preds_of_data_review, review_data_class, review_data_logits, mean_entropy)
                            elif self.losses[0].strip() in ["CE_KL_review_weighted_9","CE_KL_review_weighted_10"]: 
                                loss = loss_fn(preds_of_data, preds_of_data_review, review_data_class, review_data_logits, mean_entropy, std_entropy)
                            else:
                                loss = loss_fn(preds_of_data)
                        else: 
                            loss = loss_fn_0(preds_of_data)
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        # zero grad
                        self.optimizer.zero_grad()

                        # second step
                        if self.return_type=='xy':
                            feas_of_data_1, preds_of_data_1 = self.model(sup_data)
                            if not review_data.shape[0] == 0:
                                feas_of_data_review_1, preds_of_data_review_1 = self.model(review_data)
                            else:
                                feas_of_data_review_1 = torch.empty((0, feas_of_data.shape[1])).cuda()
                                preds_of_data_review_1 = torch.empty((0, self.num_classes)).cuda()
                        elif self.return_type == 'y':
                            preds_of_data_1 = self.model(sup_data)
                        
                        if self.memory.get_occupancy() >= int(self.capacity/2):
                            if self.losses[1].strip() in ["ConsSamples_selection_two_stage_weighted","ConsSamples_selection_two_stage_weighted_1","ConsSamples_selection_two_stage_weighted_2","ConsSamples_selection_two_stage_weighted_3","ConsSamples_selection_two_stage_weighted_4","ConsSamples_selection_two_stage_weighted_4_1"]: 
                                loss_1 = loss_fn_1(preds_of_data_1, preds_of_data.clone().detach())
                            elif self.losses[1].strip() in ["ConsSamples_selection_two_stage_weighted_4_1_double"]:
                                loss_1 = loss_fn_1(preds_of_data_1, preds_of_data.clone().detach(), feas_of_data_1)
                            elif self.losses[1].strip() in ["ConsSamples_selection_two_stage_weighted_4_1_double_1"]:
                                loss_1 = loss_fn_1(preds_of_data_1, preds_of_data.clone().detach(), feas_of_data_1, self.prototypes)
                            elif self.losses[1].strip() in ["ConsSamples_selection_two_stage_weighted_4_1_review","ConsSamples_selection_two_stage_weighted_4_1_review_1","ConsSamples_selection_two_stage_weighted_4_1_review_4","ConsSamples_selection_two_stage_weighted_4_1_review_4_1","ConsSamples_selection_two_stage_weighted_4_1_review_4_2"]: 
                                loss_1 = loss_fn_1(preds_of_data_1, preds_of_data.clone().detach(), preds_of_data_review_1) 
                            elif self.losses[1].strip() in ["ConsSamples_selection_two_stage_weighted_4_1_review_4_3"]: 
                                loss_1 = loss_fn_1(preds_of_data_1, preds_of_data.clone().detach(), preds_of_data_review_1, mean_entropy) 
                            elif self.losses[1].strip() in ["ConsSamples_selection_two_stage_weighted_4_1_review_4_4","ConsSamples_selection_two_stage_weighted_4_1_review_4_4_1","ConsSamples_selection_two_stage_weighted_4_1_review_4_4_2"]: 
                                loss_1 = loss_fn_1(preds_of_data_1, preds_of_data.clone().detach(), preds_of_data_review_1, mean_entropy, std_entropy) 
                            elif self.losses[1].strip() in ["ConsSamples_selection_two_stage_weighted_4_1_review_2", "ConsSamples_selection_two_stage_weighted_4_1_review_3"]: 
                                loss_1 = loss_fn_1(preds_of_data_1, preds_of_data.clone().detach(), review_data_logits)                     
                            else:
                                loss_1 = loss_fn_1(preds_of_data_1)
                        else:
                            loss_1 = loss_fn_1_0(preds_of_data_1, preds_of_data.clone().detach())
                        
                        loss_1.backward()
                        self.optimizer.step()

                        self.optimizer.zero_grad()

                        if self.losses[1].strip() in ["ConsSamples_selection_two_stage_adaptiveLR_2"]:
                            for param_group in self.optimizer.param_groups:
                               param_group['lr'] = original_lr

                        time_end = time.time()
                        if self.show_run_time:
                            print(f"num instance: {self.num_instance}, update time: {time_end - time_start:.2f} seconds")

    def check_updates(self):
        is_update = is_updated(self.feature_extractor, self.feature_extractor_init)
        print(f"Feature_extractor is updated: {is_update}")

        is_update = is_updated(self.feature_extractor_prune, self.feature_extractor_prune_init)
        print(f"Feature_extractor_prune is updated: {is_update}")

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.memory = deepcopy(self.memory_copy)
        self.num_instance = 0
        self.num_consistent = 0
        self.model_dropout = get_backbone_dropout(self.model, self.dropout_p)
        self.uncertainty_result = dict(uncertainty=[], domain=[])

    def eval_dropout(self, x, logits, probs, preds):
        num_dropout = self.num_dropout
        self.model_dropout.eval()

        # with drop inference
        x_repeat = torch.repeat_interleave(x, num_dropout, dim=0)
        if self.return_type=='xy':
            _, logits_dropout = self.model_dropout(x_repeat)  # num_dropout batch_size * num_classes
        elif self.return_type == 'y':
            logits_dropout = self.model_dropout(x_repeat)  # num_dropout batch_size * num_classes
        
        logits_dropout = logits_dropout.view(len(x), num_dropout, -1)
        probs_dropout = torch.softmax(logits_dropout, dim=2)
        preds_dropout = torch.argmax(probs_dropout, dim=2)  # batch_size * num_dropout

        mean_probs_dropout = torch.mean(probs_dropout, dim=1)  # batch_size * num_classes
        std_probs_dropout = torch.std(probs_dropout, dim=1)  # batch_size * num_classes
        mean_based_preds = torch.argmax(mean_probs_dropout, dim=1)
        preds = mean_based_preds
        mean_probs_dropout = mean_probs_dropout[torch.arange(len(preds)), preds]  # batch_size
        std_probs_dropout = std_probs_dropout[torch.arange(len(preds)), preds]  # batch_size

        mean_logits_dropout = torch.mean(logits_dropout, dim=1)  # batch_size * num_classes
        std_logits_dropout = torch.std(logits_dropout, dim=1)  # batch_size * num_classes
        mean_logits_dropout = mean_logits_dropout[torch.arange(len(preds)), preds]
        std_logits_dropout = std_logits_dropout[torch.arange(len(preds)), preds]

        max_logits = logits[torch.arange(len(preds)), preds]
        max_probs = probs[torch.arange(len(preds)), preds]
        change_logits = (mean_logits_dropout - max_logits).abs()
        change_probs = (mean_probs_dropout - max_probs).abs()

        consistency = (preds.unsqueeze(1) == preds_dropout).mean(dim=1, dtype=float)  # batch_size * num_dropout
        result = {'mean_logits_dropout': mean_logits_dropout, 'std_logits_dropout': std_logits_dropout,
                  'mean_probs_dropout': mean_probs_dropout, 'std_probs_dropout': std_probs_dropout,
                  'change_logits': change_logits, 'change_probs': change_probs, 'consistency': consistency}

        return result

    def eval_dropout_1(self, x):
        num_dropout = self.num_dropout
        self.model_dropout.eval()

        # with drop inference
        x_repeat = torch.repeat_interleave(x, num_dropout, dim=0)
        if self.return_type=='xy':
            _, logits_dropout = self.model_dropout(x_repeat)  # num_dropout batch_size * num_classes
        elif self.return_type == 'y':
            logits_dropout = self.model_dropout(x_repeat)  # num_dropout batch_size * num_classes
        
        logits_dropout = logits_dropout.view(len(x), num_dropout, -1)
        probs_dropout = torch.softmax(logits_dropout, dim=2)
        preds_dropout = torch.argmax(probs_dropout, dim=2)  # batch_size * num_dropout

        mean_probs_dropout = torch.mean(probs_dropout, dim=1)  # batch_size * num_classes
        std_probs_dropout = torch.std(probs_dropout, dim=1)  # batch_size * num_classes
        
        result = {'mean_probs_dropout': mean_probs_dropout, 'std_probs_dropout': std_probs_dropout}

        return result

    def compute_class_centers(self, feas_of_data, pre_source_label):
        # Get unique classes and mapping indices
        unique_labels, inverse_indices, counts = torch.unique(
            pre_source_label, 
            return_inverse=True, 
            return_counts=True,
            sorted=True  # Ensure sorted output
        )
        
        # Initialize accumulator for feature sums
        class_centers = torch.zeros(
            len(unique_labels), 
            feas_of_data.size(1), 
        ).cuda()
        
        # Use scatter_add to accumulate features per class
        expanded_indices = inverse_indices.view(-1, 1).expand(-1, feas_of_data.size(1))
        class_centers.scatter_add_(0, expanded_indices, feas_of_data)
        
        # Convert counts to float tensor for division
        counts_float = counts.float().view(-1, 1)  # Reshape for broadcasting
        
        # Compute mean features (divide sum by count per class)
        class_centers /= counts_float
        
        # Check if any expected classes are missing
        missing_classes_flag = False
        if len(unique_labels) < self.num_classes:
            missing_classes_flag = True
    
        return class_centers, unique_labels, missing_classes_flag

    def update_ema_variables(self, ema_model, model, alpha_teacher):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1.0 - alpha_teacher) * param[:].data[:]
        return ema_model
    
    def update_with_initial(self, current_model, initial_model, alpha):
        for current_param, initial_param in zip(current_model.parameters(), initial_model.parameters()):
            # current_param = α * current_param + (1-α) * initial_param
            current_param.data.mul_(alpha).add_(initial_param.data, alpha=(1.0 - alpha))
        return current_model
    """def update_with_initial(self, current_model, initial_model, alpha):
        # 添加调试信息
        # print(f"Updating with alpha={alpha}")
        with torch.no_grad():
            for current_param, initial_param in zip(current_model.parameters(), initial_model.parameters()):
                # 保存更新前的值用于比较
                before_update = current_param.data.clone()

                # 执行更新
                current_param.data.mul_(alpha).add_(initial_param.data, alpha=(1.0 - alpha))

                # 检查更新结果
                if alpha == 0:
                    # 当 α=0 时，当前参数应该等于初始参数
                    if not torch.equal(current_param.data, initial_param.data):
                        print("Error: Parameter was not correctly reset to initial value!")
                    
                    current_state = current_model.state_dict()
                    initial_state = initial_model.state_dict()

                    # 详细比较每一个键值对
                    for key in current_state:
                        if not torch.allclose(current_state[key], initial_state[key]):
                            print(f"状态不匹配: {key}")

        return current_model"""

def split_up_model(model, arch_name: str, dataset_name: str):
    """
    Split up the model into an encoder and a classifier.
    This is required for methods like RMT and AdaContrast
    Input:
        model: Model to be split up
        arch_name: Name of the network
        dataset_name: Name of the dataset
    Returns:
        encoder: The encoder of the model
        classifier The classifier of the model
    """
    if hasattr(model, "model") and hasattr(model.model, "pretrained_cfg") and hasattr(model.model,
                                                                                      model.model.pretrained_cfg[
                                                                                          "classifier"]):
        # split up models loaded from timm
        classifier = deepcopy(getattr(model.model, model.model.pretrained_cfg["classifier"]))
        encoder = model
        encoder.model.reset_classifier(0)
        if isinstance(model, ImageNetXWrapper):
            encoder = nn.Sequential(encoder.normalize, encoder.model)

    elif arch_name == "Standard" and dataset_name in {"cifar10", "cifar10_c"}:
        encoder = nn.Sequential(*list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif dataset_name == "domainnet126":
        encoder = model.encoder
        classifier = model.fc
    elif "resnet" in arch_name or "resnext" in arch_name or "wide_resnet" in arch_name or arch_name in {"Standard_R50",
                                                                                                        "Hendrycks2020AugMix",
                                                                                                        "Hendrycks2020Many",
                                                                                                        "Geirhos2018_SIN"}:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
    elif "densenet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten())
        classifier = model.model.classifier
    elif "efficientnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool, nn.Flatten())
        classifier = model.model.classifier
    elif "mnasnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.layers, nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                nn.Flatten())
        classifier = model.model.classifier
    elif "shufflenet" in arch_name:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1],
                                nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.fc
    elif "vit_" in arch_name and not "maxvit_" in arch_name:
        encoder = TransformerWrapper(model)
        classifier = model.model.heads.head
    elif "swin_" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.norm, model.model.permute,
                                model.model.avgpool, model.model.flatten)
        classifier = model.model.head
    elif "convnext" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool)
        classifier = model.model.classifier
    elif arch_name == "mobilenet_v2":
        encoder = nn.Sequential(model.normalize, model.model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    # for the EEGNet series
    elif "EEGNet" in arch_name: 
        encoder = model[0]
        classifier = model[1]
    else:
        raise ValueError(f"The model architecture '{arch_name}' is not supported for dataset '{dataset_name}'.")

    return encoder, classifier


# Function to check if the model or a specific module has been updated
def is_updated(model, initial_state):
    for name, param in model.named_parameters():
        if not torch.equal(param, initial_state[name]):
            return True
    for name, buffer in model.named_buffers():
        if not torch.equal(buffer, initial_state[name]):
            return True
    return False


class ImageNetXMaskingLayer(torch.nn.Module):
    """ Following: https://github.com/hendrycks/imagenet-r/blob/master/eval.py
    """

    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        return x[:, self.mask]


class ImageNetXWrapper(torch.nn.Module):
    def __init__(self, model, mask):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

        self.masking_layer = ImageNetXMaskingLayer(mask)

    def forward(self, x):
        logits = self.model(self.normalize(x))
        return self.masking_layer(logits)


class TransformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.normalize(x)
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x

def get_backbone_dropout(model, dropout_p):
    """
    Create a model copy with dropout for Monte Carlo Dropout evaluation.
    
    Args:
        model (nn.Module): Original backbone model
        dropout_p (float): Dropout probability value
    
    Returns:
        nn.Module: Modified model copy (with custom dropout behavior)
    """
    # Deep copy original model to avoid contaminating original weights
    model_copy = deepcopy(model)
    
    # Recursively replace all Dropout layers
    def replace_dropout(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Dropout):
                # Replace with always-active Dropout layer[6,7](@ref)
                setattr(module, name, AlwaysActiveDropout(p=dropout_p))
            else:
                replace_dropout(child)
    
    replace_dropout(model_copy)
    return model_copy

def loss_prepare(loss_name, EnergyAlignment):
    if loss_name == 'MemorySoftplusEnergyAlignment':
        return MemorySoftplusEnergyAlignment(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, temp=EnergyAlignment.temp)
    elif loss_name == 'CE_MDR':
        return CE_MDR(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, temp=EnergyAlignment.temp)
    elif loss_name == 'PresudoLabelMemorySoftplusEnergyAlignment':
        return PresudoLabelMemorySoftplusEnergyAlignment(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, 
                                                                lambda_3=1.0, temp=EnergyAlignment.temp)
    elif loss_name == 'MemorySoftplusEnergyWeightedAlignment':
        return MemorySoftplusEnergyWeightedAlignment(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, 
                                                                temp=EnergyAlignment.temp)
    elif loss_name == 'MemorySoftplusEnergyRatioSortedAlignment':
        return MemorySoftplusEnergyRatioSortedAlignment(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, 
                                                                temp=EnergyAlignment.temp)
    elif loss_name == 'MemorySoftplusEnergyFeatureWeightedAlignment':
        return MemorySoftplusEnergyFeatureWeightedAlignment(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, 
                                                                temp=EnergyAlignment.temp)
    elif loss_name == "MemorySoftplusEnergyWeightedAlignmentMDR":
        return MemorySoftplusEnergyWeightedAlignmentMDR(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, 
                                                                temp=EnergyAlignment.temp)
    elif loss_name == 'CaliE_MDR':
        return CaliE_MDR(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, temp=EnergyAlignment.temp)
    elif loss_name == 'CaliE_UKL':
        return CaliE_UKL(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, temp=EnergyAlignment.temp)
    elif loss_name == 'CE_KL':
        return CE_KL(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, temp=EnergyAlignment.temp)
    elif loss_name == 'CE_KL_review':
        return CE_KL_review(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'CE_KL_review_weighted':
        return CE_KL_review_weighted(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale)
    elif loss_name == 'CE_KL_review_weighted_1':
        return CE_KL_review_weighted_1(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale)
    elif loss_name == 'CE_KL_review_weighted_2':
        return CE_KL_review_weighted_2(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, confidence_threshold=EnergyAlignment.confidence_threshold, num_classes=EnergyAlignment.num_class)
    elif loss_name == 'CE_KL_review_weighted_3':
        return CE_KL_review_weighted_3(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, confidence_threshold=EnergyAlignment.confidence_threshold, num_classes=EnergyAlignment.num_class, entropy_threshold=EnergyAlignment.entropy_threshold)
    elif loss_name == 'CE_KL_review_weighted_3_1':
        return CE_KL_review_weighted_3_1(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, confidence_threshold=EnergyAlignment.confidence_threshold, num_classes=EnergyAlignment.num_class, entropy_threshold=EnergyAlignment.entropy_threshold)
    elif loss_name == 'CE_KL_review_weighted_3_2':
        return CE_KL_review_weighted_3_2(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, confidence_threshold=EnergyAlignment.confidence_threshold, num_classes=EnergyAlignment.num_class, entropy_threshold=EnergyAlignment.entropy_threshold)
    elif loss_name == 'CE_KL_review_weighted_4':
        return CE_KL_review_weighted_4(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, confidence_threshold=EnergyAlignment.confidence_threshold, num_classes=EnergyAlignment.num_class, entropy_threshold=EnergyAlignment.entropy_threshold)
    elif loss_name == 'CE_KL_review_weighted_5':
        return CE_KL_review_weighted_5(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, confidence_threshold=EnergyAlignment.confidence_threshold, num_classes=EnergyAlignment.num_class, entropy_threshold=EnergyAlignment.entropy_threshold)
    elif loss_name == 'CE_KL_review_weighted_6':
        return CE_KL_review_weighted_6(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, confidence_threshold=EnergyAlignment.confidence_threshold, num_classes=EnergyAlignment.num_class, entropy_threshold=EnergyAlignment.entropy_threshold)
    elif loss_name == 'CE_KL_review_3':
        return CE_KL_review_3(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, confidence_threshold=EnergyAlignment.confidence_threshold, num_classes=EnergyAlignment.num_class, entropy_threshold=EnergyAlignment.entropy_threshold)
    elif loss_name == 'CE_KL_review_weighted_7':
        return CE_KL_review_weighted_7(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, confidence_threshold=EnergyAlignment.confidence_threshold, num_classes=EnergyAlignment.num_class, entropy_threshold=EnergyAlignment.entropy_threshold, ratio=EnergyAlignment.ratio)
    elif loss_name == 'CE_KL_review_weighted_8':
        return CE_KL_review_weighted_8(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, confidence_threshold=EnergyAlignment.confidence_threshold, num_classes=EnergyAlignment.num_class, entropy_threshold=EnergyAlignment.entropy_threshold, ratio=EnergyAlignment.ratio, thre_alpha=EnergyAlignment.thre_alpha)
    elif loss_name == 'CE_KL_review_weighted_9':
        return CE_KL_review_weighted_9(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, confidence_threshold=EnergyAlignment.confidence_threshold, num_classes=EnergyAlignment.num_class, entropy_threshold=EnergyAlignment.entropy_threshold, ratio=EnergyAlignment.ratio, thre_alpha=EnergyAlignment.thre_alpha, gate_type=EnergyAlignment.gate_type)
    elif loss_name == 'CE_KL_review_weighted_10':
        return CE_KL_review_weighted_10(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, confidence_threshold=EnergyAlignment.confidence_threshold, num_classes=EnergyAlignment.num_class, entropy_threshold=EnergyAlignment.entropy_threshold, ratio=EnergyAlignment.ratio, thre_alpha=EnergyAlignment.thre_alpha)
    

    elif loss_name == 'CaliE_KL':
        return CaliE_KL(lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, temp=EnergyAlignment.temp)
    elif loss_name == 'EnergyEntropy_selected':
        return EnergyEntropy_selected(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'EnergyEntropy_selected_all':
        return EnergyEntropy_selected_all(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'EnergyEntropy_selected_align':
        return EnergyEntropy_selected_align(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'PresudoLabelEMA':
        return PresudoLabelEMA(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'PresudoLabelEMA_selection':
        return PresudoLabelEMA_selection(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'PresudoLabelEMA_energy':
        return PresudoLabelEMA_energy(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'PresudoLabelEMA_symmetric':
        return PresudoLabelEMA_symmetric(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'PresudoLabelEMA_lcs':
        return PresudoLabelEMA_lcs(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'PresudoLabelEMA_SampleCons':
        return PresudoLabelEMA_SampleCons(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    
    
    elif loss_name == 'EntropyMDREMA_lcs':
        return EntropyMDREMA_lcs(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'CaliE_MDR_lcs':
        return CaliE_MDR_lcs(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'CaliE_MDR_lcs_selection':
        return CaliE_MDR_lcs_selection(thr=0.4, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'CaliE_MDR_lcs_cons':
        return CaliE_MDR_lcs_cons(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'CaliE_MDR_lcs_ConsSamples':
        return CaliE_MDR_lcs_ConsSamples(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'CaliE_MDR_lcs_ConsSamplesFea':
        return CaliE_MDR_lcs_ConsSamplesFea(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'CE_KL_ConsSamplesFea':
        return CE_KL_ConsSamplesFea(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'CE_KL_lcs_ConsSamples':
        return CE_KL_lcs_ConsSamples(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'CE_KL_lcs_ConsSamples_selection':
        return CE_KL_lcs_ConsSamples_selection(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    


    elif loss_name == 'ConsSamples_lcs':
        return ConsSamples_lcs(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'ConsSamples_weighted':
        return ConsSamples_weighted(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'ConsSamples':
        return ConsSamples(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'ConsSamples_selection':
        return ConsSamples_selection(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name in ['ConsSamples_selection_two_stage', 'ConsSamples_selection_two_stage_adaptiveLR', 'ConsSamples_selection_two_stage_adaptiveLR_1','ConsSamples_selection_two_stage_adaptiveLR_2']:
        return ConsSamples_selection_two_stage(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted':
        return ConsSamples_selection_two_stage_weighted(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_1':
        return ConsSamples_selection_two_stage_weighted_1(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_2':
        return ConsSamples_selection_two_stage_weighted_2(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_3':
        return ConsSamples_selection_two_stage_weighted_3(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4':
        return ConsSamples_selection_two_stage_weighted_4(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1':
        return ConsSamples_selection_two_stage_weighted_4_1(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, weight_type='entropy_energy')
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_double':
        return ConsSamples_selection_two_stage_weighted_4_1_double(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_double_1':
        return ConsSamples_selection_two_stage_weighted_4_1_double_1(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_review':
        return ConsSamples_selection_two_stage_weighted_4_1_review(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_review_1':
        return ConsSamples_selection_two_stage_weighted_4_1_review_1(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_review_2':
        return ConsSamples_selection_two_stage_weighted_4_1_review_2(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_review_3':
        return ConsSamples_selection_two_stage_weighted_4_1_review_3(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_review_4':
        return ConsSamples_selection_two_stage_weighted_4_1_review_4(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_review_4_1':
        return ConsSamples_selection_two_stage_weighted_4_1_review_4_1(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, weight_type=EnergyAlignment.weight_type, ratio_reivew=EnergyAlignment.ratio_review)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_review_4_2':
        return ConsSamples_selection_two_stage_weighted_4_1_review_4_2(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, weight_type=EnergyAlignment.weight_type, ratio_reivew=EnergyAlignment.ratio_review)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_review_4_3':
        return ConsSamples_selection_two_stage_weighted_4_1_review_4_3(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, weight_type=EnergyAlignment.weight_type, ratio_reivew=EnergyAlignment.ratio_review, thre_alpha=EnergyAlignment.thre_alpha)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_review_4_4':
        return ConsSamples_selection_two_stage_weighted_4_1_review_4_4(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, weight_type=EnergyAlignment.weight_type, ratio_reivew=EnergyAlignment.ratio_review, thre_alpha=EnergyAlignment.thre_alpha)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_review_4_4_1':
        return ConsSamples_selection_two_stage_weighted_4_1_review_4_4_1(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, weight_type=EnergyAlignment.weight_type, ratio_reivew=EnergyAlignment.ratio_review, thre_alpha=EnergyAlignment.thre_alpha, loss_weight_type=EnergyAlignment.loss_weight_type, gate_type=EnergyAlignment.gate_type)
    elif loss_name == 'ConsSamples_selection_two_stage_weighted_4_1_review_4_4_2':
        return ConsSamples_selection_two_stage_weighted_4_1_review_4_4_2(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp, scale=EnergyAlignment.scale, weight_type=EnergyAlignment.weight_type, ratio_reivew=EnergyAlignment.ratio_review, thre_alpha=EnergyAlignment.thre_alpha, loss_weight_type=EnergyAlignment.loss_weight_type, gate_type=EnergyAlignment.gate_type)
    
    elif loss_name == 'ConsSamples_selection_1':
        return ConsSamples_selection_1(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'ConsSamples_selection_2':
        return ConsSamples_selection_2(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'ConsSamples_selection_dropout':
        return ConsSamples_selection_dropout(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'ConsSamples_selection_distillation':
        return ConsSamples_selection_distillation(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)
    elif loss_name == 'Weighted_ConsSamples_selection_distillation':
        return Weighted_ConsSamples_selection_distillation(ratio=EnergyAlignment.ratio, lambda_1=EnergyAlignment.lambda_1, lambda_2=EnergyAlignment.lambda_2, lambda_3=EnergyAlignment.lambda_3, temp=EnergyAlignment.temp)

class AlwaysActiveDropout(nn.Module):
    """Custom layer that always performs dropout (ignores train/eval mode)"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        # Always execute dropout in training mode[7,8](@ref)
        return F.dropout(x, p=self.p, training=True)