import random
import numpy as np
from easydict import EasyDict as edict
import torch
from collections import deque
import threading
import heapq
import math
import torch
import torch.nn.functional as F
import os

class MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0, logit=None, time_stamp=0, true_label=0):
        self.data = data.detach()
        self.uncertainty = uncertainty
        self.age = age
        self.logit = logit.detach()
        self.time_stamp = time_stamp
        # Initialize interval
        self.time_stamp_interval = 0
        self.true_label = true_label

    def increase_age(self):
        if not self.empty():
            self.age += 1

    def get_data(self):
        return self.data, self.uncertainty, self.age

    def empty(self):
        return self.data == "empty"


class DropMemoryBank_review_8:
    def __init__(self, capacity, num_class, confidence_threshold, uncertainty_threshold, type='UHUS', category_uniform=True, alpha=0.5):
        # Initialize memory bank with fixed capacity per class
        self.capacity = capacity
        self.num_class = num_class
        # Calculate fixed capacity per class (minimum 1)
        self.per_class_capacity = max(1, self.capacity // self.num_class)
        # Use list of lists for each class to enable custom management
        self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_class)]
        
        # Thresholds for memory management
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.type = type
        self.category_uniform = category_uniform

        # hyperparameters for time interval based management
        # score = alpha * norm_interval + (1 - alpha) * norm_time
        self.alpha = alpha

    def get_occupancy(self):
        """Return total number of stored instances across all classes"""
        return sum(len(q) for q in self.data)

    def per_class_dist(self):
        """Return current sample count per class"""
        return [len(q) for q in self.data]

    def get_majority_classes(self):
        """Return classes with maximum occupancy"""
        class_counts = self.per_class_dist()
        max_count = max(class_counts)
        return [i for i, count in enumerate(class_counts) if count == max_count]

    def get_non_empty_classes(self):
        """Return all classes with at least one instance"""
        return [i for i, q in enumerate(self.data) if len(q) > 0]

    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()
        return
    
    def update_time_stamp_intervals(self):
        """Update time_stamp_interval for all items in memory bank"""
        for class_list in self.data:
            if not class_list:
                continue
            for idx, item in enumerate(class_list):
                if idx > 0:
                    # the first item has no previous item to calculate interval
                    item.time_stamp_interval = item.time_stamp - class_list[idx - 1].time_stamp
        return
    
    def get_time_stamp_intervals(self):
        """Retrieve all time_stamp_intervals from memory bank"""
        intervals = []
        for class_list in self.data:
            intervals.append([item.time_stamp_interval for item in class_list])
        return intervals

    def add_instance(self, instance):
        # Extract instance components
        x, prediction, uncertainty, logit, time_stamp, true_label = (
            instance['data'], 
            instance['prediction'], 
            instance['uncertainty'],
            instance['logit'],
            instance['time_stamp'],
            instance['label'],
        )
        
        # Check for duplicate data across all classes
        for cls_idx, cls_queue in enumerate(self.data):
            if any(torch.equal(item.data, x) for item in cls_queue):
                # print(f"Item already stored in memory bank for class {cls_idx}")
                return True
        
        # Create new memory item
        new_item = MemoryItem(data=x, uncertainty=uncertainty, logit=logit, time_stamp=time_stamp, true_label=true_label)
        # add new item  
        if self.remove_instance(instance):
            self.append_with_interval(self.data[prediction], new_item)
        # update the intervals
        self.update_time_stamp_intervals()

    def append_with_interval(self, queue, item):
        # Helper function to handle interval calculation and appending
        if len(queue) == 0:
            # If queue is empty, interval is the timestamp itself
            item.time_stamp_interval = max(item.time_stamp, 1)
        queue.append(item)

    def remove_instance(self, instance):
        class_list = self.data[instance['prediction']]
        class_occupied = len(class_list)
        all_occupancy = self.get_occupancy()
        if class_occupied < self.per_class_capacity:
            if all_occupancy < self.capacity:
                return True
            else:
                majority_classes = self.get_majority_classes()
                return self.remove_from_classes_interval(majority_classes, instance)
        else:
            return self.remove_from_classes_interval([instance['prediction']], instance)
            
    def remove_from_classes_interval(self, classes: 'list[int]', instance):
        
        # --- 第一步：收集统计信息用于归一化 ---
        # 我们需要知道当前候选池中 interval 和 time_stamp 的 min/max
        all_intervals = []
        all_timestamps = []
        
        # 仅统计候选类中的样本
        for cls in classes:
            for item in self.data[cls]:
                all_intervals.append(item.time_stamp_interval)
                all_timestamps.append(item.time_stamp)
        
        if not all_intervals:
            # 如果候选类都是空的，直接允许添加
            return True

        min_int, max_int = min(all_intervals), max(all_intervals)
        min_time, max_time = min(all_timestamps), max(all_timestamps)
        
        # 防止除以零 (如果只有一个样本或所有值相同)
        range_int = max_int - min_int + 1e-8
        range_time = max_time - min_time + 1e-8

        # --- 第二步：寻找得分最低的样本 (Victim) ---
        min_score = float('inf')
        victim_info = None # (class_idx, item_idx)

        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                # 归一化 (映射到 0-1)
                norm_int = (item.time_stamp_interval - min_int + 1e-5) / range_int
                norm_time = (item.time_stamp - min_time + 1e-5) / range_time
                
                # 计算得分 (越小越容易被移除)
                # alpha 接近 1：主要看 interval (保留稀疏)
                # alpha 接近 0：主要看 time (保留最新)
                score = self.alpha * norm_int + (1 - self.alpha) * norm_time
                
                if score < min_score:
                    min_score = score
                    victim_info = (cls, idx)
        
        if victim_info is not None:
            victim_cls, victim_idx = victim_info
            # calculate the new interval
            target_queue = self.data[instance['prediction']]
            if len(target_queue) > 0:
                new_interval = instance['time_stamp'] - target_queue[-1].time_stamp
            else:
                new_interval = max(instance['time_stamp'], 1)
            
            # 使用与上面相同的基准对新实例进行打分
            # 注意：新实例的值可能超出之前的 max，导致归一化值 > 1，这是合理的（说明它更优秀）
            new_norm_int = (new_interval - min_int) / range_int
            new_norm_time = min((instance['time_stamp'] - min_time) / range_time, 1.0)

            new_score = self.alpha * new_norm_int + (1 - self.alpha) * new_norm_time

            # --- 第四步：比较与执行 ---
            # 只有当 新样本的价值(得分) >= 现有最差样本的价值 时，才进行替换
            if min_score <= new_score:
                
                if victim_idx  == 0 and len(self.data[victim_cls]) > 1:
                    # 特殊处理：如果移除的是头部 (min_index == 0)
                    # 且后面还有元素，则必须将旧头部的 interval 传给新头部
                    # 因为 update_time_stamp_intervals 不会计算 idx=0 的值
                    self.data[victim_cls][1].time_stamp_interval += self.data[victim_cls][0].time_stamp_interval


                # remove the target item
                self.data[victim_cls].pop(victim_idx)
                return True
            else:
                return False
        else:
            return True

    def get_memory(self):
        """Retrieve all stored data, logits, and class indices (all samples per class)"""
        tmp_data = []
        tmp_logits = []
        tmp_class = []
        for cls_idx, cls_queue in enumerate(self.data):
            for item in cls_queue:
                tmp_data.append(item.data)
                tmp_logits.append(item.logit)
                tmp_class.append(cls_idx)
        return tmp_data, tmp_logits, tmp_class
    
    def compute_logits_entropy(self):
        """
        计算所有 memory item 的 logit 的香农熵，并返回所有熵和均值。
        Returns:
            entropies: List[float]，每个样本的熵
            mean_entropy: float，所有样本熵的均值
        """
        entropies = []
        for class_list in self.data:
            for item in class_list:
                logits = item.logit
                # 计算 softmax 概率
                probs = F.softmax(logits, dim=-1)
                # 计算香农熵
                entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
                entropies.append(entropy)
        if entropies:
            mean_entropy = float(sum(entropies) / len(entropies))
            std_entropy = float((sum((x - mean_entropy) ** 2 for x in entropies) / len(entropies)) ** 0.5)
        else:
            mean_entropy = 0.0
            std_entropy = 0.0
        return entropies, mean_entropy, std_entropy
    
    def compute_logits_entropy_median_iqr(self):
        """
        计算所有 memory item 的 logit 的香农熵，并返回所有熵的中位数和四分位距（IQR）。
        Returns:
            entropies: List[float]，每个样本的熵
            median_entropy: float，所有样本熵的中位数
            iqr_entropy: float，所有样本熵的四分位距
        """
        entropies = []
        for class_list in self.data:
            for item in class_list:
                logits = item.logit
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
                entropies.append(entropy)
        if entropies:
            entropies_tensor = torch.tensor(entropies)
            median_entropy = torch.median(entropies_tensor).item()
            q1 = torch.quantile(entropies_tensor, 0.25).item()
            q3 = torch.quantile(entropies_tensor, 0.75).item()
            iqr_entropy = q3 - q1
        else:
            median_entropy = 0.0
            iqr_entropy = 0.0
        return entropies, median_entropy, iqr_entropy/2
    
    def save_buffer(self, path, save_data=False):
        """
        保存当前的内存缓冲区到指定路径。
        Args:
            path (str): 文件路径，用于保存缓冲区数据。
            save_data (bool): 是否存储每一个 MemoryItem 的实际数据data
        """
        buffer_to_save = []
        for class_list in self.data:
            class_items = []
            for item in class_list:
                item_dict = {
                    'uncertainty': item.uncertainty,
                    'logit': item.logit,
                    'time_stamp': item.time_stamp,
                    'time_stamp_interval': item.time_stamp_interval,
                    'label': item.true_label,
                }
                if save_data:
                    item_dict['data'] = item.data
                class_items.append(item_dict)
            buffer_to_save.append(class_items)
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        torch.save(buffer_to_save, path)


class HUS:
    def __init__(self, capacity, num_class, confidence_threshold, uncertainty_threshold, type='UHUS', category_uniform=True, alpha=0.5):
        # Initialize memory bank with fixed capacity per class
        self.capacity = capacity
        self.num_class = num_class
        # Calculate fixed capacity per class (minimum 1)
        self.per_class_capacity = max(1, self.capacity // self.num_class)
        # Use list of lists for each class to enable custom management
        self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_class)]
        
        # Thresholds for memory management
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.type = type
        self.category_uniform = category_uniform

        # hyperparameters for time interval based management
        # score = alpha * norm_interval + (1 - alpha) * norm_time
        self.alpha = alpha

    def get_occupancy(self):
        """Return total number of stored instances across all classes"""
        return sum(len(q) for q in self.data)

    def per_class_dist(self):
        """Return current sample count per class"""
        return [len(q) for q in self.data]

    def get_majority_classes(self):
        """Return classes with maximum occupancy"""
        class_counts = self.per_class_dist()
        max_count = max(class_counts)
        return [i for i, count in enumerate(class_counts) if count == max_count]

    def get_target_index(self, data):
        return random.randrange(0, len(data))

    def get_non_empty_classes(self):
        """Return all classes with at least one instance"""
        return [i for i, q in enumerate(self.data) if len(q) > 0]

    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()
        return

    def add_instance(self, instance):
        # Extract instance components
        x, prediction, uncertainty, logit, time_stamp, true_label = (
            instance['data'], 
            instance['prediction'], 
            instance['uncertainty'],
            instance['logit'],
            instance['time_stamp'],
            instance['label'],
        )
        
        # Check for duplicate data across all classes
        for cls_idx, cls_queue in enumerate(self.data):
            if any(torch.equal(item.data, x) for item in cls_queue):
                # print(f"Item already stored in memory bank for class {cls_idx}")
                return True
        
        # Create new memory item
        new_item = MemoryItem(data=x, uncertainty=uncertainty, logit=logit, time_stamp=time_stamp, true_label=true_label)
        # add new item  
        if self.get_occupancy() < self.capacity:
            self.data[prediction].append(new_item)
        else:
            if self.remove_instance(instance):
                self.data[prediction].append(new_item)

    def remove_instance(self, instance):
        if self.type == 'none':
            pass
        largest_indices = self.get_majority_classes()
        if instance['prediction'] not in largest_indices:
            class_index = random.choice(largest_indices)
            self.data[class_index].pop(random.randint(0, len(self.data[class_index]) - 1))
        else:
            class_index = instance['prediction']
            self.data[class_index].pop(random.randint(0, len(self.data[class_index]) - 1))

        return True

    
    def get_memory(self):
        """Retrieve all stored data, logits, and class indices (all samples per class)"""
        tmp_data = []
        tmp_logits = []
        tmp_class = []
        for cls_idx, cls_queue in enumerate(self.data):
            for item in cls_queue:
                tmp_data.append(item.data)
                tmp_logits.append(item.logit)
                tmp_class.append(cls_idx)
        return tmp_data, tmp_logits, tmp_class
    
    def compute_logits_entropy(self):
        """
        计算所有 memory item 的 logit 的香农熵，并返回所有熵和均值。
        Returns:
            entropies: List[float]，每个样本的熵
            mean_entropy: float，所有样本熵的均值
        """
        entropies = []
        for class_list in self.data:
            for item in class_list:
                logits = item.logit
                # 计算 softmax 概率
                probs = F.softmax(logits, dim=-1)
                # 计算香农熵
                entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
                entropies.append(entropy)
        if entropies:
            mean_entropy = float(sum(entropies) / len(entropies))
            std_entropy = float((sum((x - mean_entropy) ** 2 for x in entropies) / len(entropies)) ** 0.5)
        else:
            mean_entropy = 0.0
            std_entropy = 0.0
        return entropies, mean_entropy, std_entropy
    
    def compute_logits_entropy_median_iqr(self):
        """
        计算所有 memory item 的 logit 的香农熵，并返回所有熵的中位数和四分位距（IQR）。
        Returns:
            entropies: List[float]，每个样本的熵
            median_entropy: float，所有样本熵的中位数
            iqr_entropy: float，所有样本熵的四分位距
        """
        entropies = []
        for class_list in self.data:
            for item in class_list:
                logits = item.logit
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
                entropies.append(entropy)
        if entropies:
            entropies_tensor = torch.tensor(entropies)
            median_entropy = torch.median(entropies_tensor).item()
            q1 = torch.quantile(entropies_tensor, 0.25).item()
            q3 = torch.quantile(entropies_tensor, 0.75).item()
            iqr_entropy = q3 - q1
        else:
            median_entropy = 0.0
            iqr_entropy = 0.0
        return entropies, median_entropy, iqr_entropy/2
    
    def save_buffer(self, path, save_data=False):
        """
        保存当前的内存缓冲区到指定路径。
        Args:
            path (str): 文件路径，用于保存缓冲区数据。
            save_data (bool): 是否存储每一个 MemoryItem 的实际数据data
        """
        buffer_to_save = []
        for class_list in self.data:
            class_items = []
            for item in class_list:
                item_dict = {
                    'uncertainty': item.uncertainty,
                    'logit': item.logit,
                    'time_stamp': item.time_stamp,
                    'time_stamp_interval': item.time_stamp_interval,
                    'label': item.true_label,
                }
                if save_data:
                    item_dict['data'] = item.data
                class_items.append(item_dict)
            buffer_to_save.append(class_items)
        
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        torch.save(buffer_to_save, path)

class DropMemoryBank_review_9:
    def __init__(self, capacity, num_class, confidence_threshold, uncertainty_threshold, type='UHUS', category_uniform=True, alpha=0.5):
        # Initialize memory bank with fixed capacity per class
        self.capacity = capacity
        self.num_class = num_class
        # Calculate fixed capacity per class (minimum 1)
        self.per_class_capacity = max(1, self.capacity // self.num_class)
        # Use list of lists for each class to enable custom management
        self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_class)]
        
        # Thresholds for memory management
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.type = type
        self.category_uniform = category_uniform

        # hyperparameters for time interval based management
        # score = alpha * norm_interval + (1 - alpha) * norm_time
        self.alpha = alpha

    def get_occupancy(self):
        """Return total number of stored instances across all classes"""
        return sum(len(q) for q in self.data)

    def per_class_dist(self):
        """Return current sample count per class"""
        return [len(q) for q in self.data]

    def get_majority_classes(self):
        """Return classes with maximum occupancy"""
        class_counts = self.per_class_dist()
        max_count = max(class_counts)
        return [i for i, count in enumerate(class_counts) if count == max_count]

    def get_non_empty_classes(self):
        """Return all classes with at least one instance"""
        return [i for i, q in enumerate(self.data) if len(q) > 0]

    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()
        return
    
    def update_time_stamp_intervals(self):
        """Update time_stamp_interval for all items in memory bank"""
        for class_list in self.data:
            if not class_list:
                continue
            for idx, item in enumerate(class_list):
                if idx > 0:
                    # the first item has no previous item to calculate interval
                    item.time_stamp_interval = item.time_stamp - class_list[idx - 1].time_stamp
        return
    
    def get_time_stamp_intervals(self):
        """Retrieve all time_stamp_intervals from memory bank"""
        intervals = []
        for class_list in self.data:
            intervals.append([item.time_stamp_interval for item in class_list])
        return intervals

    def add_instance(self, instance):
        # Extract instance components
        x, prediction, uncertainty, logit, time_stamp, true_label = (
            instance['data'], 
            instance['prediction'], 
            instance['uncertainty'],
            instance['logit'],
            instance['time_stamp'],
            instance['label'],
        )
        
        # Check for duplicate data across all classes
        for cls_idx, cls_queue in enumerate(self.data):
            if any(torch.equal(item.data, x) for item in cls_queue):
                # print(f"Item already stored in memory bank for class {cls_idx}")
                return True
        
        # Create new memory item
        new_item = MemoryItem(data=x, uncertainty=uncertainty, logit=logit, time_stamp=time_stamp, true_label=true_label)
        # add new item  
        if self.remove_instance(instance):
            self.append_with_interval(self.data[prediction], new_item)
        # update the intervals
        self.update_time_stamp_intervals()

    def append_with_interval(self, queue, item):
        # Helper function to handle interval calculation and appending
        if len(queue) == 0:
            # If queue is empty, interval is the timestamp itself
            item.time_stamp_interval = max(item.time_stamp, 1)
        queue.append(item)

    def remove_instance(self, instance):
        class_list = self.data[instance['prediction']]
        class_occupied = len(class_list)
        all_occupancy = self.get_occupancy()
        if class_occupied < self.per_class_capacity:
            if all_occupancy < self.capacity:
                return True
            else:
                majority_classes = self.get_majority_classes()
                return self.remove_from_classes_interval(majority_classes, instance)
        else:
            return self.remove_from_classes_interval([instance['prediction']], instance)
            
    def remove_from_classes_interval(self, classes: 'list[int]', instance):
        
        # --- 第一步：收集统计信息用于归一化 ---
        # 我们需要知道当前候选池中 interval 和 time_stamp 的 min/max
        all_intervals = []
        all_timestamps = []
        
        # 仅统计候选类中的样本
        for cls in classes:
            for item in self.data[cls]:
                all_intervals.append(item.time_stamp_interval)
                all_timestamps.append(item.time_stamp)
        
        if not all_intervals:
            # 如果候选类都是空的，直接允许添加
            return True

        min_int, max_int = min(all_intervals), max(all_intervals)
        min_time, max_time = min(all_timestamps), max(all_timestamps)
        
        # 防止除以零 (如果只有一个样本或所有值相同)
        range_int = max_int - min_int + 1e-8
        range_time = max_time - min_time + 1e-8

        # --- 第二步：寻找得分最低的样本 (Victim) ---
        min_score = float('inf')
        victim_info = None # (class_idx, item_idx)

        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                # 归一化 (映射到 0-1)
                norm_int = (item.time_stamp_interval - min_int + 1e-5) / range_int
                norm_time = (item.time_stamp - min_time + 1e-5) / range_time
                
                # 计算得分 (越小越容易被移除)
                # alpha 接近 1：主要看 interval (保留稀疏)
                # alpha 接近 0：主要看 time (保留最新)
                score = self.alpha * norm_int + (1 - self.alpha) * norm_time
                
                if score < min_score:
                    min_score = score
                    victim_info = (cls, idx)
        
        if victim_info is not None:
            victim_cls, victim_idx = victim_info
            # calculate the new interval
            target_queue = self.data[instance['prediction']]
            if len(target_queue) > 0:
                new_interval = instance['time_stamp'] - target_queue[-1].time_stamp
            else:
                new_interval = max(instance['time_stamp'], 1)
            
            # 使用与上面相同的基准对新实例进行打分
            # 注意：新实例的值可能超出之前的 max，导致归一化值 > 1，这是合理的（说明它更优秀）
            new_norm_int = (new_interval - min_int) / range_int
            new_norm_time = min((instance['time_stamp'] - min_time) / range_time, 1.0)

            new_score = self.alpha * new_norm_int + (1 - self.alpha) * new_norm_time

            # --- 第四步：比较与执行 ---
            # 只有当 新样本的价值(得分) >= 现有最差样本的价值 时，才进行替换
            if min_score <= new_score:
                
                """
                if victim_idx  == 0 and len(self.data[victim_cls]) > 1:
                    # 特殊处理：如果移除的是头部 (min_index == 0)
                    # 且后面还有元素，则必须将旧头部的 interval 传给新头部
                    # 因为 update_time_stamp_intervals 不会计算 idx=0 的值
                    self.data[victim_cls][1].time_stamp_interval += self.data[victim_cls][0].time_stamp_interval
                """

                # remove the target item
                self.data[victim_cls].pop(victim_idx)
                return True
            else:
                return False
        else:
            return True

    def get_memory(self):
        """Retrieve all stored data, logits, and class indices (all samples per class)"""
        tmp_data = []
        tmp_logits = []
        tmp_class = []
        for cls_idx, cls_queue in enumerate(self.data):
            for item in cls_queue:
                tmp_data.append(item.data)
                tmp_logits.append(item.logit)
                tmp_class.append(cls_idx)
        return tmp_data, tmp_logits, tmp_class
    
    def compute_logits_entropy(self):
        """
        计算所有 memory item 的 logit 的香农熵，并返回所有熵和均值。
        Returns:
            entropies: List[float]，每个样本的熵
            mean_entropy: float，所有样本熵的均值
        """
        entropies = []
        for class_list in self.data:
            for item in class_list:
                logits = item.logit
                # 计算 softmax 概率
                probs = F.softmax(logits, dim=-1)
                # 计算香农熵
                entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
                entropies.append(entropy)
        if entropies:
            mean_entropy = float(sum(entropies) / len(entropies))
            std_entropy = float((sum((x - mean_entropy) ** 2 for x in entropies) / len(entropies)) ** 0.5)
        else:
            mean_entropy = 0.0
            std_entropy = 0.0
        return entropies, mean_entropy, std_entropy
    
    def compute_logits_entropy_median_iqr(self):
        """
        计算所有 memory item 的 logit 的香农熵，并返回所有熵的中位数和四分位距（IQR）。
        Returns:
            entropies: List[float]，每个样本的熵
            median_entropy: float，所有样本熵的中位数
            iqr_entropy: float，所有样本熵的四分位距
        """
        entropies = []
        for class_list in self.data:
            for item in class_list:
                logits = item.logit
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
                entropies.append(entropy)
        if entropies:
            entropies_tensor = torch.tensor(entropies)
            median_entropy = torch.median(entropies_tensor).item()
            q1 = torch.quantile(entropies_tensor, 0.25).item()
            q3 = torch.quantile(entropies_tensor, 0.75).item()
            iqr_entropy = q3 - q1
        else:
            median_entropy = 0.0
            iqr_entropy = 0.0
        return entropies, median_entropy, iqr_entropy/2
    
    def save_buffer(self, path, save_data=False):
        """
        保存当前的内存缓冲区到指定路径。
        Args:
            path (str): 文件路径，用于保存缓冲区数据。
            save_data (bool): 是否存储每一个 MemoryItem 的实际数据data
        """
        buffer_to_save = []
        for class_list in self.data:
            class_items = []
            for item in class_list:
                item_dict = {
                    'uncertainty': item.uncertainty,
                    'logit': item.logit,
                    'time_stamp': item.time_stamp,
                    'time_stamp_interval': item.time_stamp_interval,
                    'label': item.true_label,
                }
                if save_data:
                    item_dict['data'] = item.data
                class_items.append(item_dict)
            buffer_to_save.append(class_items)
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        torch.save(buffer_to_save, path)


class CSTU:
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.num_class = num_class
        # Calculate fixed capacity per class (minimum 1)
        self.per_class_capacity = max(1, self.capacity // self.num_class)
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u

        self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_class)]

    def get_occupancy(self):
        """Return total number of stored instances across all classes"""
        return sum(len(q) for q in self.data)

    def per_class_dist(self):
        """Return current sample count per class"""
        return [len(q) for q in self.data]

    def add_instance(self, instance):

        # Extract instance components
        x, prediction, uncertainty, logit, time_stamp, true_label = (
            instance['data'], 
            instance['prediction'], 
            instance['uncertainty'],
            instance['logit'],
            instance['time_stamp'],
            instance['label'],
        )
        
        # Check for duplicate data across all classes
        for cls_idx, cls_queue in enumerate(self.data):
            if any(torch.equal(item.data, x) for item in cls_queue):
                # print(f"Item already stored in memory bank for class {cls_idx}")
                return True
        
        # Create new memory item
        new_item = MemoryItem(data=x, uncertainty=uncertainty, logit=logit, time_stamp=time_stamp, age=0, true_label=true_label)
        
        new_score = self.heuristic_score(0, logit)
        if self.remove_instance(prediction, new_score):
            self.data[prediction].append(new_item)
        self.add_age()

    def remove_instance(self, cls, score):
        class_list = self.data[cls]
        class_occupied = len(class_list)
        all_occupancy = self.get_occupancy()
        if class_occupied < self.per_class_capacity:
            if all_occupancy < self.capacity:
                return True
            else:
                majority_classes = self.get_majority_classes()
                return self.remove_from_classes(majority_classes, score)
        else:
            return self.remove_from_classes([cls], score)

    def remove_from_classes(self, classes: 'list[int]', score_base):
        max_class = None
        max_index = None
        max_score = None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                logit = item.logit
                age = item.age
                score = self.heuristic_score(age=age, logit=logit)
                if max_score is None or score >= max_score:
                    max_score = score
                    max_index = idx
                    max_class = cls

        if max_class is not None:
            if max_score > score_base:
                self.data[max_class].pop(max_index)
                return True
            else:
                return False
        else:
            return True

    def get_majority_classes(self):
        per_class_dist = self.per_class_dist()
        max_occupied = max(per_class_dist)
        classes = []
        for i, occupied in enumerate(per_class_dist):
            if occupied == max_occupied:
                classes.append(i)

        return classes

    def heuristic_score(self, age, logit):
        # 计算 softmax 概率
        probs = F.softmax(logit, dim=-1)
        # 计算香农熵
        entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
        return self.lambda_t * 1 / (1 + math.exp(-age / self.capacity)) + self.lambda_u * entropy / math.log(self.num_class)

    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()
        return

    def get_memory(self):
        """Retrieve all stored data, logits, and class indices (all samples per class)"""
        tmp_data = []
        tmp_logits = []
        tmp_class = []
        for cls_idx, cls_queue in enumerate(self.data):
            for item in cls_queue:
                tmp_data.append(item.data)
                tmp_logits.append(item.logit)
                tmp_class.append(cls_idx)
        return tmp_data, tmp_logits, tmp_class
    
    
    def compute_logits_entropy(self):
        """
        计算所有 memory item 的 logit 的香农熵，并返回所有熵和均值。
        Returns:
            entropies: List[float]，每个样本的熵
            mean_entropy: float，所有样本熵的均值
        """
        entropies = []
        for class_list in self.data:
            for item in class_list:
                logits = item.logit
                # 计算 softmax 概率
                probs = F.softmax(logits, dim=-1)
                # 计算香农熵
                entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
                entropies.append(entropy)
        if entropies:
            mean_entropy = float(sum(entropies) / len(entropies))
            std_entropy = float((sum((x - mean_entropy) ** 2 for x in entropies) / len(entropies)) ** 0.5)
        else:
            mean_entropy = 0.0
            std_entropy = 0.0
        return entropies, mean_entropy, std_entropy
    
    def compute_logits_entropy_median_iqr(self):
        """
        计算所有 memory item 的 logit 的香农熵，并返回所有熵的中位数和四分位距（IQR）。
        Returns:
            entropies: List[float]，每个样本的熵
            median_entropy: float，所有样本熵的中位数
            iqr_entropy: float，所有样本熵的四分位距
        """
        entropies = []
        for class_list in self.data:
            for item in class_list:
                logits = item.logit
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
                entropies.append(entropy)
        if entropies:
            entropies_tensor = torch.tensor(entropies)
            median_entropy = torch.median(entropies_tensor).item()
            q1 = torch.quantile(entropies_tensor, 0.25).item()
            q3 = torch.quantile(entropies_tensor, 0.75).item()
            iqr_entropy = q3 - q1
        else:
            median_entropy = 0.0
            iqr_entropy = 0.0
        return entropies, median_entropy, iqr_entropy/2
    
    def save_buffer(self, path, save_data=False):
        """
        保存当前的内存缓冲区到指定路径。
        Args:
            path (str): 文件路径，用于保存缓冲区数据。
            save_data (bool): 是否存储每一个 MemoryItem 的实际数据data
        """
        buffer_to_save = []
        for class_list in self.data:
            class_items = []
            for item in class_list:
                item_dict = {
                    'uncertainty': item.uncertainty,
                    'logit': item.logit,
                    'time_stamp': item.time_stamp,
                    'time_stamp_interval': item.time_stamp_interval,
                    'label': item.true_label,
                }
                if save_data:
                    item_dict['data'] = item.data
                class_items.append(item_dict)
            buffer_to_save.append(class_items)
        
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        torch.save(buffer_to_save, path)

class FIFO:
    def __init__(self, capacity, num_class, confidence_threshold, uncertainty_threshold, type='UHUS', category_uniform=True, alpha=0.5):
        # Initialize memory bank with fixed capacity per class
        self.capacity = capacity
        self.num_class = num_class
        # Calculate fixed capacity per class (minimum 1)
        self.per_class_capacity = max(1, self.capacity // self.num_class)
        # Use deque for each class to enable automatic FIFO behavior
        self.data = [deque(maxlen=self.per_class_capacity) 
                     for _ in range(self.num_class)]
        
        # Thresholds for memory management
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.type = type
        self.category_uniform = category_uniform


    def get_occupancy(self):
        """Return total number of stored instances across all classes"""
        return sum(len(q) for q in self.data)

    def per_class_dist(self):
        """Return current sample count per class"""
        return [len(q) for q in self.data]

    def get_majority_classes(self):
        """Return classes with maximum occupancy"""
        class_counts = self.per_class_dist()
        max_count = max(class_counts)
        return [i for i, count in enumerate(class_counts) if count == max_count]

    def get_target_index(self, data):
        return random.randrange(0, len(data))

    def get_non_empty_classes(self):
        """Return all classes with at least one instance"""
        return [i for i, q in enumerate(self.data) if len(q) > 0]

    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()
        return

    def add_instance(self, instance):
        # Extract instance components
        x, prediction, uncertainty, logit, time_stamp, true_label = (
            instance['data'], 
            instance['prediction'], 
            instance['uncertainty'],
            instance['logit'],
            instance['time_stamp'],
            instance['label'],
        )
        
        # Check for duplicate data across all classes
        for cls_idx, cls_queue in enumerate(self.data):
            if any(torch.equal(item.data, x) for item in cls_queue):
                # print(f"Item already stored in memory bank for class {cls_idx}")
                return True
        
        # Create new memory item
        new_item = MemoryItem(data=x, uncertainty=uncertainty, logit=logit, time_stamp=time_stamp, true_label=true_label)
        # add new item  
        self.data[prediction].append(new_item)
        

    def get_memory(self):
        """Retrieve all stored data, logits, and class indices (all samples per class)"""
        tmp_data = []
        tmp_logits = []
        tmp_class = []
        for cls_idx, cls_queue in enumerate(self.data):
            for item in cls_queue:
                tmp_data.append(item.data)
                tmp_logits.append(item.logit)
                tmp_class.append(cls_idx)
        return tmp_data, tmp_logits, tmp_class
    
    def compute_logits_entropy(self):
        """
        计算所有 memory item 的 logit 的香农熵，并返回所有熵和均值。
        Returns:
            entropies: List[float]，每个样本的熵
            mean_entropy: float，所有样本熵的均值
        """
        entropies = []
        for class_list in self.data:
            for item in class_list:
                logits = item.logit
                # 计算 softmax 概率
                probs = F.softmax(logits, dim=-1)
                # 计算香农熵
                entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
                entropies.append(entropy)
        if entropies:
            mean_entropy = float(sum(entropies) / len(entropies))
            std_entropy = float((sum((x - mean_entropy) ** 2 for x in entropies) / len(entropies)) ** 0.5)
        else:
            mean_entropy = 0.0
            std_entropy = 0.0
        return entropies, mean_entropy, std_entropy
    
    def compute_logits_entropy_median_iqr(self):
        """
        计算所有 memory item 的 logit 的香农熵，并返回所有熵的中位数和四分位距（IQR）。
        Returns:
            entropies: List[float]，每个样本的熵
            median_entropy: float，所有样本熵的中位数
            iqr_entropy: float，所有样本熵的四分位距
        """
        entropies = []
        for class_list in self.data:
            for item in class_list:
                logits = item.logit
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
                entropies.append(entropy)
        if entropies:
            entropies_tensor = torch.tensor(entropies)
            median_entropy = torch.median(entropies_tensor).item()
            q1 = torch.quantile(entropies_tensor, 0.25).item()
            q3 = torch.quantile(entropies_tensor, 0.75).item()
            iqr_entropy = q3 - q1
        else:
            median_entropy = 0.0
            iqr_entropy = 0.0
        return entropies, median_entropy, iqr_entropy/2
    
    def save_buffer(self, path, save_data=False):
        """
        保存当前的内存缓冲区到指定路径。
        Args:
            path (str): 文件路径，用于保存缓冲区数据。
            save_data (bool): 是否存储每一个 MemoryItem 的实际数据data
        """
        buffer_to_save = []
        for class_list in self.data:
            class_items = []
            for item in class_list:
                item_dict = {
                    'uncertainty': item.uncertainty,
                    'logit': item.logit,
                    'time_stamp': item.time_stamp,
                    'time_stamp_interval': item.time_stamp_interval,
                    'label': item.true_label,
                }
                if save_data:
                    item_dict['data'] = item.data
                class_items.append(item_dict)
            buffer_to_save.append(class_items)
        
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        torch.save(buffer_to_save, path)


class OnlineBufferInstance: 
    def __init__(self, buffer_size: int):
        """
        Initializes a fixed-size FIFO buffer for streaming data.
        
        Args:
            buffer_size: Maximum capacity of the buffer (number of data samples)
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)  # FIFO queue that automatically discards old data
        self.buffer_instance = deque(maxlen=buffer_size)  # FIFO queue that automatically discards old data 
        self.lock = threading.Lock()  # Thread safety lock

    def add_data(self, data: torch.Tensor) -> None:
        """
        Adds new data to the buffer (automatically discards oldest data if full)
        
        Args:
            data: Tensor data to add (supports arbitrary dimensions)
        """
        with self.lock:  # Ensures thread-safe operation
            # Store detached clone to prevent interference with original data
            self.buffer.append(data.detach().clone())

    def add_instance(self, instance):
        """
        Adds new instance to the buffer (automatically discards oldest data if full)
        
        Args:
            instance: Tensor instance to add (supports arbitrary dimensions)
        """
        with self.lock:
            self.buffer_instance.append(instance)

    def get_data(self) -> torch.Tensor:
        """
        Retrieves all current data from the buffer as a stacked tensor.
        
        Returns:
            stacked_data: Tensor with shape [N, ...] where N is current data count
        """
        with self.lock:
            if not self.buffer:
                return torch.tensor([])  # Return empty tensor if buffer is empty
            # Concatenate all tensors along new dimension (dim=0)
            return tuple(self.buffer)
    
    def get_instance(self):
        """
        Retrieves all current instance from the buffer as a stacked tensor.
        
        Returns:
            stacked_insatnce: edict with shape [N, ...] where N is current data count
        """
        with self.lock:
            if not self.buffer_instance:
                return torch.tensor([])  # Return empty tensor if buffer is empty
            # Concatenate all tensors along new dimension (dim=0)
            return tuple(self.buffer_instance)
    
    def get_weights(self):
        """get weights (uncertainty) from the instances"""
        with self.lock:
            weights = []
            if not self.buffer_instance:
                return torch.tensor([])  # Return empty tensor if buffer is empty
            else:
                for _instance in self.buffer_instance:
                    _weight = _instance['uncertainty']
                    weights.append(_weight)
            
            return torch.tensor(weights)

    def get_confidence(self):
        """get confidence from the instances"""
        with self.lock:
            confidences = []
            if not self.buffer_instance:
                return torch.tensor([])  # Return empty tensor if buffer is empty
            else:
                for _instance in self.buffer_instance:
                    _confidence = _instance['confidence']  # Extract confidence from each instance
                    confidences.append(_confidence)
            
            return torch.tensor(confidences)

    def get_logits(self):
        """get logits from the instances"""
        with self.lock:
            logits = []
            if not self.buffer_instance:
                return torch.tensor([])  # Return empty tensor if buffer is empty
            else:
                for _instance in self.buffer_instance:
                    _logit = _instance['logit']  # Extract confidence from each instance
                    logits.append(_logit)
            
            return torch.stack(logits, dim=0)

    def size(self) -> int:
        """Returns current number of data samples in the buffer"""
        return len(self.buffer), len(self.buffer_instance)
    
    
    def save_buffer(self, path, save_data=True):
        """
        Save all buffer_instance contents to a file.
        Args:
            path (str): File path to save buffer.
            save_data (bool): Whether to save the 'data' field.
        """
        with self.lock:
            buffer_to_save = []
            for instance in self.buffer_instance:
                instance_dict = {}
                for k, v in instance.items():
                    if not save_data and k == 'data':
                        continue
                    instance_dict[k] = v
                buffer_to_save.append(instance_dict)

            dir_path = os.path.dirname(path)
            os.makedirs(dir_path, exist_ok=True)
            torch.save(buffer_to_save, path)