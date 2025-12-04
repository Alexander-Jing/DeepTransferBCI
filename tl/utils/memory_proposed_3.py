import random
import numpy as np
from easydict import EasyDict as edict
import torch
from collections import deque
import threading

# the memory buffer for the presudo source
class MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0, logit=torch.tensor([0.25,0.25,0.25,0.25])):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age
        self.logit = logit

    def increase_age(self):
        if not self.empty():
            self.age += 1

    def get_data(self):
        return self.data, self.uncertainty, self.age

    def empty(self):
        return self.data == "empty"


class DropMemoryBank_review_2:
    def __init__(self, capacity, num_class, confidence_threshold, uncertainty_threshold, type='UHUS', category_uniform=True):
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

    def get_non_empty_classes(self):
        """Return all classes with at least one instance"""
        return [i for i, q in enumerate(self.data) if len(q) > 0]

    def add_instance(self, instance):
        # Extract instance components
        x, prediction, uncertainty, logit = (
            instance['data'], 
            instance['prediction'], 
            instance['uncertainty'],
            instance['logit'],
        )
        
        # Check for duplicate data across all classes
        for cls_idx, cls_queue in enumerate(self.data):
            if any(torch.equal(item.data, x) for item in cls_queue):
                # print(f"Item already stored in memory bank for class {cls_idx}")
                return True
        
        # Create new memory item
        new_item = MemoryItem(data=x, uncertainty=uncertainty, logit=logit)
        
        # Add to corresponding class (automatically handles FIFO)
        if self.get_occupancy() < self.capacity:
            self.data[prediction].append(new_item)
        else:
            if self.remove_instance():
                self.data[prediction].append(new_item)

    def remove_instance(self):
        class_index = self.get_majority_classes()
        cls_queue = self.data[class_index[0]]
        if len(cls_queue) == 0:
            return False
        # Randomly select an index to remove
        idx_to_remove = random.randint(0, len(cls_queue) - 1)
        # Remove the item at idx_to_remove
        # deque does not support direct index deletion, so rebuild the queue
        cls_queue.rotate(-idx_to_remove)
        cls_queue.popleft()
        cls_queue.rotate(idx_to_remove)
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


    def get_memory_review(self, batch_size):
        """Retrieve data with batch_size samples per class (FIFO order)"""
        tmp_data = []
        tmp_logits = []
        tmp_class = []
        
        for cls_idx, cls_queue in enumerate(self.data):
            # Take the first batch_size samples from each class queue
            selected_items = list(cls_queue)[:batch_size]
            
            for item in selected_items:
                tmp_data.append(item.data)
                tmp_logits.append(item.logit)
                tmp_class.append(cls_idx)
            
        return tmp_data, tmp_logits, tmp_class
    

    def get_memory_review_1(self, batch_size):
        """
        对每个类别，使用Reservoir Sampling随机抽取batch_size个样本。
        返回：tmp_data, tmp_logits, tmp_class
        """
        tmp_data = []
        tmp_logits = []
        tmp_class = []

        for cls_idx, cls_queue in enumerate(self.data):
            items = list(cls_queue)
            n = len(items)
            if n <= batch_size:
                selected_items = items
            else:
                # Reservoir Sampling
                reservoir = items[:batch_size]
                for i in range(batch_size, n):
                    j = random.randint(0, i)
                    if j < batch_size:
                        reservoir[j] = items[i]
                selected_items = reservoir

            for item in selected_items:
                tmp_data.append(item.data)
                tmp_logits.append(item.logit)
                tmp_class.append(cls_idx)

        return tmp_data, tmp_logits, tmp_class
    
    def get_memory_review_2(self, batch_size, sup_data):
        """
        对每个类别，使用Reservoir Sampling随机抽取batch_size个样本，且不与sup_data重合。
        返回：tmp_data, tmp_logits, tmp_class
        """
        tmp_data = []
        tmp_logits = []
        tmp_class = []

        # 将sup_data转为hashable集合（tuple），便于查重
        if isinstance(sup_data, torch.Tensor):
            sup_data_set = set(tuple(d.cpu().numpy().flatten()) for d in sup_data)
        elif isinstance(sup_data, (list, tuple)):
            sup_data_set = set(tuple(d.cpu().numpy().flatten()) for d in sup_data)
        else:
            sup_data_set = set()

        for cls_idx, cls_queue in enumerate(self.data):
            items = list(cls_queue)
            # 过滤掉与sup_data重合的item
            filtered_items = []
            for item in items:
                item_tuple = tuple(item.data.cpu().numpy().flatten())
                if item_tuple not in sup_data_set:
                    filtered_items.append(item)
            n = len(filtered_items)
            if n <= batch_size:
                selected_items = filtered_items
            else:
                # Reservoir Sampling
                reservoir = filtered_items[:batch_size]
                for i in range(batch_size, n):
                    j = random.randint(0, i)
                    if j < batch_size:
                        reservoir[j] = filtered_items[i]
                selected_items = reservoir

            for item in selected_items:
                tmp_data.append(item.data)
                tmp_logits.append(item.logit)
                tmp_class.append(cls_idx)

        return tmp_data, tmp_logits, tmp_class

    def get_prototypes(self, ratio=0.2):
        """
        Compute prototype vectors per class
        :param ratio: Fraction of lowest-uncertainty samples to use
        :return: Prototype tensor of shape [num_class, feature_dim]
        """
        # Determine feature dimension from first non-empty item
        feature_dim = next((item.data.shape[-1] 
                           for q in self.data for item in q if item.data is not None), None)
        if feature_dim is None:
            return torch.empty(0, 0)
        
        prototypes = torch.zeros(self.num_class, feature_dim)
        
        for cls_idx in range(self.num_class):
            cls_items = list(self.data[cls_idx])  # All items in current class
            if not cls_items:
                continue
                
            # Select lowest uncertainty samples
            sorted_items = sorted(cls_items, key=lambda x: x.uncertainty)
            n_select = max(1, int(ratio * len(cls_items)))
            selected_data = [item.data for item in sorted_items[:n_select]]
            
            # Compute mean prototype vector
            prototypes[cls_idx] = torch.stack(selected_data).mean(dim=0)
            
        return prototypes


class DropMemoryBank_review_3:
    def __init__(self, capacity, num_class, confidence_threshold, uncertainty_threshold, type='UHUS', category_uniform=True):
        # Initialize memory bank with fixed capacity per class
        self.capacity = capacity
        self.num_class = num_class
        # Calculate fixed capacity per class (minimum 1)
        self.per_class_capacity = max(1, self.capacity)
        
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

    def get_non_empty_classes(self):
        """Return all classes with at least one instance"""
        return [i for i, q in enumerate(self.data) if len(q) > 0]

    def add_instance(self, instance):
        # Extract instance components
        x, prediction, uncertainty, logit = (
            instance['data'], 
            instance['prediction'], 
            instance['uncertainty'],
            instance['logit'],
        )
        
        # Check for duplicate data across all classes
        for cls_idx, cls_queue in enumerate(self.data):
            if any(torch.equal(item.data, x) for item in cls_queue):
                # print(f"Item already stored in memory bank for class {cls_idx}")
                return True
        
        # Create new memory item
        new_item = MemoryItem(data=x, uncertainty=uncertainty, logit=logit)
        
        # Add to corresponding class (automatically handles FIFO)
        if self.get_occupancy() < self.capacity:
            self.data[prediction].append(new_item)
        else:
            if self.remove_instance():
                self.data[prediction].append(new_item)

    def remove_instance(self):
        class_index = self.get_majority_classes()
        cls_queue = self.data[class_index[0]]
        if len(cls_queue) == 0:
            return False
        # Randomly select an index to remove
        idx_to_remove = random.randint(0, len(cls_queue) - 1)
        # Remove the item at idx_to_remove
        # deque does not support direct index deletion, so rebuild the queue
        cls_queue.rotate(-idx_to_remove)
        cls_queue.popleft()
        cls_queue.rotate(idx_to_remove)
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


    def get_memory_review(self, batch_size):
        """Retrieve data with batch_size samples per class (FIFO order)"""
        tmp_data = []
        tmp_logits = []
        tmp_class = []
        
        for cls_idx, cls_queue in enumerate(self.data):
            # Take the first batch_size samples from each class queue
            selected_items = list(cls_queue)[:batch_size]
            
            for item in selected_items:
                tmp_data.append(item.data)
                tmp_logits.append(item.logit)
                tmp_class.append(cls_idx)
            
        return tmp_data, tmp_logits, tmp_class
    

    def get_memory_review_1(self, batch_size):
        """
        对每个类别，使用Reservoir Sampling随机抽取batch_size个样本。
        返回：tmp_data, tmp_logits, tmp_class
        """
        tmp_data = []
        tmp_logits = []
        tmp_class = []

        for cls_idx, cls_queue in enumerate(self.data):
            items = list(cls_queue)
            n = len(items)
            if n <= batch_size:
                selected_items = items
            else:
                # Reservoir Sampling
                reservoir = items[:batch_size]
                for i in range(batch_size, n):
                    j = random.randint(0, i)
                    if j < batch_size:
                        reservoir[j] = items[i]
                selected_items = reservoir

            for item in selected_items:
                tmp_data.append(item.data)
                tmp_logits.append(item.logit)
                tmp_class.append(cls_idx)

        return tmp_data, tmp_logits, tmp_class

    def get_prototypes(self, ratio=0.2):
        """
        Compute prototype vectors per class
        :param ratio: Fraction of lowest-uncertainty samples to use
        :return: Prototype tensor of shape [num_class, feature_dim]
        """
        # Determine feature dimension from first non-empty item
        feature_dim = next((item.data.shape[-1] 
                           for q in self.data for item in q if item.data is not None), None)
        if feature_dim is None:
            return torch.empty(0, 0)
        
        prototypes = torch.zeros(self.num_class, feature_dim)
        
        for cls_idx in range(self.num_class):
            cls_items = list(self.data[cls_idx])  # All items in current class
            if not cls_items:
                continue
                
            # Select lowest uncertainty samples
            sorted_items = sorted(cls_items, key=lambda x: x.uncertainty)
            n_select = max(1, int(ratio * len(cls_items)))
            selected_data = [item.data for item in sorted_items[:n_select]]
            
            # Compute mean prototype vector
            prototypes[cls_idx] = torch.stack(selected_data).mean(dim=0)
            
        return prototypes


# the buffer for storing the batch based data
class OnlineBuffer:
    def __init__(self, buffer_size: int):
        """
        Initializes a fixed-size FIFO buffer for streaming data.
        
        Args:
            buffer_size: Maximum capacity of the buffer (number of data samples)
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)  # FIFO queue that automatically discards old data
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

    def size(self) -> int:
        """Returns current number of data samples in the buffer"""
        return len(self.buffer)
    

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