import random
import numpy as np
from easydict import EasyDict as edict
import torch
from collections import deque
import threading

# the memory buffer for the presudo source
class MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age

    def increase_age(self):
        if not self.empty():
            self.age += 1

    def get_data(self):
        return self.data, self.uncertainty, self.age

    def empty(self):
        return self.data == "empty"


class DropMemoryBank:
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
        x, prediction, uncertainty, confidence = (
            instance['data'], 
            instance['prediction'], 
            instance['uncertainty'],
            instance['confidence']
        )
        
        # Check for duplicate data across all classes
        for cls_idx, cls_queue in enumerate(self.data):
            if any(torch.equal(item.data, x) for item in cls_queue):
                print(f"Item already stored in memory bank for class {cls_idx}")
                return True
        
        # Create new memory item
        new_item = MemoryItem(data=x, uncertainty=uncertainty)
        
        # Add to corresponding class (automatically handles FIFO)
        self.data[prediction].append(new_item)
        return False

    def get_memory(self):
        """Retrieve all stored data in flattened format"""
        tmp_data = []
        tmp_uncertainty = []
        tmp_class = []
        for cls_idx, cls_queue in enumerate(self.data):
            for item in cls_queue:
                tmp_data.append(item.data)
                tmp_uncertainty.append(item.uncertainty)
                tmp_class.append(cls_idx)
        return tmp_data, tmp_uncertainty, tmp_class

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

    def size(self) -> int:
        """Returns current number of data samples in the buffer"""
        return len(self.buffer), len(self.buffer_instance)