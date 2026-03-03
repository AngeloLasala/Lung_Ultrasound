"""
Create a Cross Entropy Loss for multi-class classification with class weights support,
"""
import torch
import torch.nn as nn

class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss with class weight support.
    Gives more importance to correctly predicting classes with higher weights.
    """
    
    def __init__(self, class_weights=None):
        """
        Initialize the weighted Cross Entropy loss.
        
        Args:
            class_weights (torch.Tensor): Weights for each class, shape (num_classes,)
                                          Higher values give more importance to correctly predicting those classes.
                                          Default: None (no weighting)
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        
    def forward(self, predictions, targets):
        """
        Compute weighted Cross Entropy loss.
        
        Args:
            predictions (torch.Tensor): Model predictions (logits), shape (batch_size, num_classes)
            targets (torch.Tensor): Ground truth labels, shape (batch_size,)
                                    Integer values in range [0, num_classes-1]
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        if self.class_weights is not None:
            # Ensure class_weights is on the same device as predictions
            class_weights = self.class_weights.to(predictions.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        loss = criterion(predictions, targets)
        
        return loss

def compute_class_weights(labels, num_classes, epsilon=1e-5):
    """
    Compute class weights inverse proportional to class frequencies.
    
    Args:
        labels (torch.Tensor): All training labels, shape (num_samples,)
                               Integer values in range [0, num_classes-1]
        num_classes (int): Total number of classes
        epsilon (float): Small value to avoid division by zero
    
    Returns:
        torch.Tensor: Class weights, shape (num_classes,)
    """
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    total_samples = labels.size(0)
    
    # Compute weights: inverse of frequency
    class_weights = total_samples / (class_counts + epsilon)
    
    return class_weights