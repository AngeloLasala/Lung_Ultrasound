"""
Create the Binary Cross Entropy Loss for multi-label classification with 
pos weights support, inverse proportional to class frequencies.
"""
import torch
import torch.nn as nn


class WeightedBCELoss(nn.Module):
    """
    Binary Cross Entropy Loss with positive weight support.
    Gives more importance when the model predicts correctly the positive class (1).
    """
    
    def __init__(self, pos_weight=2.0):
        """
        Initialize the weighted BCE loss.
        
        Args:
            pos_weight (float or torch.Tensor): Weight for positive class.
                If float, same weight for all classes.
                If tensor, shape should be (num_classes,) for per-class weights.
                Higher values give more importance to correctly predicting 1s.
                Default: 2.0 (positive class is twice as important)
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, predictions, targets):
        """
        Compute weighted BCE loss.
        
        Args:
            predictions (torch.Tensor): Model predictions, shape (batch_size, num_classes)
                                       Values should be in range [0, 1] (use sigmoid)
            targets (torch.Tensor): Ground truth labels, shape (batch_size, num_classes)
                                   Binary values: [0., 0., 0., 1., 0.]
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        # BCEWithLogitsLoss includes sigmoid, so if your predictions are already sigmoid,
        # use BCELoss instead. Here using BCEWithLogitsLoss for numerical stability.
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight).to(predictions.device))
        loss = criterion(predictions, targets)
        
        return loss

def compute_pos_weights(labels, epsilon=1e-5):
    """
    Compute positive weights inverse proportional to class frequencies.
    
    Args:
        labels (torch.Tensor): All training labels, shape (num_samples, num_classes)
                              Binary values: [0., 0., 0., 1., 0.]
        epsilon (float): Small value to avoid division by zero
    
    Returns:
        torch.Tensor: Positive weights for each class, shape (num_classes,)
    """
    # Count positive samples for each class
    num_positives = labels.sum(dim=0)
    
    # Count negative samples for each class
    num_negatives = (1 - labels).sum(dim=0)
    
    # Compute weight as ratio of negatives to positives
    # More rare positive samples get higher weight
    pos_weights = (num_negatives + epsilon) / (num_positives + epsilon)
    
    return pos_weights

if __name__ == "__main__":
    ## Playgraound 

    batch_targets = torch.tensor([
        [0., 0., 0., 1., 0.],  # Rare class 3
        [1., 0., 0., 0., 0.],  # Frequent class 0
        [0., 1., 0., 1., 0.],  # Medium class 1 + rare class 3
        [0., 0., 1., 0., 1.]   # Less frequent class 2 + very rare class 4
    ])
    
    # Model predictions (logits, before sigmoid)
    batch_logits = torch.tensor([
        [-2.0, -1.5, -2.0, 1.5, -1.8],  # Good prediction for rare class 3
        [1.8, -1.2, -1.5, -2.0, -1.3],  # Good prediction for frequent class 0
        [-1.3, 1.2, -1.8, 1.6, -1.5],   # Good predictions
        [-1.5, -1.8, 1.7, -1.9, 1.8]    # Good predictions
    ])

    # Test with uniform weight
    print("\n=== WeightedBCELoss with uniform weight (pos_weight=2.0) ===")
    criterion_uniform = WeightedBCELoss(pos_weight=2.0)
    loss_uniform = criterion_uniform(batch_logits, batch_targets)
    print(f"Loss: {loss_uniform.item():.4f}")

    # Test with computed positive weights
    print("\n=== WeightedBCELoss with frequency-based weights ===")
    pos_weights = compute_pos_weights(batch_targets)
    print(pos_weights)
    criterion_weighted = WeightedBCELoss(pos_weight=pos_weights)
    loss_weighted = criterion_weighted(batch_logits, batch_targets)
    print(f"Loss: {loss_weighted.item():.4f}")
    print("The loss is higher because rare classes (3, 4) have more weight!")


