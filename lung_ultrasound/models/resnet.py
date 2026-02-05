"""
Create ResNet model fod LUS milti instance classification compatible with CAM computation
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetLUSCAM(nn.Module):
    """
    ResNet model for Lung Ultrasound (LUS) multi-instance classification
    with support for Class Activation Maps (CAM) computation.
    
    This model uses a pre-trained ResNet backbone and modifies it for
    multi-instance learning scenarios common in medical imaging.
    """
    
    def __init__(self, num_classes=2, backbone='resnet50', pretrained=True, 
                 freeze_backbone=False, pooling='avg'):
        """
        Initialize ResNet model for LUS classification.
        
        Args:
            num_classes (int): Number of output classes
            backbone (str): ResNet variant ('resnet18', 'resnet34', 'resnet50', 
                           'resnet101', 'resnet152')
            pretrained (bool): Whether to use ImageNet pretrained weights
            freeze_backbone (bool): Whether to freeze backbone weights during training
            pooling (str): Global pooling type ('avg' or 'max')
        """
        super(ResNetLUSCAM, self).__init__()
        
        # Load appropriate ResNet backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the original fully connected layer and avgpool
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Global pooling layer
        if pooling == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")
        
        # Classification layer (important for CAM computation)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # Store intermediate feature maps for CAM
        self.feature_maps = None
        self.gradients = None
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Class predictions of shape (batch_size, num_classes)
        """
        # Extract feature maps from backbone
        self.feature_maps = self.features(x)
        
        # Register hook for gradient computation (needed for Grad-CAM)
        if self.training or self.feature_maps.requires_grad:
            self.feature_maps.register_hook(self.save_gradient)
        
        # Global pooling
        pooled_features = self.global_pool(self.feature_maps)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Classification
        output = self.classifier(pooled_features)
        
        return output
    
    def save_gradient(self, grad):
        """
        Hook function to save gradients during backward pass.
        Used for Grad-CAM computation.
        
        Args:
            grad (torch.Tensor): Gradient tensor
        """
        self.gradients = grad
    
    def get_cam_weights(self):
        """
        Get CAM weights from the classifier layer.
        
        Returns:
            torch.Tensor: Weights of shape (num_classes, feature_dim)
        """
        return self.classifier.weight.data
    
    def get_feature_maps(self):
        """
        Get the last convolutional feature maps.
        
        Returns:
            torch.Tensor: Feature maps from last conv layer
        """
        return self.feature_maps
    
    def get_gradients(self):
        """
        Get gradients of feature maps (for Grad-CAM).
        
        Returns:
            torch.Tensor: Gradients of feature maps
        """
        return self.gradients
    
    def generate_cam(self, class_idx=None):
        """
        Generate Class Activation Map for a specific class.
        
        Args:
            class_idx (int, optional): Class index for CAM generation.
                                      If None, uses the predicted class.
        
        Returns:
            torch.Tensor: CAM of shape (batch_size, height, width)
        """
        if self.feature_maps is None:
            raise RuntimeError("No feature maps available. Run forward pass first.")
        
        # Get weights for the target class
        weights = self.get_cam_weights()
        
        if class_idx is not None:
            weights = weights[class_idx].unsqueeze(0)
        
        # Compute CAM: weighted sum of feature maps
        # feature_maps shape: (batch_size, channels, height, width)
        # weights shape: (num_classes, channels) or (1, channels)
        
        batch_size, num_channels, height, width = self.feature_maps.shape
        
        # Reshape feature maps for matrix multiplication
        feature_maps_flat = self.feature_maps.view(batch_size, num_channels, -1)
        
        # Compute weighted combination
        if class_idx is not None:
            cam = torch.matmul(weights, feature_maps_flat)
            cam = cam.view(batch_size, height, width)
        else:
            # Generate CAM for all classes
            cam = torch.matmul(weights, feature_maps_flat)
            cam = cam.view(-1, batch_size, height, width)
            cam = cam.permute(1, 0, 2, 3)  # (batch_size, num_classes, height, width)
        
        # Apply ReLU to focus on positive activations
        cam = torch.relu(cam)
        
        return cam

if __name__ == "__main__":
    model = ResNetLUSCAM(num_classes=6, backbone='resnet152', pretrained=True)

    a = torch.randn(2, 3, 256, 256)
    output = model(a)
    print(output.shape)

    cam = model.generate_cam(0)
    print(cam.shape)