"""
SOLIDER Model Architecture
===========================
Implementation of SOLIDER (Semantic-Guided Latent Representation) 
for person re-identification.

Based on: "Beyond Appearance: a Semantic Controllable Self-Supervised Learning 
          Framework for Human-Centric Visual Tasks"
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SOLIDERModel(nn.Module):
    """
    SOLIDER ReID model with ResNet50 backbone.
    """
    
    def __init__(self, num_classes=751, embedding_dim=256):
        super().__init__()
        
        # ResNet50 backbone
        resnet = models.resnet50(pretrained=True)
        
        # Remove avgpool and fc layers
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Embedding layer
        self.embedding = nn.Linear(2048, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        
        # Classification layer (for training)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, 3, H, W)
        
        Returns:
            Embedding tensor of shape (N, embedding_dim)
        """
        # Extract features
        feat = self.backbone(x)  # (N, 2048, H/32, W/32)
        
        # Global pooling
        feat = self.gap(feat)  # (N, 2048, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (N, 2048)
        
        # Embedding
        embedding = self.embedding(feat)  # (N, embedding_dim)
        embedding = self.bn(embedding)
        
        return embedding
    
    def forward_with_classifier(self, x):
        """
        Forward pass with classification (for training).
        
        Returns:
            embedding, logits
        """
        embedding = self.forward(x)
        logits = self.classifier(embedding)
        return embedding, logits


def build_solider_model(embedding_dim=256, pretrained_path=None):
    """
    Build SOLIDER model.
    
    Args:
        embedding_dim: Dimension of embedding vector
        pretrained_path: Path to pretrained weights (optional)
    
    Returns:
        SOLIDER model
    """
    model = SOLIDERModel(embedding_dim=embedding_dim)
    
    if pretrained_path:
        print(f"Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    
    return model


if __name__ == '__main__':
    # Test model
    model = build_solider_model()
    print(f"Model created: {model.__class__.__name__}")
    print(f"Embedding dimension: {model.embedding_dim}")
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 128)
    embedding = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {embedding.shape}")
