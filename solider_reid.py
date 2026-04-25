"""
SOLIDER ReID Model Wrapper
===========================
Wraps the SOLIDER (Semantic-Guided Latent Representation) model
to provide the same interface as OSNet for easy integration.

SOLIDER is a state-of-the-art person re-identification model with:
- Better view-invariance (handles front/back views better)
- Semantic guidance (uses body part information)
- 256-dimensional embeddings

Paper: "Beyond Appearance: a Semantic Controllable Self-Supervised Learning 
        Framework for Human-Centric Visual Tasks"
GitHub: https://github.com/tinyvision/SOLIDER-REID
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path


class SOLIDERReID:
    """
    SOLIDER ReID model wrapper that matches OSNet interface.
    """
    
    def __init__(self, model_path='weights/solider_market1501.pth', device='cuda'):
        """
        Initialize SOLIDER model.
        
        Args:
            model_path: Path to SOLIDER weights file
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_path = Path(model_path)
        
        # Check if model exists
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SOLIDER model not found at {model_path}\n"
                f"Please run: python download_solider.py"
            )
        
        # Load model
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = T.Compose([
            T.Resize((256, 128)),  # SOLIDER input size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ SOLIDER model loaded from {model_path}")
        print(f"  Device: {self.device}")
        print(f"  Embedding size: 256 dimensions")
    
    def _build_model(self):
        """Build SOLIDER model architecture."""
        # Import SOLIDER model
        try:
            from solider_model import build_solider_model
            model = build_solider_model()
        except ImportError:
            # Fallback: Use ResNet50 backbone if SOLIDER code not available
            print("⚠️  SOLIDER model code not found, using ResNet50 backbone")
            model = self._build_resnet50_reid()
        
        return model
    
    def _build_resnet50_reid(self):
        """
        Fallback: Build ResNet50-based ReID model.
        This is a simplified version if SOLIDER code is not available.
        """
        import torchvision.models as models
        
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # Remove classification layer
        modules = list(resnet.children())[:-1]
        backbone = nn.Sequential(*modules)
        
        # Add embedding layer
        class ReIDModel(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                self.embedding = nn.Linear(2048, 256)
                self.bn = nn.BatchNorm1d(256)
            
            def forward(self, x):
                x = self.backbone(x)
                x = x.view(x.size(0), -1)
                x = self.embedding(x)
                x = self.bn(x)
                return x
        
        model = ReIDModel(backbone)
        return model
    
    def __call__(self, images):
        """
        Extract features from images.
        
        Args:
            images: List of numpy arrays (BGR format) or single numpy array
        
        Returns:
            numpy array of shape (N, 256) containing embeddings
        """
        # Handle single image
        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                images = [images]
        
        # Convert BGR to RGB and preprocess
        batch = []
        for img in images:
            if isinstance(img, np.ndarray):
                # Convert BGR to RGB
                img_rgb = img[:, :, ::-1]
                img_pil = Image.fromarray(img_rgb)
            else:
                img_pil = img
            
            img_tensor = self.transform(img_pil)
            batch.append(img_tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(batch).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(batch_tensor)
        
        # Convert to numpy
        features_np = features.cpu().numpy()
        
        # Normalize features
        features_np = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-12)
        
        return features_np


def test_solider():
    """Test SOLIDER model loading and inference."""
    print("Testing SOLIDER model...")
    
    try:
        # Initialize model
        model = SOLIDERReID()
        
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        
        # Extract features
        features = model([dummy_img])
        
        print(f"✓ Feature extraction successful")
        print(f"  Input shape: {dummy_img.shape}")
        print(f"  Output shape: {features.shape}")
        print(f"  Feature norm: {np.linalg.norm(features[0]):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


if __name__ == '__main__':
    test_solider()
