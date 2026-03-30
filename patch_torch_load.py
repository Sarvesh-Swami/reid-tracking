"""
Temporary patch to fix torch.load weights_only issue in PyTorch 2.6+
Run this before running track.py
"""
import torch.serialization
from ultralytics.nn.tasks import DetectionModel, SegmentationModel, PoseModel

# Add safe globals for ultralytics models
torch.serialization.add_safe_globals([DetectionModel, SegmentationModel, PoseModel])

print("✓ Torch load patch applied successfully!")
print("Now you can run: python examples/track.py --source test_3.mp4 --tracking-method deepocsort --save")
