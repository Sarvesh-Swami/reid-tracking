"""
Wrapper script to run tracking with PyTorch 2.6+ compatibility
"""
import sys
import torch.serialization

# Add safe globals for ultralytics models
try:
    from ultralytics.nn.tasks import DetectionModel, SegmentationModel, PoseModel
    torch.serialization.add_safe_globals([DetectionModel, SegmentationModel, PoseModel])
    print("✓ PyTorch 2.6+ compatibility patch applied")
except Exception as e:
    print(f"Warning: Could not apply patch: {e}")

# Now import and run the tracking script
sys.path.insert(0, 'examples')
from track import main, parse_opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
