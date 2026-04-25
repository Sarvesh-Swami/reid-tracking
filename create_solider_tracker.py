"""
Create SOLIDER-based Tracker
=============================
This script creates track_attendance_solider.py by copying track_attendance.py
and modifying it to use SOLIDER instead of OSNet.

Run this after downloading SOLIDER weights.
"""

import shutil
from pathlib import Path


def create_solider_tracker():
    """Create SOLIDER-based tracker from OSNet tracker."""
    
    print("=" * 60)
    print("CREATING SOLIDER-BASED TRACKER")
    print("=" * 60)
    
    # Source and destination
    source_file = Path("track_attendance.py")
    dest_file = Path("track_attendance_solider.py")
    
    # Check source exists
    if not source_file.exists():
        print(f"✗ Source file not found: {source_file}")
        return False
    
    print(f"\nReading: {source_file}")
    
    # Read source file
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Modifications to make
    modifications = [
        # Update docstring
        ('"""', '"""\nSOLIDER-Based ', 1),
        ('Attendance Tracker v3', 'Attendance Tracker v3 (SOLIDER ReID)', 1),
        
        # Update imports - add SOLIDER import
        ('from boxmot.utils import WEIGHTS',
         'from boxmot.utils import WEIGHTS\nfrom solider_reid import SOLIDERReID'),
        
        # Update reid model initialization
        ("reid_model='osnet_x1_0_msmt17.pt'",
         "reid_model='solider'"),
        
        # Update model loading comment
        ('self.reid_model_name = reid_model',
         'self.reid_model_name = reid_model  # "solider" for SOLIDER model'),
        
        # Update tracker initialization
        ('reid_weights = WEIGHTS / self.reid_model_name',
         '''# Use SOLIDER instead of OSNet
        if self.reid_model_name == 'solider':
            # SOLIDER model will be loaded separately
            reid_weights = None
        else:
            reid_weights = WEIGHTS / self.reid_model_name'''),
        
        # Update model assignment
        ('self.reid_model = self.tracker.model',
         '''# Load ReID model
        if self.reid_model_name == 'solider':
            self.reid_model = SOLIDERReID(device=self.device)
            print("  Using SOLIDER ReID model")
        else:
            self.reid_model = self.tracker.model
            print("  Using OSNet ReID model")'''),
        
        # Update print statement
        ('print(f"  Device: {self.device}")',
         'print(f"  Device: {self.device}")\\n        print(f"  ReID Model: {self.reid_model_name}")'),
    ]
    
    # Apply modifications
    modified_content = content
    for old, new, *count in modifications:
        if count:
            modified_content = modified_content.replace(old, new, count[0])
        else:
            modified_content = modified_content.replace(old, new)
    
    # Add header comment
    header = '''"""
SOLIDER-Based Person Tracking
==============================
This is a modified version of track_attendance.py that uses SOLIDER ReID model
instead of OSNet for better view-invariance and front/back profile handling.

SOLIDER (Semantic-Guided Latent Representation) provides:
- Better view-invariance (handles 360° rotation better)
- Semantic guidance (uses body part information)
- 256-dimensional embeddings

Expected improvement: 12 → 7-9 persons (25-42% reduction in duplicates)

Usage:
    python track_attendance_solider.py --source test_6.mp4 --output output_solider.mp4

Compare with OSNet:
    python track_attendance.py --source test_6.mp4 --output output_osnet.mp4
    python track_attendance_solider.py --source test_6.mp4 --output output_solider.mp4

Requirements:
    - SOLIDER model weights (run: python download_solider.py)
    - solider_reid.py module
    - solider_model.py module
"""

'''
    
    modified_content = header + modified_content
    
    # Write destination file
    print(f"Writing: {dest_file}")
    with open(dest_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"\n✓ Created: {dest_file}")
    print(f"  Size: {dest_file.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "=" * 60)
    print("SOLIDER TRACKER CREATED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Make sure SOLIDER weights are downloaded:")
    print("   python download_solider.py")
    print()
    print("2. Run tracking with SOLIDER:")
    print("   python track_attendance_solider.py --source test_6.mp4")
    print()
    print("3. Compare results:")
    print("   OSNet:    output_v31.mp4 (12 persons)")
    print("   SOLIDER:  output_solider.mp4 (7-9 persons expected)")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    create_solider_tracker()
