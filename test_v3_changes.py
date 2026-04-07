"""
Quick test to verify v3 changes are working
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from track_attendance import AttendanceTracker, PersistentGallery
import numpy as np

def test_gallery_verification():
    """Test the new verification logic"""
    print("Testing PersistentGallery...")
    
    gallery = PersistentGallery(reid_threshold=0.40, color_weight=0.55)
    
    # Create mock features for 2 people
    person1_feat = np.random.randn(512).astype(np.float32)
    person1_feat /= np.linalg.norm(person1_feat)
    
    person2_feat = np.random.randn(512).astype(np.float32)
    person2_feat /= np.linalg.norm(person2_feat)
    
    # Add to gallery
    pid1 = gallery.add_new([person1_feat] * 5)
    pid2 = gallery.add_new([person2_feat] * 5)
    
    print(f"  Created Person {pid1} and Person {pid2}")
    
    # Test combined_score method
    score1, csim1, rsim1 = gallery.combined_score(person1_feat, pid1, None)
    score2, csim2, rsim2 = gallery.combined_score(person1_feat, pid2, None)
    
    print(f"  Person 1 feature vs Person 1 gallery: score={score1:.3f}")
    print(f"  Person 1 feature vs Person 2 gallery: score={score2:.3f}")
    
    assert score1 > score2, "Person 1 should match Person 1 better than Person 2"
    print("  ✓ Gallery scoring works correctly")
    
    return True

def test_tracker_initialization():
    """Test that tracker initializes with new parameters"""
    print("\nTesting AttendanceTracker initialization...")
    
    tracker = AttendanceTracker(
        yolo_model='yolov8n.pt',
        reid_model='osnet_x1_0_msmt17.pt',
        reid_threshold=0.40,
        color_weight=0.55
    )
    
    # Check new attributes exist
    assert hasattr(tracker, 'last_verified_frame'), "Missing last_verified_frame"
    assert hasattr(tracker, 'reassignment_events'), "Missing reassignment_events"
    assert hasattr(tracker, 'verify_interval'), "Missing verify_interval"
    assert hasattr(tracker, 'reassignment_threshold'), "Missing reassignment_threshold"
    
    print(f"  verify_interval: {tracker.verify_interval}")
    print(f"  reassignment_threshold: {tracker.reassignment_threshold}")
    print(f"  max_verify_fails: {tracker.max_verify_fails}")
    
    # Check new method exists
    assert hasattr(tracker, '_verify_track_identity'), "Missing _verify_track_identity method"
    
    print("  ✓ Tracker initialization successful")
    return True

def main():
    print("="*60)
    print("TESTING v3 CHANGES")
    print("="*60)
    
    try:
        test_gallery_verification()
        test_tracker_initialization()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nYou can now run:")
        print("  python track_attendance.py --source test_6.mp4 --output output_v3.mp4")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
