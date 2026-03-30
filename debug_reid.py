"""
Debug script to understand why ReID isn't working
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from boxmot.trackers.strongsort_persistent import StrongSORTPersistent
from boxmot.utils import WEIGHTS
import cv2
import numpy as np
from ultralytics import YOLO

# Configuration
VIDEO = "test_3.mp4"
YOLO_MODEL = "yolov8n.pt"
REID_MODEL = "osnet_x0_25_msmt17.pt"
REID_THRESHOLD = 0.5  # More lenient for debugging
CONF = 0.3  # Lower confidence to detect more

print("="*80)
print("🔍 ReID Debug Script")
print("="*80)

# Load models
print("\n1. Loading models...")
yolo = YOLO(YOLO_MODEL)
reid_weights = WEIGHTS / REID_MODEL

tracker = StrongSORTPersistent(
    model_weights=reid_weights,
    device='cpu',
    fp16=False,
    max_age=30,  # Shorter for faster deletion
    n_init=1,  # Confirm tracks immediately
    enable_persistent_reid=True,
    reid_threshold=REID_THRESHOLD,
    persistent_budget=500,
)

print(f"✅ Tracker initialized with reid_threshold={REID_THRESHOLD}")

# Open video
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"❌ Could not open {VIDEO}")
    sys.exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"✅ Video: {total_frames} frames @ {fps}fps")

print("\n2. Processing video...")
print("-"*80)

frame_idx = 0
track_history = {}  # Track when each ID was seen

try:
    while frame_idx < min(300, total_frames):  # Process first 300 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Detect
        results = yolo(frame, conf=CONF, verbose=False)[0]
        
        if len(results.boxes) > 0:
            dets = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                dets.append([x1, y1, x2, y2, conf, cls])
            dets = np.array(dets)
        else:
            dets = np.empty((0, 6))
        
        print(f"\n[Frame {frame_idx}] Detections: {len(dets)}")
        
        # Track
        tracks = tracker.update(dets, frame)
        
        # Record track history
        for track in tracks:
            track_id = int(track[4])
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append(frame_idx)
        
        if len(tracks) > 0:
            print(f"[Frame {frame_idx}] Active tracks: {[int(t[4]) for t in tracks]}")
        
        # Print gallery stats every 30 frames
        if frame_idx % 30 == 0:
            if hasattr(tracker.tracker.metric, 'get_stats'):
                stats = tracker.tracker.metric.get_stats()
                print(f"\n📊 Gallery Stats at frame {frame_idx}:")
                print(f"   Active IDs: {stats['active_ids']}")
                print(f"   Deleted IDs: {stats['deleted_ids']}")
                print(f"   Total IDs ever: {stats['total_ids_ever']}")
                print(f"   Total features: {stats['total_features']}")

finally:
    cap.release()

print("\n" + "="*80)
print("📊 Final Analysis")
print("="*80)

# Analyze track history
print(f"\nTotal unique IDs seen: {len(track_history)}")
print("\nTrack History:")
for track_id in sorted(track_history.keys()):
    frames = track_history[track_id]
    gaps = []
    for i in range(1, len(frames)):
        gap = frames[i] - frames[i-1]
        if gap > 1:
            gaps.append(gap)
    
    print(f"  ID {track_id}: Seen in {len(frames)} frames (first: {frames[0]}, last: {frames[-1]})")
    if gaps:
        print(f"          Gaps: {gaps} frames")

# Final gallery stats
if hasattr(tracker.tracker.metric, 'get_stats'):
    stats = tracker.tracker.metric.get_stats()
    print(f"\n📊 Final Gallery Stats:")
    print(f"   Active IDs: {stats['active_ids']}")
    print(f"   Deleted IDs (available for re-ID): {stats['deleted_ids']}")
    print(f"   Total IDs ever seen: {stats['total_ids_ever']}")
    print(f"   Total features stored: {stats['total_features']}")
    
    if stats['deleted_ids'] > 0:
        print(f"\n✅ {stats['deleted_ids']} IDs are available for re-identification!")
    else:
        print(f"\n⚠️ No deleted IDs yet - tracks may not have been deleted")

print("\n" + "="*80)
