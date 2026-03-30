"""
Stable ID Tracking - Maintains IDs during brief occlusions and overlaps
Optimized for scenarios where people briefly disappear and reappear
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import DeepOCSORT
from boxmot.utils import WEIGHTS
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Video file')
    parser.add_argument('--output', type=str, default='output_stable_ids.mp4', help='Output video')
    parser.add_argument('--show', action='store_true', help='Display video')
    parser.add_argument('--conf', type=float, default=0.3, help='Detection confidence (lower = more detections)')
    args = parser.parse_args()
    
    print("="*80)
    print("🎯 Stable ID Tracking")
    print("="*80)
    print("Optimized for:")
    print("  ✓ Brief occlusions (people walking in front)")
    print("  ✓ Partial overlaps")
    print("  ✓ Temporary disappearances")
    print("="*80)
    
    # Load YOLO
    print("\nLoading YOLO model...")
    yolo = YOLO('yolov8n.pt')
    
    # Initialize DeepOCSORT with optimized settings
    print("Initializing DeepOCSORT tracker...")
    print("  - max_age: 100 frames (keeps tracks alive longer)")
    print("  - min_hits: 1 (confirms tracks immediately)")
    print("  - iou_threshold: 0.1 (lenient spatial matching)")
    print("  - Using ReID for appearance matching")
    
    tracker = DeepOCSORT(
        model_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',
        device='cpu',
        fp16=False,
        per_class=False,
        det_thresh=0.0,
        max_age=100,  # CRITICAL: Keep tracks alive for 100 frames
        min_hits=1,   # CRITICAL: Confirm immediately
        iou_threshold=0.1,  # CRITICAL: Very lenient IoU matching
        delta_t=3,
        asso_func="giou",  # Use GIoU for better matching
        inertia=0.2,
    )
    
    # Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open {args.source}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_idx = 0
    colors = {}
    track_lifetimes = {}  # Track how long each ID has been tracked
    
    print("\n🎬 Processing...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Detect with lower confidence
            results = yolo(frame, conf=args.conf, verbose=False)[0]
            
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
            
            # Track
            tracks = tracker.update(dets, frame)
            
            # Update track lifetimes
            current_ids = set()
            for track in tracks:
                track_id = int(track[4])
                current_ids.add(track_id)
                
                if track_id not in track_lifetimes:
                    track_lifetimes[track_id] = 0
                    print(f"  [Frame {frame_idx}] 🆕 New ID: {track_id}")
                
                track_lifetimes[track_id] += 1
                
                # Assign color
                if track_id not in colors:
                    colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
                
                color = colors[track_id]
                x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID with lifetime
                label = f"ID:{track_id} ({track_lifetimes[track_id]}f)"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Info overlay
            info = f"Frame: {frame_idx}/{total_frames} | Active: {len(tracks)} | Total IDs: {len(colors)}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write and show
            out.write(frame)
            if args.show:
                cv2.imshow('Stable ID Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"  Progress: {progress:.1f}% | Active: {len(tracks)} | Total IDs: {len(colors)}")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    print("\n" + "="*80)
    print("✅ Done!")
    print("="*80)
    print(f"Output: {args.output}")
    print(f"Total unique IDs: {len(colors)}")
    print("\nID Lifetimes (frames tracked):")
    for track_id in sorted(track_lifetimes.keys()):
        print(f"  ID {track_id}: {track_lifetimes[track_id]} frames")
    print("="*80)

if __name__ == '__main__':
    main()
