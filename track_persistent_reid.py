"""
Track with Persistent Re-Identification
This script uses StrongSORT with a persistent gallery that remembers all persons
and reassigns the same ID when they reappear in the frame.
"""
import sys
import argparse
from pathlib import Path

# Add boxmot to path
sys.path.insert(0, str(Path(__file__).parent))

from boxmot.trackers.strongsort_persistent import StrongSORTPersistent
from boxmot.utils import WEIGHTS
import cv2
import numpy as np
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='Track with Persistent ReID')
    parser.add_argument('--source', type=str, required=True, help='Video file path')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt', help='YOLO model')
    parser.add_argument('--reid-model', type=str, default='osnet_x0_25_msmt17.pt', help='ReID model')
    parser.add_argument('--reid-threshold', type=float, default=0.4, help='ReID matching threshold (lower=stricter)')
    parser.add_argument('--conf', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--max-age', type=int, default=70, help='Max frames to keep lost tracks')
    parser.add_argument('--output', type=str, default='output_persistent_reid.mp4', help='Output video path')
    parser.add_argument('--show', action='store_true', help='Display video while processing')
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 Persistent Re-Identification Tracker")
    print("="*80)
    print(f"Video: {args.source}")
    print(f"YOLO Model: {args.yolo_model}")
    print(f"ReID Model: {args.reid_model}")
    print(f"ReID Threshold: {args.reid_threshold} (lower = stricter matching)")
    print(f"Max Age: {args.max_age} frames")
    print("="*80)
    
    # Load YOLO model
    print("Loading YOLO model...")
    yolo = YOLO(args.yolo_model)
    
    # Initialize persistent tracker
    print("Initializing Persistent ReID Tracker...")
    reid_weights = WEIGHTS / args.reid_model
    tracker = StrongSORTPersistent(
        model_weights=reid_weights,
        device='cuda:0' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu',
        fp16=False,
        max_age=args.max_age,
        enable_persistent_reid=True,
        reid_threshold=args.reid_threshold,
        persistent_budget=500,  # Keep 500 features per person
    )
    
    # Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {args.source}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_idx = 0
    colors = {}  # Track ID -> color mapping
    
    print("\n🎬 Processing video...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Run YOLO detection
            results = yolo(frame, conf=args.conf, verbose=False)[0]
            
            # Get detections
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
            
            # Update tracker
            tracks = tracker.update(dets, frame)
            
            # Draw tracks
            for track in tracks:
                x1, y1, x2, y2, track_id, conf, cls = track
                track_id = int(track_id)
                
                # Assign color to track ID
                if track_id not in colors:
                    colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
                
                color = colors[track_id]
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw ID label
                label = f"ID: {track_id}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (int(x1), int(y1) - label_height - 10), 
                            (int(x1) + label_width, int(y1)), color, -1)
                cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add frame info
            info_text = f"Frame: {frame_idx}/{total_frames} | Tracks: {len(tracks)} | Total IDs: {len(colors)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write frame
            out.write(frame)
            
            # Show frame
            if args.show:
                cv2.imshow('Persistent ReID Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames}) | Active tracks: {len(tracks)} | Total IDs seen: {len(colors)}")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    print(f"\n✅ Done! Output saved to: {args.output}")
    print(f"📊 Total unique IDs tracked: {len(colors)}")
    
    # Print gallery stats
    if hasattr(tracker.tracker.metric, 'get_stats'):
        stats = tracker.tracker.metric.get_stats()
        print(f"📊 Final Gallery Stats:")
        print(f"   - Active IDs: {stats['active_ids']}")
        print(f"   - Deleted IDs (available for re-ID): {stats['deleted_ids']}")
        print(f"   - Total IDs ever seen: {stats['total_ids_ever']}")
        print(f"   - Total features stored: {stats['total_features']}")


if __name__ == '__main__':
    main()
