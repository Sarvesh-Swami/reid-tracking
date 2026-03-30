"""
Ultimate Tracking Solution
Uses every trick possible to maintain IDs during occlusions
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import BoTSORT
from boxmot.utils import WEIGHTS
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_ultimate.mp4')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--yolo-model', type=str, default='yolov8x.pt', 
                       help='Use yolov8x for better detection')
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 ULTIMATE Tracking Solution")
    print("="*80)
    print("Using EVERY optimization:")
    print("  ✓ YOLOv8x (best detector)")
    print("  ✓ BoTSORT (appearance + motion)")
    print("  ✓ OSNet x1.0 (best ReID)")
    print("  ✓ Very low confidence (0.15)")
    print("  ✓ 300 frame track buffer (10 seconds)")
    print("  ✓ Lenient matching thresholds")
    print("="*80)
    
    # Load YOLO
    print(f"\nLoading {args.yolo_model}...")
    yolo = YOLO(args.yolo_model)
    
    # BoTSORT with EXTREME settings
    print("Initializing BoTSORT...")
    tracker = BoTSORT(
        model_weights=WEIGHTS / 'osnet_x1_0_msmt17.pt',  # Best ReID model
        device='cpu',
        fp16=False,
        track_high_thresh=0.3,      # Low threshold for high confidence tracks
        new_track_thresh=0.2,       # Low threshold for new tracks
        track_buffer=300,           # Keep tracks for 300 frames (10 seconds!)
        match_thresh=0.95,          # Very lenient matching
        proximity_thresh=0.8,       # Lenient proximity
        appearance_thresh=0.8,      # Lenient appearance
        cmc_method='sparseOptFlow', # Camera motion compensation
        frame_rate=30,
        lambda_=0.985               # Balance appearance and motion
    )
    
    # Open video
    cap = cv2.VideoCapture(args.source)
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
    id_first_seen = {}
    id_last_seen = {}
    
    print("\n🎬 Processing...\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Detect with VERY low confidence
            results = yolo(frame, conf=0.15, classes=[0], verbose=False)[0]  # Only person class
            
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
            
            # Track ID changes
            current_ids = set()
            for track in tracks:
                track_id = int(track[4])
                current_ids.add(track_id)
                
                if track_id not in id_first_seen:
                    id_first_seen[track_id] = frame_idx
                    print(f"[Frame {frame_idx}] 🆕 New ID: {track_id}")
                
                id_last_seen[track_id] = frame_idx
                
                if track_id not in colors:
                    colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
                
                color = colors[track_id]
                x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
                conf = track[5]
                
                # Draw box (thicker for better visibility)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw ID with lifetime
                lifetime = frame_idx - id_first_seen[track_id]
                label = f"ID:{track_id} ({lifetime}f)"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1-h-15), (x1+w+10, y1), color, -1)
                cv2.putText(frame, label, (x1+5, y1-8), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                
                # Draw confidence
                conf_text = f"{conf:.2f}"
                cv2.putText(frame, conf_text, (x1, y2+20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Info overlay
            info = f"Frame:{frame_idx}/{total_frames} | Detections:{len(dets)} | Tracks:{len(tracks)} | Total IDs:{len(colors)}"
            cv2.rectangle(frame, (5, 5), (width-5, 50), (0, 0, 0), -1)
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            out.write(frame)
            if args.show:
                cv2.imshow('Ultimate Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx}/{total_frames} | Detections: {len(dets)} | Tracks: {len(tracks)} | Total IDs: {len(colors)}")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    print("\n" + "="*80)
    print("✅ Processing Complete!")
    print("="*80)
    print(f"Total unique IDs: {len(colors)}")
    print(f"\nID Summary:")
    for track_id in sorted(id_first_seen.keys()):
        first = id_first_seen[track_id]
        last = id_last_seen[track_id]
        duration = last - first + 1
        print(f"  ID {track_id}: Frames {first}-{last} (duration: {duration} frames)")
    print(f"\nOutput: {args.output}")
    print("="*80)
    
    # Analysis
    if len(colors) > 10:
        print("\n⚠️  WARNING: Too many IDs detected!")
        print("This suggests:")
        print("  - Video has many people, OR")
        print("  - IDs are still switching (tracking failing)")
        print("\nIf IDs are switching, the video conditions may be too challenging:")
        print("  - Too crowded")
        print("  - Occlusions too long")
        print("  - Poor lighting/quality")
        print("  - Consider using commercial solution")

if __name__ == '__main__':
    main()
