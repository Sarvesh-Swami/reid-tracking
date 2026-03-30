"""
Ultra-Simple Stable Tracking
Uses VERY lenient matching to maintain IDs during occlusions
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import BYTETracker  # Simplest, fastest tracker
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_simple.mp4')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    
    print("="*80)
    print("🎯 Ultra-Simple Stable Tracking")
    print("="*80)
    print("Strategy: VERY lenient matching + long track lifetime")
    print("="*80)
    
    # Load YOLO
    yolo = YOLO('yolov8n.pt')
    
    # ByteTrack with EXTREME settings for stability
    tracker = BYTETracker(
        track_thresh=0.1,      # Very low - accept weak detections
        match_thresh=0.9,      # Very high - very lenient matching
        track_buffer=300,      # Keep tracks for 300 frames (10 seconds at 30fps)
        frame_rate=30
    )
    
    print("Tracker settings:")
    print("  - track_buffer: 300 frames (keeps IDs alive for 10 seconds!)")
    print("  - match_thresh: 0.9 (very lenient IoU matching)")
    print("  - track_thresh: 0.1 (accepts weak detections)")
    
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
    
    print("\n🎬 Processing...\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Detect with VERY low confidence
            results = yolo(frame, conf=0.2, verbose=False)[0]
            
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
            
            # Draw
            for track in tracks:
                x1, y1, x2, y2, track_id, conf, cls = track
                track_id = int(track_id)
                
                if track_id not in colors:
                    colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
                    print(f"[Frame {frame_idx}] New ID: {track_id}")
                
                color = colors[track_id]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                label = f"ID:{track_id}"
                cv2.rectangle(frame, (int(x1), int(y1)-25), (int(x1)+80, int(y1)), color, -1)
                cv2.putText(frame, label, (int(x1)+5, int(y1)-8), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            # Info
            info = f"Frame:{frame_idx}/{total_frames} | Active:{len(tracks)} | Total IDs:{len(colors)}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            out.write(frame)
            if args.show:
                cv2.imshow('Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx}/{total_frames} | Active: {len(tracks)} | Total IDs: {len(colors)}")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    print(f"\n✅ Done! Total unique IDs: {len(colors)}")
    print(f"Output: {args.output}")

if __name__ == '__main__':
    main()
