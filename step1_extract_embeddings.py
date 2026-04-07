"""
STEP 1: Extract embeddings for all tracks in the video
Runs tracking and saves all embeddings per track to a file
"""
import sys
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from boxmot import DeepOCSORT

def extract_embeddings(video_path, output_file='embeddings.pkl', reid_model='osnet_x1_0_msmt17.pt'):
    """
    Extract all embeddings from video - RAW embeddings directly from ReID model
    
    Returns:
        embeddings_dict: {track_id: [emb1, emb2, emb3, ...]}
    """
    print("=" * 80)
    print("STEP 1: EXTRACTING RAW EMBEDDINGS FROM VIDEO")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"ReID Model: {reid_model}")
    print(f"Output: {output_file}")
    print("⚠️  IMPORTANT: Extracting RAW embeddings (not smoothed by tracker)")
    print("=" * 80)
    
    # Load models
    print("\n📦 Loading models...")
    yolo = YOLO('yolov8n.pt')
    
    # Load ReID model separately (not through tracker)
    from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend
    import torch
    
    device = 'cpu'
    reid_embedder = ReIDDetectMultiBackend(weights=reid_model, device=torch.device(device), fp16=False)
    print(f"   ✅ ReID model loaded: {reid_model}")
    
    # Create tracker for ID assignment only (we'll extract embeddings ourselves)
    tracker = DeepOCSORT(
        model_weights=reid_model,
        device=device,
        fp16=False,
        max_age=30,  # Shorter - we only need frame-to-frame tracking
        min_hits=1,
        iou_threshold=0.3,
    )
    
    # Storage for embeddings
    track_embeddings = defaultdict(list)  # {track_id: [embeddings]}
    track_info = {}  # {track_id: {'first_frame': X, 'last_frame': Y, 'count': Z}}
    
    # Process video
    print(f"\n🎬 Processing video...")
    import cv2
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Resolution: {width}x{height}")
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Detect people
        results = yolo(frame, classes=[0], conf=0.2, verbose=False)  # class 0 = person
        
        if len(results[0].boxes) == 0:
            if frame_idx % 100 == 0:
                print(f"   Frame {frame_idx}/{total_frames} - No detections")
            continue
        
        # Get detections
        dets = results[0].boxes.data.cpu().numpy()
        
        # Track (for ID assignment only)
        tracks = tracker.update(dets, frame)
        
        if len(tracks) == 0:
            if frame_idx % 100 == 0:
                print(f"   Frame {frame_idx}/{total_frames} - No tracks")
            continue
        
        # Extract RAW embeddings directly from ReID model for each track
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls = track
            track_id = int(track_id)
            
            # Crop person from frame
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                continue
            
            # Extract RAW embedding directly from ReID model
            try:
                raw_emb = reid_embedder([person_crop]).cpu().numpy()
                raw_emb = np.asarray(raw_emb).flatten()
                
                # Normalize
                emb_norm = raw_emb / (np.linalg.norm(raw_emb) + 1e-8)
                emb_norm_value = np.linalg.norm(emb_norm)
                
                # Check if this is a new track
                is_new_track = track_id not in track_embeddings
                
                # Store RAW embedding
                track_embeddings[track_id].append(emb_norm)
                
                # Update info
                if track_id not in track_info:
                    track_info[track_id] = {
                        'first_frame': frame_idx,
                        'last_frame': frame_idx,
                        'count': 0
                    }
                track_info[track_id]['last_frame'] = frame_idx
                track_info[track_id]['count'] += 1
                
                # Detailed logging
                total_embs = len(track_embeddings[track_id])
                if is_new_track:
                    print(f"   🆕 Frame {frame_idx:4d}: Track {track_id:2d} NEW TRACK → RAW Embedding {emb_norm.shape} norm={emb_norm_value:.3f} → Stored (total: {total_embs})")
                else:
                    # Log every 10th embedding for existing tracks to avoid spam
                    if total_embs % 10 == 0 or total_embs <= 3:
                        print(f"   📊 Frame {frame_idx:4d}: Track {track_id:2d} → RAW Embedding {emb_norm.shape} norm={emb_norm_value:.3f} → Stored (total: {total_embs})")
            except Exception as e:
                print(f"   ⚠️  Frame {frame_idx}: Failed to extract embedding for Track {track_id}: {e}")
                continue
        
        # Progress
        if frame_idx % 100 == 0:
            print(f"   ⏱️  Frame {frame_idx}/{total_frames} - {len(tracks)} active tracks, {len(track_embeddings)} unique IDs so far")
    
    cap.release()
    
    # Summary
    print(f"\n✅ Extraction complete!")
    print(f"   Total unique track IDs: {len(track_embeddings)}")
    print(f"   Total embeddings extracted: {sum(len(embs) for embs in track_embeddings.values())}")
    
    print(f"\n📊 Track Statistics:")
    for track_id in sorted(track_embeddings.keys()):
        info = track_info[track_id]
        num_embs = len(track_embeddings[track_id])
        duration = info['last_frame'] - info['first_frame']
        print(f"   Track {track_id:3d}: {num_embs:4d} embeddings | "
              f"Frames {info['first_frame']:4d}-{info['last_frame']:4d} ({duration:4d} frames)")
    
    # Save to file
    print(f"\n💾 Saving to {output_file}...")
    data = {
        'embeddings': dict(track_embeddings),
        'info': track_info,
        'video': video_path,
        'reid_model': reid_model,
        'total_frames': total_frames,
        'fps': fps
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Saved!")
    print("=" * 80)
    
    return data

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract embeddings from video')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default='embeddings.pkl', help='Output pickle file')
    parser.add_argument('--reid-model', type=str, default='osnet_x1_0_msmt17.pt', help='ReID model')
    
    args = parser.parse_args()
    
    extract_embeddings(args.video, args.output, args.reid_model)
