"""
STEP 3: Re-track video using gallery with CONTINUOUS validation
Extracts RAW embeddings and continuously validates against gallery
"""
import sys
import pickle
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from boxmot import DeepOCSORT
from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend
import torch

def gallery_tracking_v2(video_path, gallery_file='gallery.pkl', output_video='output_persistent.mp4', 
                        reid_model='osnet_x1_0_msmt17.pt', match_threshold=0.35, validation_interval=10):
    """
    Track video using gallery with CONTINUOUS validation
    
    Parameters:
        video_path: Input video
        gallery_file: Gallery from step2
        output_video: Output video with persistent IDs
        reid_model: ReID model
        match_threshold: Maximum distance to match against gallery (LOWER = stricter)
        validation_interval: Check gallery every N frames (default: 10)
    """
    print("=" * 80)
    print("STEP 3 V2: GALLERY-BASED TRACKING WITH CONTINUOUS VALIDATION")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Gallery: {gallery_file}")
    print(f"Output: {output_video}")
    print(f"Match Threshold: {match_threshold}")
    print(f"Validation Interval: Every {validation_interval} frames")
    print("=" * 80)
    
    # Load gallery
    print("\n📂 Loading gallery...")
    with open(gallery_file, 'rb') as f:
        gallery_data = pickle.load(f)
    
    gallery = gallery_data['gallery']
    person_info = gallery_data['person_info']
    
    print(f"   Loaded gallery with {len(gallery)} persons")
    for person_id, info in person_info.items():
        print(f"   Person {person_id}: {info['total_embeddings']} embeddings")
    
    # Precompute gallery embeddings (use ALL embeddings, not just average)
    print("\n🧮 Preparing gallery for matching...")
    gallery_all_embeddings = {}  # {person_id: [emb1, emb2, ...]}
    
    for person_id in sorted(gallery.keys()):
        embs = np.array(gallery[person_id])
        gallery_all_embeddings[person_id] = embs
        print(f"   Person {person_id}: {len(embs)} embeddings ready for matching")
    
    # Load models
    print("\n📦 Loading models...")
    yolo = YOLO('yolov8n.pt')
    
    # Load ReID model separately for RAW embedding extraction
    device = 'cpu'
    reid_embedder = ReIDDetectMultiBackend(weights=reid_model, device=torch.device(device), fp16=False)
    
    # Tracker for ID assignment only
    tracker = DeepOCSORT(
        model_weights=reid_model,
        device=device,
        fp16=False,
        max_age=30,
        min_hits=1,
        iou_threshold=0.3,
    )
    
    # Open video
    print(f"\n🎬 Processing video...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Resolution: {width}x{height}")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Track ID mapping: {temp_track_id: gallery_person_id}
    track_to_person = {}
    track_last_validated = {}  # {track_id: frame_number}
    
    # Statistics
    stats = {
        'frames_processed': 0,
        'detections': 0,
        'initial_matches': 0,
        'validations': 0,
        'corrections': 0,
        'new_tracks': 0
    }
    
    frame_idx = 0
    
    def match_to_gallery(embedding, person_id_hint=None):
        """Match embedding against gallery using minimum distance across ALL embeddings"""
        best_person_id = None
        best_distance = float('inf')
        
        # If hint provided, check that person first
        check_order = [person_id_hint] + [pid for pid in gallery_all_embeddings.keys() if pid != person_id_hint] if person_id_hint else list(gallery_all_embeddings.keys())
        
        for person_id in check_order:
            person_embs = gallery_all_embeddings[person_id]
            
            # Compute distance to ALL embeddings, use MINIMUM
            min_dist = float('inf')
            for gallery_emb in person_embs:
                similarity = np.dot(embedding, gallery_emb)
                distance = 1.0 - similarity
                if distance < min_dist:
                    min_dist = distance
            
            if min_dist < best_distance:
                best_distance = min_dist
                best_person_id = person_id
        
        if best_distance < match_threshold:
            return best_person_id, best_distance
        else:
            return None, best_distance
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Detect people
        results = yolo(frame, classes=[0], conf=0.2, verbose=False)
        
        if len(results[0].boxes) == 0:
            out.write(frame)
            if frame_idx % 100 == 0:
                print(f"   Frame {frame_idx}/{total_frames} - No detections")
            continue
        
        # Get detections
        dets = results[0].boxes.data.cpu().numpy()
        
        # Track
        tracks = tracker.update(dets, frame)
        
        if len(tracks) == 0:
            out.write(frame)
            if frame_idx % 100 == 0:
                print(f"   Frame {frame_idx}/{total_frames} - No tracks")
            continue
        
        stats['frames_processed'] += 1
        stats['detections'] += len(tracks)
        
        # Process each track
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls = track
            track_id = int(track_id)
            
            # Crop person from frame
            x1_crop, y1_crop, x2_crop, y2_crop = int(x1), int(y1), int(x2), int(y2)
            x1_crop, y1_crop = max(0, x1_crop), max(0, y1_crop)
            x2_crop, y2_crop = min(width, x2_crop), min(height, y2_crop)
            
            if x2_crop <= x1_crop or y2_crop <= y1_crop:
                continue
            
            person_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            
            if person_crop.size == 0:
                continue
            
            # Extract RAW embedding
            try:
                raw_emb = reid_embedder([person_crop]).cpu().numpy()
                raw_emb = np.asarray(raw_emb).flatten()
                raw_emb = raw_emb / (np.linalg.norm(raw_emb) + 1e-8)
            except:
                raw_emb = None
            
            # Initial match (when track first appears)
            if track_id not in track_to_person and raw_emb is not None:
                person_id, distance = match_to_gallery(raw_emb)
                
                if person_id is not None:
                    track_to_person[track_id] = person_id
                    track_last_validated[track_id] = frame_idx
                    stats['initial_matches'] += 1
                    print(f"   ✅ Frame {frame_idx}: Track {track_id} → Person {person_id} (distance: {distance:.3f})")
                else:
                    # No match - assign new ID
                    new_person_id = max(gallery_all_embeddings.keys()) + stats['new_tracks'] + 1
                    track_to_person[track_id] = new_person_id
                    track_last_validated[track_id] = frame_idx
                    stats['new_tracks'] += 1
                    print(f"   🆕 Frame {frame_idx}: Track {track_id} → New Person {new_person_id} (min distance: {distance:.3f})")
            
            # CONTINUOUS VALIDATION: Re-check against gallery periodically
            elif track_id in track_to_person and raw_emb is not None:
                frames_since_validation = frame_idx - track_last_validated.get(track_id, 0)
                
                if frames_since_validation >= validation_interval:
                    current_person_id = track_to_person[track_id]
                    
                    # Re-match against gallery
                    person_id, distance = match_to_gallery(raw_emb, person_id_hint=current_person_id)
                    
                    track_last_validated[track_id] = frame_idx
                    stats['validations'] += 1
                    
                    # Check if ID needs correction
                    if person_id is not None and person_id != current_person_id:
                        print(f"   🔄 Frame {frame_idx}: Track {track_id} CORRECTED: Person {current_person_id} → Person {person_id} (distance: {distance:.3f})")
                        track_to_person[track_id] = person_id
                        stats['corrections'] += 1
            
            # Get persistent ID
            person_id = track_to_person.get(track_id, track_id)
            
            # Draw on frame
            x1_draw, y1_draw, x2_draw, y2_draw = int(x1), int(y1), int(x2), int(y2)
            
            # Color based on person ID
            color = (int((person_id * 50) % 255), int((person_id * 100) % 255), int((person_id * 150) % 255))
            
            # Draw box
            cv2.rectangle(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), color, 2)
            
            # Draw ID
            label = f"ID: {person_id}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1_draw, y1_draw - label_h - 10), (x1_draw + label_w, y1_draw), color, -1)
            cv2.putText(frame, label, (x1_draw, y1_draw - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Progress
        if frame_idx % 100 == 0:
            print(f"   ⏱️  Frame {frame_idx}/{total_frames} - {len(tracks)} tracks, {stats['corrections']} corrections so far")
    
    cap.release()
    out.release()
    
    # Summary
    print(f"\n✅ Tracking complete!")
    print(f"\n📊 Statistics:")
    print(f"   Frames processed: {stats['frames_processed']}")
    print(f"   Total detections: {stats['detections']}")
    print(f"   Initial gallery matches: {stats['initial_matches']}")
    print(f"   Continuous validations: {stats['validations']}")
    print(f"   ID corrections made: {stats['corrections']}")
    print(f"   New persons (not in gallery): {stats['new_tracks']}")
    print(f"   Total unique persons: {len(set(track_to_person.values()))}")
    print(f"\n💾 Output saved to: {output_video}")
    print("=" * 80)
    
    return track_to_person

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Track video using gallery with continuous validation')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--gallery', type=str, default='gallery.pkl', help='Gallery file from step2')
    parser.add_argument('--output', type=str, default='output_persistent_v2.mp4', help='Output video file')
    parser.add_argument('--reid-model', type=str, default='osnet_x1_0_msmt17.pt', help='ReID model')
    parser.add_argument('--threshold', type=float, default=0.35, help='Match threshold (lower = stricter, default: 0.35)')
    parser.add_argument('--validation-interval', type=int, default=10, help='Validate every N frames (default: 10)')
    
    args = parser.parse_args()
    
    gallery_tracking_v2(args.video, args.gallery, args.output, args.reid_model, args.threshold, args.validation_interval)
