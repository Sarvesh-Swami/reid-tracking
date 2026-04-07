"""
STEP 3: Re-track video using gallery
Runs tracking again but matches against the clustered gallery for persistent IDs
"""
import sys
import pickle
import numpy as np
import cv2
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from boxmot import DeepOCSORT

def gallery_tracking(video_path, gallery_file='gallery.pkl', output_video='output_persistent.mp4', 
                     reid_model='osnet_x1_0_msmt17.pt', match_threshold=0.45):
    """
    Track video using gallery for persistent IDs
    
    Parameters:
        video_path: Input video
        gallery_file: Gallery from step2
        output_video: Output video with persistent IDs
        reid_model: ReID model
        match_threshold: Maximum distance to match against gallery
    """
    print("=" * 80)
    print("STEP 3: GALLERY-BASED TRACKING")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Gallery: {gallery_file}")
    print(f"Output: {output_video}")
    print(f"Match Threshold: {match_threshold}")
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
    
    # Precompute gallery embeddings (average per person)
    print("\n🧮 Computing gallery embeddings...")
    gallery_ids = []
    gallery_embeddings = []
    
    for person_id in sorted(gallery.keys()):
        embs = np.array(gallery[person_id])
        # Use average embedding for matching
        avg_emb = np.mean(embs, axis=0)
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
        
        gallery_ids.append(person_id)
        gallery_embeddings.append(avg_emb)
    
    gallery_embeddings = np.array(gallery_embeddings)
    print(f"   Gallery ready: {len(gallery_ids)} persons")
    
    # Load models
    print("\n📦 Loading models...")
    yolo = YOLO('yolov8n.pt')
    tracker = DeepOCSORT(
        model_weights=reid_model,
        device='cpu',
        fp16=False,
        max_age=30,  # Shorter for initial tracking
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
    
    # Statistics
    stats = {
        'frames_processed': 0,
        'detections': 0,
        'matches': 0,
        'new_tracks': 0
    }
    
    frame_idx = 0
    
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
        
        # Match tracks to gallery
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls = track
            track_id = int(track_id)
            
            # Get embedding for this track
            track_emb = None
            for trk in tracker.trackers:
                if trk.id + 1 == track_id:
                    if hasattr(trk, 'emb') and trk.emb is not None:
                        emb = trk.emb.cpu().numpy() if hasattr(trk.emb, 'cpu') else trk.emb
                        track_emb = np.asarray(emb).flatten()
                        track_emb = track_emb / (np.linalg.norm(track_emb) + 1e-8)
                    break
            
            # Match to gallery if not already matched
            if track_id not in track_to_person and track_emb is not None:
                # Compute distances to all gallery persons
                distances = []
                for gallery_emb in gallery_embeddings:
                    similarity = np.dot(track_emb, gallery_emb)
                    distance = 1.0 - similarity
                    distances.append(distance)
                
                # Find best match
                min_dist = min(distances)
                if min_dist < match_threshold:
                    best_person_id = gallery_ids[distances.index(min_dist)]
                    track_to_person[track_id] = best_person_id
                    stats['matches'] += 1
                    print(f"   ✅ Track {track_id} → Person {best_person_id} (distance: {min_dist:.3f})")
                else:
                    # No match - assign new ID
                    new_person_id = max(gallery_ids) + len([t for t in track_to_person.values() if t > max(gallery_ids)]) + 1
                    track_to_person[track_id] = new_person_id
                    stats['new_tracks'] += 1
                    print(f"   🆕 Track {track_id} → New Person {new_person_id} (min distance: {min_dist:.3f})")
            
            # Get persistent ID
            person_id = track_to_person.get(track_id, track_id)
            
            # Draw on frame
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Color based on person ID (ensure integers)
            color = (int((person_id * 50) % 255), int((person_id * 100) % 255), int((person_id * 150) % 255))
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID
            label = f"ID: {person_id}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Progress
        if frame_idx % 100 == 0:
            print(f"   Frame {frame_idx}/{total_frames} - {len(tracks)} tracks")
    
    cap.release()
    out.release()
    
    # Summary
    print(f"\n✅ Tracking complete!")
    print(f"\n📊 Statistics:")
    print(f"   Frames processed: {stats['frames_processed']}")
    print(f"   Total detections: {stats['detections']}")
    print(f"   Gallery matches: {stats['matches']}")
    print(f"   New persons: {stats['new_tracks']}")
    print(f"   Total unique persons: {len(set(track_to_person.values()))}")
    print(f"\n💾 Output saved to: {output_video}")
    print("=" * 80)
    
    return track_to_person

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Track video using gallery for persistent IDs')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--gallery', type=str, default='gallery.pkl', help='Gallery file from step2')
    parser.add_argument('--output', type=str, default='output_persistent.mp4', help='Output video file')
    parser.add_argument('--reid-model', type=str, default='osnet_x1_0_msmt17.pt', help='ReID model')
    parser.add_argument('--threshold', type=float, default=0.45, help='Match threshold (lower = stricter)')
    
    args = parser.parse_args()
    
    gallery_tracking(args.video, args.gallery, args.output, args.reid_model, args.threshold)
