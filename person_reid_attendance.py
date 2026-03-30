"""
Person Re-Identification for Attendance/Counting
Maintains a persistent database of all persons and reassigns IDs when they return
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from ultralytics import YOLO
from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend
from boxmot.utils import WEIGHTS
from scipy.spatial.distance import cosine
import argparse
from collections import defaultdict

class PersonDatabase:
    """
    Persistent database of all persons ever seen
    """
    def __init__(self, reid_threshold=0.6, min_features=1):
        self.persons = {}  # {person_id: [list of feature vectors]}
        self.next_id = 1
        self.reid_threshold = reid_threshold
        self.min_features = min_features  # Minimum features before matching
        print(f"[Database] Initialized with threshold={reid_threshold}")
        
    def add_person(self, features):
        """Add a new person to database"""
        person_id = self.next_id
        # Normalize features
        features_norm = features / (np.linalg.norm(features) + 1e-8)
        self.persons[person_id] = [features_norm]
        self.next_id += 1
        print(f"[Database] Added new person ID {person_id}")
        return person_id
    
    def update_person(self, person_id, features):
        """Add more features for an existing person"""
        if person_id in self.persons:
            # Normalize features
            features_norm = features / (np.linalg.norm(features) + 1e-8)
            self.persons[person_id].append(features_norm)
            # Keep only last 50 features per person
            if len(self.persons[person_id]) > 50:
                self.persons[person_id] = self.persons[person_id][-50:]
    
    def find_match(self, query_features):
        """
        Find best matching person in database
        Returns: (person_id, distance) or (None, None)
        """
        if len(self.persons) == 0:
            return None, None
        
        best_id = None
        best_distance = float('inf')
        
        # Normalize query
        query_norm = query_features / (np.linalg.norm(query_features) + 1e-8)
        
        for person_id, feature_list in self.persons.items():
            # Compare with all stored features for this person
            distances = []
            for stored_features in feature_list:
                # Cosine distance (already normalized)
                dist = 1.0 - np.dot(query_norm, stored_features)
                distances.append(dist)
            
            # Use minimum distance (best match)
            min_dist = min(distances)
            
            if min_dist < best_distance:
                best_distance = min_dist
                best_id = person_id
        
        print(f"[Database] Best match: ID {best_id}, distance: {best_distance:.4f}, threshold: {self.reid_threshold}")
        
        # Only return if below threshold
        if best_distance < self.reid_threshold:
            return best_id, best_distance
        
        return None, None
    
    def get_stats(self):
        return {
            'total_persons': len(self.persons),
            'total_features': sum(len(f) for f in self.persons.values())
        }


class SimpleTracker:
    """
    Simple short-term tracker to maintain IDs within a few frames
    """
    def __init__(self, max_age=30):
        self.tracks = {}  # {track_id: {'bbox': [x1,y1,x2,y2], 'age': int, 'person_id': int}}
        self.next_track_id = 1
        self.max_age = max_age
    
    def update(self, detections, person_ids):
        """
        Update tracks with new detections
        detections: list of [x1, y1, x2, y2]
        person_ids: list of person IDs from database
        """
        # Age existing tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]
        
        # Match detections to tracks using IoU
        matched_tracks = set()
        matched_dets = set()
        
        for det_idx, (det_bbox, person_id) in enumerate(zip(detections, person_ids)):
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                iou = self._compute_iou(det_bbox, track['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id]['bbox'] = det_bbox
                self.tracks[best_track_id]['age'] = 0
                self.tracks[best_track_id]['person_id'] = person_id
                matched_tracks.add(best_track_id)
                matched_dets.add(det_idx)
            else:
                # Create new track
                self.tracks[self.next_track_id] = {
                    'bbox': det_bbox,
                    'age': 0,
                    'person_id': person_id
                }
                self.next_track_id += 1
        
        return self.tracks
    
    def _compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Video file')
    parser.add_argument('--output', type=str, default='output_reid_attendance.mp4')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--reid-threshold', type=float, default=0.7, 
                       help='ReID matching threshold (lower=stricter, 0.6-0.8 recommended)')
    parser.add_argument('--conf', type=float, default=0.5, help='Detection confidence')
    args = parser.parse_args()
    
    print("="*80)
    print("👥 Person Re-Identification for Attendance/Counting")
    print("="*80)
    print("This system:")
    print("  ✓ Maintains a database of all unique persons")
    print("  ✓ Reassigns same ID when person returns (even after minutes)")
    print("  ✓ Works like attendance/counting systems")
    print("="*80)
    print(f"ReID Threshold: {args.reid_threshold} (lower = stricter matching)")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    yolo = YOLO('yolov8n.pt')
    
    # Load ReID model
    reid_model = ReIDDetectMultiBackend(
        weights=WEIGHTS / 'osnet_x1_0_msmt17.pt',  # Better ReID model
        device='cpu',
        fp16=False
    )
    print("✅ Using OSNet x1.0 (better accuracy for re-identification)")
    
    # Initialize database and tracker
    database = PersonDatabase(reid_threshold=args.reid_threshold, min_features=1)
    tracker = SimpleTracker(max_age=30)
    
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
    person_appearances = defaultdict(list)  # Track when each person appears
    
    print("\n🎬 Processing...\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Detect people
            results = yolo(frame, conf=args.conf, classes=[0], verbose=False)[0]  # class 0 = person
            
            if len(results.boxes) == 0:
                # No detections, just age tracks
                tracker.update([], [])
                out.write(frame)
                if args.show:
                    cv2.imshow('ReID Attendance', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            
            # Extract bboxes and features
            bboxes = []
            features_list = []
            
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                bboxes.append([x1, y1, x2, y2])
                
                # Extract ReID features
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size > 0:
                    features = reid_model([person_crop]).detach().cpu().numpy()[0]
                    features_list.append(features)
                else:
                    features_list.append(None)
            
            # Match each detection to database
            person_ids = []
            for bbox, features in zip(bboxes, features_list):
                if features is None:
                    person_ids.append(None)
                    continue
                
                # Try to find match in database
                matched_id, distance = database.find_match(features)
                
                if matched_id is not None:
                    # Found existing person
                    person_ids.append(matched_id)
                    database.update_person(matched_id, features)
                    if frame_idx - person_appearances[matched_id][-1] > 30 if person_appearances[matched_id] else True:
                        print(f"[Frame {frame_idx}] 🔄 Person {matched_id} returned! (distance: {distance:.3f})")
                else:
                    # New person
                    new_id = database.add_person(features)
                    person_ids.append(new_id)
                    print(f"[Frame {frame_idx}] 🆕 New person detected: ID {new_id}")
                
                person_appearances[person_ids[-1]].append(frame_idx)
            
            # Update short-term tracker
            tracks = tracker.update(bboxes, person_ids)
            
            # Draw results
            for track_id, track in tracks.items():
                person_id = track['person_id']
                if person_id is None:
                    continue
                
                bbox = track['bbox']
                
                # Assign color
                if person_id not in colors:
                    colors[person_id] = tuple(np.random.randint(0, 255, 3).tolist())
                
                color = colors[person_id]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw ID
                label = f"Person {person_id}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x1, y1-h-15), (x1+w+10, y1), color, -1)
                cv2.putText(frame, label, (x1+5, y1-8), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            # Info overlay
            stats = database.get_stats()
            info = f"Frame:{frame_idx}/{total_frames} | Active:{len(tracks)} | Total Persons:{stats['total_persons']}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            out.write(frame)
            if args.show:
                cv2.imshow('ReID Attendance', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if frame_idx % 30 == 0:
                print(f"Progress: {frame_idx}/{total_frames} | Total unique persons: {stats['total_persons']}")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    print("\n" + "="*80)
    print("✅ Processing Complete!")
    print("="*80)
    stats = database.get_stats()
    print(f"Total unique persons identified: {stats['total_persons']}")
    print(f"Total features stored: {stats['total_features']}")
    print(f"\nPerson appearance summary:")
    for person_id in sorted(person_appearances.keys()):
        frames = person_appearances[person_id]
        print(f"  Person {person_id}: Appeared {len(frames)} times (frames: {frames[0]}-{frames[-1]})")
    print(f"\nOutput saved to: {args.output}")
    print("="*80)

if __name__ == '__main__':
    main()
