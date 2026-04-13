"""
Persistent Person Identity Tracking using Embeddings
Maintains a global database of person embeddings for consistent ID assignment
"""
import sys
import argparse
from pathlib import Path
import json
import pickle

# Add boxmot to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend
from boxmot.utils import WEIGHTS
import cv2
import numpy as np
from ultralytics import YOLO


class EmbeddingPersistentTracker:
    """
    Tracks persons using persistent embeddings stored in memory.
    Assigns consistent IDs based on embedding similarity and track continuity.
    """

    def __init__(
        self,
        reid_model_path,
        device='cpu',
        similarity_threshold=0.8,
        min_match_threshold=0.75,
        max_embeddings_per_person=10,
        track_buffer_seconds=4.0,
    ):
        self.reid_model = ReIDDetectMultiBackend(
            weights=reid_model_path,
            device=device,
            fp16=False,
        )

        # person_id -> list of embeddings
        self.embedding_db = {}

        self.next_person_id = 1
        self.similarity_threshold = similarity_threshold
        self.min_match_threshold = min_match_threshold

        self.max_embeddings_per_person = max_embeddings_per_person

        # Short-term track buffer for anti-ID switching
        self.track_buffer_seconds = track_buffer_seconds
        self.fps = 30
        self.max_buffer_frames = int(self.track_buffer_seconds * self.fps)
        self.track_buffer = {}  # person_id -> {bbox, last_seen}

        self.frame_idx = 0

        print(
            f"Initialized EmbeddingPersistentTracker with threshold {similarity_threshold} "
            f"and min threshold {min_match_threshold}"
        )

    def set_fps(self, fps):
        self.fps = fps
        self.max_buffer_frames = int(self.track_buffer_seconds * self.fps)

    def _compute_iou(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0

        area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
        area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
        union_area = area1 + area2 - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _prune_track_buffer(self):
        to_delete = []
        for person_id, data in self.track_buffer.items():
            if self.frame_idx - data['last_seen'] > self.max_buffer_frames:
                to_delete.append(person_id)
        for person_id in to_delete:
            del self.track_buffer[person_id]

    def _normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def extract_embedding(self, image, bbox):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        person_crop = image[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None

        try:
            embedding = self.reid_model([person_crop])
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            elif isinstance(embedding, list):
                embedding = np.array(embedding[0])

            if embedding.ndim > 1:
                embedding = embedding.flatten()

            embedding = self._normalize_embedding(embedding)
            return embedding

        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

    def cosine_similarity(self, emb1, emb2):
        return float(np.dot(emb1, emb2))

    def match_person(self, embedding):
        if not self.embedding_db:
            return None, 0.0

        best_person_id = None
        best_similarity = 0.0

        for person_id, embedding_list in self.embedding_db.items():
            stored = np.stack(embedding_list, axis=0)
            sims = stored.dot(embedding)
            max_sim = float(np.max(sims))
            if max_sim > best_similarity:
                best_similarity = max_sim
                best_person_id = person_id

        return best_person_id, best_similarity

    def append_embedding(self, person_id, embedding):
        if person_id not in self.embedding_db:
            self.embedding_db[person_id] = []
        self.embedding_db[person_id].append(embedding)
        if len(self.embedding_db[person_id]) > self.max_embeddings_per_person:
            self.embedding_db[person_id] = self.embedding_db[person_id][-self.max_embeddings_per_person:]

    def process_detections(self, frame, detections):
        self.frame_idx += 1
        self._prune_track_buffer()

        results = []
        used_ids = set()

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            bbox = (x1, y1, x2, y2)
            assigned_person_id = None

            best_iou = 0.0
            best_buffer_id = None
            for person_id, data in self.track_buffer.items():
                if person_id in used_ids:
                    continue
                iou = self._compute_iou(bbox, data['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_buffer_id = person_id

            if best_iou >= 0.55:
                assigned_person_id = best_buffer_id
                print(f"Temporal hit: Keep ID {assigned_person_id} by IoU (iou={best_iou:.3f})")

            if assigned_person_id is None:
                embedding = self.extract_embedding(frame, bbox)
                if embedding is None:
                    continue

                best_person_id, best_similarity = self.match_person(embedding)

                if best_person_id is not None and best_similarity >= self.similarity_threshold:
                    assigned_person_id = best_person_id
                    print(f"Reuse ID {assigned_person_id} by embedding (sim={best_similarity:.3f})")
                elif best_person_id is not None and best_similarity >= self.min_match_threshold:
                    print(
                        f"Borderline similarity {best_similarity:.3f} for ID {best_person_id}; creating new ID"
                    )
                    assigned_person_id = self.next_person_id
                    self.next_person_id += 1
                else:
                    assigned_person_id = self.next_person_id
                    self.next_person_id += 1
                    print(f"Created new person ID {assigned_person_id} (best_sim={best_similarity:.3f})")

                self.append_embedding(assigned_person_id, embedding)
            else:
                embedding = self.extract_embedding(frame, bbox)
                if embedding is not None:
                    self.append_embedding(assigned_person_id, embedding)

            self.track_buffer[assigned_person_id] = {
                'bbox': bbox,
                'last_seen': self.frame_idx,
            }
            used_ids.add(assigned_person_id)
            results.append((x1, y1, x2, y2, assigned_person_id, conf))

        return results

    def save_embeddings(self, filepath):
        """
        Save embeddings to disk (JSON format)
        """
        serializable_db = {}
        for person_id, embeddings in self.embedding_db.items():
            serializable_db[str(person_id)] = [emb.tolist() for emb in embeddings]

        data = {
            'embedding_db': serializable_db,
            'next_person_id': self.next_person_id,
            'similarity_threshold': self.similarity_threshold,
            'min_match_threshold': self.min_match_threshold,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(self.embedding_db)} person embeddings to {filepath}")

    def load_embeddings(self, filepath):
        """
        Load embeddings from disk
        """
        if not Path(filepath).exists():
            print(f"Embedding file {filepath} not found, starting fresh")
            return

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.embedding_db = {}
            for person_id_str, embedding_list in data['embedding_db'].items():
                person_id = int(person_id_str)
                self.embedding_db[person_id] = [np.array(x) for x in embedding_list]

            self.next_person_id = data.get('next_person_id', 1)
            self.similarity_threshold = data.get('similarity_threshold', 0.8)
            self.min_match_threshold = data.get('min_match_threshold', 0.75)

            print(f"Loaded {len(self.embedding_db)} person embeddings from {filepath}")

        except Exception as e:
            print(f"Error loading embeddings: {e}, starting fresh")


def main():
    parser = argparse.ArgumentParser(description='Track with Persistent Embeddings')
    parser.add_argument('--source', type=str, required=True, help='Video file path')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt', help='YOLO model')
    parser.add_argument('--reid-model', type=str, default='osnet_x1_0_msmt17.pt', help='ReID model')
    parser.add_argument('--similarity-threshold', type=float, default=0.8, help='Strict threshold for matching (0.75-0.85 recommended)')
    parser.add_argument('--min-match-threshold', type=float, default=0.75, help='Minimum threshold to consider near match as borderline')
    parser.add_argument('--max-embeddings-per-person', type=int, default=10, help='Max stored embeddings per person')
    parser.add_argument('--track-buffer-seconds', type=float, default=4.0, help='Duration to keep track buffer for lost IDs (seconds)')
    parser.add_argument('--embeddings', type=str, default='embeddings.json', help='Path to save/load embeddings')
    parser.add_argument('--conf', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--output', type=str, default='output_embedding_persistent.mp4', help='Output video path')
    parser.add_argument('--show', action='store_true', help='Display video while processing')
    args = parser.parse_args()

    print("=" * 80)
    print("Persistent Embedding-Based Person Identity Tracking")
    print("=" * 80)
    print(f"Video: {args.source}")
    print(f"YOLO Model: {args.yolo_model}")
    print(f"ReID Model: {args.reid_model}")
    print(f"Similarity Threshold: {args.similarity_threshold}")
    print(f"Embeddings file: {args.embeddings}")
    print("="*80)

    # Load YOLO model
    print("Loading YOLO model...")
    yolo = YOLO(args.yolo_model)

    # Initialize embedding tracker
    print("Initializing Embedding Persistent Tracker...")
    device = 'cuda:0' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    reid_weights = WEIGHTS / args.reid_model
    tracker = EmbeddingPersistentTracker(
        reid_model_path=reid_weights,
        device=device,
        similarity_threshold=args.similarity_threshold,
        min_match_threshold=args.min_match_threshold,
        max_embeddings_per_person=args.max_embeddings_per_person,
        track_buffer_seconds=args.track_buffer_seconds,
    )

    # Load existing embeddings
    tracker.load_embeddings(args.embeddings)

    # Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.source}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    tracker.set_fps(fps)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    colors = {}  # Person ID -> color mapping

    print("\nProcessing video...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Run YOLO detection
            results = yolo(frame, conf=args.conf, verbose=False)[0]

            # Get detections (only persons, class 0)
            detections = []
            if len(results.boxes) > 0:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    if int(cls) == 0:  # Person class
                        detections.append([x1, y1, x2, y2, conf, cls])

            # Process detections with embedding matching
            tracked_persons = tracker.process_detections(frame, detections)

            # Draw results
            for x1, y1, x2, y2, person_id, conf in tracked_persons:
                # Assign color to person ID
                if person_id not in colors:
                    colors[person_id] = tuple(np.random.randint(0, 255, 3).tolist())

                color = colors[person_id]

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Draw ID label
                label = f"ID: {person_id}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (int(x1), int(y1) - label_height - 10),
                            (int(x1) + label_width, int(y1)), color, -1)
                cv2.putText(frame, label, (int(x1), int(y1) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Add frame info
            info_text = f"Frame: {frame_idx}/{total_frames} | Persons: {len(tracked_persons)} | Total IDs: {len(colors)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write frame
            out.write(frame)

            # Show frame
            if args.show:
                cv2.imshow('Embedding Persistent Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Progress
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames}) | Active persons: {len(tracked_persons)} | Total IDs seen: {len(colors)}")

    finally:
        # Save embeddings
        tracker.save_embeddings(args.embeddings)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"\nDone! Output saved to: {args.output}")
    print(f"Total unique person IDs tracked: {len(colors)}")
    print(f"Embeddings saved to: {args.embeddings}")


if __name__ == '__main__':
    import torch
    main()