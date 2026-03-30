"""
Attendance Tracker - Persistent Person Re-Identification
=========================================================
Fixed camera, people come and go, same person = same ID always.

Architecture:
  Layer 1: BoTSORT handles frame-to-frame tracking (brief occlusions, smooth motion)
  Layer 2: Persistent Gallery matches returning people to their original ID

Usage:
  python track_attendance.py --source test_3.mp4 --show
  python track_attendance.py --source test_3.mp4 --reid-threshold 0.5
"""
import sys
import os
from pathlib import Path

# Fix Windows Unicode encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import torch
import argparse
from collections import defaultdict
from ultralytics import YOLO
from boxmot import BoTSORT
from boxmot.utils import WEIGHTS


class PersistentGallery:
    """
    Stores appearance features for ALL persons ever seen.
    When someone returns, matches them to their original ID.
    """

    def __init__(self, threshold=0.45, max_features=50):
        self.threshold = threshold
        self.max_features = max_features
        self.entries = {}       # pid -> list of L2-normalized feature vectors
        self.next_id = 1

    def add_new(self, features):
        pid = self.next_id
        self.next_id += 1
        self.entries[pid] = [features.copy()]
        return pid

    def update(self, pid, features):
        if pid not in self.entries:
            return
        self.entries[pid].append(features.copy())
        if len(self.entries[pid]) > self.max_features:
            self.entries[pid] = self.entries[pid][-self.max_features:]

    def query(self, features, exclude_pids=None):
        """Find best matching person. Returns (pid, distance) or None."""
        if not self.entries:
            return None

        best_pid = None
        best_dist = float('inf')

        for pid, feat_list in self.entries.items():
            if exclude_pids and pid in exclude_pids:
                continue

            gallery = np.array(feat_list)
            similarities = gallery @ features
            dist = 1.0 - float(similarities.max())

            if dist < best_dist:
                best_dist = dist
                best_pid = pid

        if best_pid is not None and best_dist < self.threshold:
            return (best_pid, best_dist)
        return None


class AttendanceTracker:
    """
    Two-layer tracking:
    - BoTSORT: frame-to-frame (handles brief occlusions via Kalman + ReID)
    - Gallery: long-term re-ID (handles people leaving and returning)
    """

    def __init__(self, yolo_model='yolov8n.pt', reid_model='osnet_x1_0_msmt17.pt',
                 reid_threshold=0.45, detection_conf=0.25, track_buffer_sec=5):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.yolo = YOLO(yolo_model)
        self.detection_conf = detection_conf
        self.reid_model_name = reid_model
        self.reid_threshold = reid_threshold
        self.track_buffer_sec = track_buffer_sec

        self.tracker = None
        self.reid_model = None
        self.gallery = None
        self.id_map = {}                    # tracker_id -> persistent_id
        self.colors = {}
        self.appearance_log = defaultdict(list)
        self.reid_events = []
        self.frame_count = 0

    def _init_tracker(self, fps):
        track_buffer = int(self.track_buffer_sec * fps)
        reid_weights = WEIGHTS / self.reid_model_name

        self.tracker = BoTSORT(
            model_weights=reid_weights,
            device=self.device,
            fp16=False,
            track_high_thresh=0.3,
            new_track_thresh=0.4,
            track_buffer=track_buffer,
            match_thresh=0.8,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            cmc_method='sparseOptFlow',
            frame_rate=fps,
            lambda_=0.985,
        )

        self.reid_model = self.tracker.model
        self.gallery = PersistentGallery(threshold=self.reid_threshold)

        print(f"  Device: {self.device}")
        print(f"  Track buffer: {track_buffer} frames ({self.track_buffer_sec}s)")
        print(f"  ReID threshold: {self.reid_threshold}")

    @torch.no_grad()
    def _extract_features(self, img, bbox):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if (x2 - x1) < 20 or (y2 - y1) < 40:
            return None

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        features = self.reid_model([crop])
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        features = features.flatten().astype(np.float32)
        norm = np.linalg.norm(features)
        if norm < 1e-6:
            return None
        return features / norm

    def _map_ids(self, tracks, img):
        if len(tracks) == 0:
            return np.empty((0, 7))

        # Collect PIDs already active this frame
        active_pids = set()
        for t in tracks:
            tid = int(t[4])
            if tid in self.id_map:
                active_pids.add(self.id_map[tid])

        output = []
        for t in tracks:
            bbox = t[:4]
            tid = int(t[4])
            conf = float(t[5])
            cls = t[6] if len(t) > 6 else 0

            if tid in self.id_map:
                pid = self.id_map[tid]
                # Update gallery features every 10 frames
                if self.frame_count % 10 == 0 and conf > 0.4:
                    feat = self._extract_features(img, bbox)
                    if feat is not None:
                        self.gallery.update(pid, feat)
            else:
                # New tracker ID -> check gallery for returning person
                feat = self._extract_features(img, bbox)
                if feat is not None:
                    result = self.gallery.query(feat, exclude_pids=active_pids)
                    if result is not None:
                        pid, dist = result
                        self.id_map[tid] = pid
                        self.gallery.update(pid, feat)
                        self.reid_events.append((self.frame_count, pid, dist))
                        active_pids.add(pid)
                        print(f"  [RETURNED] Frame {self.frame_count}: Person {pid} RETURNED (dist: {dist:.3f})")
                    else:
                        pid = self.gallery.add_new(feat)
                        self.id_map[tid] = pid
                        active_pids.add(pid)
                        print(f"  [NEW] Frame {self.frame_count}: NEW Person {pid}")
                else:
                    pid = self.gallery.next_id
                    self.gallery.next_id += 1
                    self.gallery.entries[pid] = []
                    self.id_map[tid] = pid
                    print(f"  [NEW] Frame {self.frame_count}: NEW Person {pid} (no features)")

            self.appearance_log[pid].append(self.frame_count)

            if pid not in self.colors:
                np.random.seed(pid * 37)
                self.colors[pid] = tuple(np.random.randint(60, 230, 3).tolist())

            output.append([bbox[0], bbox[1], bbox[2], bbox[3], pid, conf, cls])

        return np.array(output)

    def _draw(self, frame, tracks, total_frames):
        h, w = frame.shape[:2]

        for t in tracks:
            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
            pid = int(t[4])
            color = self.colors.get(pid, (200, 200, 200))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"Person {pid}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 12), (x1 + lw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show "RETURNED!" for recently re-identified people
            for ef, ep, ed in self.reid_events[-10:]:
                if ep == pid and (self.frame_count - ef) < 45:
                    cv2.putText(frame, "RETURNED!", (x1, y2 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    break

        n_unique = len(self.gallery.entries)
        n_active = len(tracks)
        info = f"Frame: {self.frame_count}/{total_frames} | Active: {n_active} | Unique: {n_unique}"
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    def run(self, source, output='output_attendance.mp4', show=False):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {source}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nVideo: {w}x{h} @ {fps}fps, {total} frames")
        self._init_tracker(fps)

        out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        print(f"\nProcessing...\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.frame_count += 1

                # Detect persons only
                results = self.yolo(frame, conf=self.detection_conf, classes=[0], verbose=False)[0]

                if len(results.boxes) > 0:
                    dets = []
                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        c = float(box.conf[0].cpu().numpy())
                        cl = float(box.cls[0].cpu().numpy())
                        dets.append([x1, y1, x2, y2, c, cl])
                    dets = np.array(dets)
                else:
                    dets = np.empty((0, 6))

                raw_tracks = self.tracker.update(dets, frame)
                tracks = self._map_ids(raw_tracks, frame)
                self._draw(frame, tracks, total)

                out.write(frame)
                if show:
                    cv2.imshow('Attendance Tracker', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if self.frame_count % (fps * 2) == 0:
                    pct = (self.frame_count / total) * 100
                    print(f"  {pct:.0f}% | Active: {len(tracks)} | Unique: {len(self.gallery.entries)}")

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        print(f"\n[DONE] Saved: {output}")
        self._summary(total, fps)

    def _summary(self, total_frames, fps):
        print(f"\n{'='*60}")
        print(f"ATTENDANCE REPORT")
        print(f"{'='*60}")
        print(f"Total unique persons: {len(self.gallery.entries)}")
        print(f"Re-identification events: {len(self.reid_events)}")
        print()

        for pid in sorted(self.appearance_log.keys()):
            frames = self.appearance_log[pid]
            first, last = frames[0], frames[-1]
            dur = (last - first + 1) / fps
            feats = len(self.gallery.entries.get(pid, []))
            reids = sum(1 for _, p, _ in self.reid_events if p == pid)
            print(f"  Person {pid}: Frames {first}-{last} ({dur:.1f}s) | "
                  f"Features: {feats} | Re-IDs: {reids}")

        print(f"{'='*60}")

        if self.reid_events:
            print(f"\nRe-ID Events:")
            for f, p, d in self.reid_events:
                print(f"  Frame {f}: Person {p} returned (distance: {d:.3f})")


def main():
    parser = argparse.ArgumentParser(description='Attendance Tracker')
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_attendance.mp4')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt')
    parser.add_argument('--reid-model', type=str, default='osnet_x1_0_msmt17.pt')
    parser.add_argument('--reid-threshold', type=float, default=0.45,
                        help='Gallery matching threshold (lower=stricter, 0.3-0.5)')
    parser.add_argument('--detection-conf', type=float, default=0.25)
    parser.add_argument('--track-buffer', type=float, default=5.0,
                        help='Seconds to keep lost tracks before deleting')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("ATTENDANCE TRACKER")
    print("=" * 60)
    print("Layer 1: BoTSORT (frame-to-frame tracking)")
    print("Layer 2: Persistent Gallery (re-ID when people return)")
    print("=" * 60)

    tracker = AttendanceTracker(
        yolo_model=args.yolo_model,
        reid_model=args.reid_model,
        reid_threshold=args.reid_threshold,
        detection_conf=args.detection_conf,
        track_buffer_sec=args.track_buffer,
    )
    tracker.run(source=args.source, output=args.output, show=args.show)


if __name__ == '__main__':
    main()
