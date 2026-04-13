"""
Robust Person Detection, Tracking & Counting System
=====================================================
Fixes:
  - ID switching across frames (via ReID embedding gallery + global ID mapping)
  - ID reuse for different people (strict cosine similarity threshold)
  - Incorrect IN/OUT counting (direction-based line-crossing with dedup sets)
  - Wrong total unique count (global identity dictionary)
  - Occlusion / re-entry failures (long track buffer + embedding re-match)

Architecture:
  Detector  -> raw person bounding boxes from YOLOv8
  Tracker   -> frame-to-frame association via DeepOCSORT
  IdentityManager -> global ID via OSNet embedding gallery
  PeopleCounter   -> direction-aware line-crossing logic
  Visualizer      -> overlay drawing & debug logging
"""

import sys
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

# Ensure boxmot is importable
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from boxmot import DeepOCSORT
from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend
from boxmot.utils import WEIGHTS


# ---------------------------------------------------------------------------
#  Module 1 – Detection
# ---------------------------------------------------------------------------
class Detector:
    """Wraps YOLOv8 and returns person-only detections."""

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame: np.ndarray):
        """
        Returns ndarray (N, 6) with columns [x1, y1, x2, y2, conf, cls].
        Only class 0 (person) is retained.
        """
        results = self.model(frame, conf=self.conf, classes=[0], verbose=False)[0]
        if len(results.boxes) == 0:
            return np.empty((0, 6))
        dets = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = float(box.cls[0].cpu().numpy())
            dets.append([x1, y1, x2, y2, conf, cls])
        return np.array(dets)


# ---------------------------------------------------------------------------
#  Module 2 – Short-term Tracker (DeepOCSORT)
# ---------------------------------------------------------------------------
class Tracker:
    """Thin wrapper around DeepOCSORT with tuned parameters."""

    def __init__(
        self,
        reid_weights: str = "osnet_x1_0_msmt17.pt",
        device: str = "cpu",
        max_age: int = 100,
        min_hits: int = 1,
        iou_threshold: float = 0.15,
    ):
        self.tracker = DeepOCSORT(
            model_weights=Path(reid_weights),
            device=device,
            fp16=False,
            per_class=False,
            det_thresh=0.0,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            delta_t=3,
            asso_func="giou",
            inertia=0.2,
        )

    def update(self, dets: np.ndarray, frame: np.ndarray):
        """
        Returns ndarray (M, 7+) – each row: [x1,y1,x2,y2, track_id, conf, cls, ...]
        """
        return self.tracker.update(dets, frame)


# ---------------------------------------------------------------------------
#  Module 3 – Global Identity Manager (ReID gallery)
# ---------------------------------------------------------------------------
class IdentityManager:
    """
    Maintains:
      • A gallery of normalised OSNet embeddings per global person ID.
      • A mapping from short-term tracker_id → stable global_id.

    When the tracker emits a *new* tracker_id the manager:
      1. Extracts an embedding for the crop.
      2. Searches the gallery for a match above `similarity_threshold`.
      3. If found → reuses the existing global_id.
      4. Otherwise → mints a new global_id.
    """

    def __init__(
        self,
        reid_model_path: str,
        device: str = "cpu",
        similarity_threshold: float = 0.70,
        max_embeddings_per_id: int = 15,
        embedding_update_interval: int = 5,
    ):
        self.reid_model = ReIDDetectMultiBackend(
            weights=Path(reid_model_path), device=torch.device(device), fp16=False
        )
        self.similarity_threshold = similarity_threshold
        self.max_embeddings_per_id = max_embeddings_per_id
        self.embedding_update_interval = embedding_update_interval

        # global_id → list[np.ndarray]   (normalised embeddings)
        self.gallery: dict[int, list[np.ndarray]] = {}
        # tracker_id → global_id
        self.tid_to_gid: dict[int, int] = {}
        # global_id → last frame seen (for stale cleanup)
        self.gid_last_seen: dict[int, int] = {}

        self._next_global_id = 1
        self._frame_count = 0

    # ---- embedding helpers ------------------------------------------------
    def _normalise(self, emb: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb

    def _extract_embedding(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            return None
        try:
            emb = self.reid_model([crop])
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            if emb.ndim > 1:
                emb = emb.flatten()
            return self._normalise(emb)
        except Exception as e:
            print(f"  [IdentityManager] embedding extraction error: {e}")
            return None

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _best_gallery_match(self, emb: np.ndarray):
        """Return (global_id, similarity) or (None, 0.0)."""
        best_gid, best_sim = None, 0.0
        for gid, emb_list in self.gallery.items():
            mat = np.stack(emb_list)             # (K, D)
            sims = mat @ emb                      # (K,)
            s = float(np.max(sims))
            if s > best_sim:
                best_sim = s
                best_gid = gid
        return best_gid, best_sim

    def _add_embedding(self, gid: int, emb: np.ndarray):
        self.gallery.setdefault(gid, []).append(emb)
        if len(self.gallery[gid]) > self.max_embeddings_per_id:
            self.gallery[gid] = self.gallery[gid][-self.max_embeddings_per_id:]

    # ---- public API -------------------------------------------------------
    def resolve(self, tracks: np.ndarray, frame: np.ndarray) -> list[dict]:
        """
        For every track row coming out of DeepOCSORT, produce a dict:
          { 'bbox': (x1,y1,x2,y2),
            'global_id': int,
            'tracker_id': int,
            'conf': float,
            'similarity': float | None,
            'is_new': bool }
        """
        self._frame_count += 1
        results = []

        for track in tracks:
            bbox = track[:4]
            tracker_id = int(track[4])
            conf = float(track[5]) if len(track) > 5 else 0.0
            similarity = None
            is_new = False

            # --- Case A: tracker_id already mapped ---
            if tracker_id in self.tid_to_gid:
                gid = self.tid_to_gid[tracker_id]
                # Periodically refresh the embedding
                if self._frame_count % self.embedding_update_interval == 0:
                    emb = self._extract_embedding(frame, bbox)
                    if emb is not None:
                        self._add_embedding(gid, emb)
                self.gid_last_seen[gid] = self._frame_count
                results.append(dict(
                    bbox=tuple(bbox), global_id=gid, tracker_id=tracker_id,
                    conf=conf, similarity=None, is_new=False))
                continue

            # --- Case B: new tracker_id → need embedding match ---
            emb = self._extract_embedding(frame, bbox)
            if emb is None:
                # Can't extract → mint a throwaway global id
                gid = self._next_global_id; self._next_global_id += 1
                self.tid_to_gid[tracker_id] = gid
                self.gid_last_seen[gid] = self._frame_count
                results.append(dict(
                    bbox=tuple(bbox), global_id=gid, tracker_id=tracker_id,
                    conf=conf, similarity=None, is_new=True))
                continue

            match_gid, sim = self._best_gallery_match(emb)
            similarity = sim

            if match_gid is not None and sim >= self.similarity_threshold:
                # Re-identified!
                gid = match_gid
                self._add_embedding(gid, emb)
                print(f"  [ReID] Tracker {tracker_id} → Re-identified as Global ID {gid}  (sim={sim:.3f})")
            else:
                # Genuinely new person
                gid = self._next_global_id; self._next_global_id += 1
                self._add_embedding(gid, emb)
                is_new = True
                print(f"  [ReID] Tracker {tracker_id} → NEW Global ID {gid}  (best_sim={sim:.3f})")

            self.tid_to_gid[tracker_id] = gid
            self.gid_last_seen[gid] = self._frame_count
            results.append(dict(
                bbox=tuple(bbox), global_id=gid, tracker_id=tracker_id,
                conf=conf, similarity=similarity, is_new=is_new))

        return results

    # ---- Persistence ------------------------------------------------------
    def save_gallery(self, path: str):
        data = {
            "gallery": {str(gid): [e.tolist() for e in el]
                        for gid, el in self.gallery.items()},
            "next_id": self._next_global_id,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"  [IdentityManager] Saved gallery ({len(self.gallery)} IDs) → {path}")

    def load_gallery(self, path: str):
        p = Path(path)
        if not p.exists():
            print(f"  [IdentityManager] No gallery at {path}, starting fresh.")
            return
        with open(p, "r") as f:
            data = json.load(f)
        self.gallery = {
            int(gid): [np.array(e, dtype=np.float32) for e in el]
            for gid, el in data["gallery"].items()
        }
        self._next_global_id = data.get("next_id", max(self.gallery.keys(), default=0) + 1)
        print(f"  [IdentityManager] Loaded gallery ({len(self.gallery)} IDs) from {path}")


# ---------------------------------------------------------------------------
#  Module 4 – People Counter (line-crossing with direction)
# ---------------------------------------------------------------------------
class PeopleCounter:
    """
    Direction-based line-crossing counter.
    • crossing downward (y increases past line) → ENTER
    • crossing upward  (y decreases past line) → EXIT
    Each global_id is counted at most once per direction.
    """

    def __init__(self, line_y: int):
        self.line_y = line_y
        # global_id → previous center-y
        self.prev_cy: dict[int, float] = {}
        self.counted_enter: set[int] = set()
        self.counted_exit: set[int] = set()

    def update(self, resolved: list[dict]) -> tuple[int, int]:
        """
        Takes the output of IdentityManager.resolve() and returns
        (total_enter, total_exit) after processing this frame.
        """
        for r in resolved:
            gid = r["global_id"]
            x1, y1, x2, y2 = r["bbox"]
            cy = (y1 + y2) / 2.0

            if gid in self.prev_cy:
                prev = self.prev_cy[gid]
                # Crossed downward → ENTER
                if prev < self.line_y and cy >= self.line_y and gid not in self.counted_enter:
                    self.counted_enter.add(gid)
                    print(f"  [Counter] Global ID {gid} → ENTERED ↓")
                # Crossed upward → EXIT
                elif prev > self.line_y and cy <= self.line_y and gid not in self.counted_exit:
                    self.counted_exit.add(gid)
                    print(f"  [Counter] Global ID {gid} → EXITED  ↑")

            self.prev_cy[gid] = cy

        return len(self.counted_enter), len(self.counted_exit)


# ---------------------------------------------------------------------------
#  Module 5 – Visualizer
# ---------------------------------------------------------------------------
class Visualizer:
    """Draws bounding boxes, IDs, centers, counting line, and the stats overlay."""

    # Pre-baked palette (distinguishable colours)
    PALETTE = [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
        (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
        (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
        (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
        (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    ]

    @staticmethod
    def _color(gid: int):
        c = Visualizer.PALETTE[gid % len(Visualizer.PALETTE)]
        return (int(c[2]), int(c[1]), int(c[0]))   # BGR

    @staticmethod
    def draw(frame, resolved, line_y, enter_count, exit_count, total_unique, frame_idx):
        h, w = frame.shape[:2]

        # --- Counting line ---
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
        cv2.putText(frame, "COUNTING LINE", (10, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # --- Per-person visualisation ---
        for r in resolved:
            gid = r["global_id"]
            tid = r["tracker_id"]
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            conf = r["conf"]
            color = Visualizer._color(gid)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Center point
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 4, color, -1)

            # Label: Global ID + Tracker ID
            label = f"G{gid} T{tid} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # --- Top overlay bar ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, f"ENTER: {enter_count}", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, f"EXIT: {exit_count}", (200, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frame, f"TOTAL UNIQUE: {total_unique}", (370, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_idx}  Active: {len(resolved)}", (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        return frame


# ---------------------------------------------------------------------------
#  Main orchestrator
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Robust Person Tracker & Counter")
    p.add_argument("--source", type=str, required=True, help="Input video path")
    p.add_argument("--output", type=str, default="robust_tracking_output.mp4")
    p.add_argument("--yolo-model", type=str, default="yolov8n.pt")
    p.add_argument("--reid-model", type=str, default="osnet_x1_0_msmt17.pt")
    p.add_argument("--conf", type=float, default=0.35, help="YOLO detection confidence")
    p.add_argument("--similarity", type=float, default=0.70,
                    help="Cosine similarity threshold for ReID matching")
    p.add_argument("--line-ratio", type=float, default=0.50,
                    help="Counting line position as fraction of frame height (0-1)")
    p.add_argument("--max-age", type=int, default=100,
                    help="DeepOCSORT track buffer (frames to keep lost tracks)")
    p.add_argument("--gallery", type=str, default="person_gallery.json",
                    help="Path to save/load the embedding gallery")
    p.add_argument("--show", action="store_true", help="Display live window")
    p.add_argument("--debug", action="store_true", help="Print per-frame debug info")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  ROBUST PERSON TRACKER & COUNTER")
    print("=" * 70)
    print(f"  Source        : {args.source}")
    print(f"  YOLO model    : {args.yolo_model}")
    print(f"  ReID model    : {args.reid_model}")
    print(f"  Similarity    : {args.similarity}")
    print(f"  Line ratio    : {args.line_ratio}")
    print(f"  Track buffer  : {args.max_age} frames")
    print(f"  Gallery file  : {args.gallery}")
    print("=" * 70)

    # ---- Initialise modules ----
    detector = Detector(model_path=args.yolo_model, conf=args.conf)

    reid_path = str(WEIGHTS / args.reid_model)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tracker = Tracker(
        reid_weights=reid_path,
        device=device,
        max_age=args.max_age,
        min_hits=1,
        iou_threshold=0.15,
    )
    identity_mgr = IdentityManager(
        reid_model_path=reid_path,
        device=device,
        similarity_threshold=args.similarity,
    )
    identity_mgr.load_gallery(args.gallery)

    # ---- Open video ----
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.source}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    line_y = int(height * args.line_ratio)

    counter = PeopleCounter(line_y=line_y)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"\n  Video: {width}x{height} @ {fps}fps  ({total_frames} frames)")
    print(f"  Counting line at y={line_y}")
    print(f"  Device: {device}\n")

    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # 1. Detect
            dets = detector.detect(frame)

            # 2. Track
            tracks = tracker.update(dets, frame)

            # 3. Resolve global IDs
            resolved = identity_mgr.resolve(tracks, frame)

            # 4. Count
            enter_count, exit_count = counter.update(resolved)
            total_unique = len(identity_mgr.gallery)

            # 5. Debug log (every frame or every 30 frames)
            if args.debug or frame_idx % 30 == 0:
                for r in resolved:
                    sim_str = f"{r['similarity']:.3f}" if r["similarity"] is not None else "—"
                    print(
                        f"  [F{frame_idx:>5}] T{r['tracker_id']:>3} → G{r['global_id']:>3}  "
                        f"bbox=({int(r['bbox'][0])},{int(r['bbox'][1])},{int(r['bbox'][2])},{int(r['bbox'][3])})  "
                        f"conf={r['conf']:.2f}  sim={sim_str}"
                    )

            # 6. Visualise
            frame = Visualizer.draw(
                frame, resolved, line_y, enter_count, exit_count, total_unique, frame_idx
            )

            out.write(frame)
            if args.show:
                cv2.imshow("Robust Person Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Progress
            if frame_idx % 60 == 0:
                elapsed = time.time() - t0
                fps_real = frame_idx / elapsed if elapsed > 0 else 0
                print(
                    f"  Progress: {frame_idx}/{total_frames} "
                    f"({100*frame_idx/total_frames:.1f}%)  "
                    f"{fps_real:.1f} fps  |  "
                    f"ENTER={enter_count}  EXIT={exit_count}  UNIQUE={total_unique}"
                )

    finally:
        identity_mgr.save_gallery(args.gallery)
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  PROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Frames processed : {frame_idx}")
    print(f"  Wall-clock time  : {elapsed:.1f}s  ({frame_idx/elapsed:.1f} fps)")
    print(f"  ENTER count      : {enter_count}")
    print(f"  EXIT count       : {exit_count}")
    print(f"  Total unique IDs : {total_unique}")
    print(f"  Output video     : {args.output}")
    print(f"  Gallery saved    : {args.gallery}")
    print("=" * 70)

    # Detailed ID summary
    print("\n  Global ID summary:")
    for gid in sorted(identity_mgr.gallery.keys()):
        n_emb = len(identity_mgr.gallery[gid])
        entered = "✓" if gid in counter.counted_enter else "—"
        exited  = "✓" if gid in counter.counted_exit else "—"
        print(f"    G{gid:>3}  embeddings={n_emb:>2}  entered={entered}  exited={exited}")


if __name__ == "__main__":
    main()
