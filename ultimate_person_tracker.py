"""
Ultimate Person Detection, Tracking & Re-Identification System
================================================================
Key design goals:
  1. Every unique person gets exactly ONE global ID – for their entire lifetime.
  2. When a person leaves the frame and returns (even minutes later), they get
     their SAME global ID back via embedding-based re-identification.
  3. Two different people must NEVER share the same global ID.
  4. The video overlay shows IN count, OUT count, and total unique persons.

Architecture:
  Detector          → YOLOv8 person detections
  Tracker           → DeepOCSORT short-term association
  IdentityManager   → OSNet embedding gallery with Hungarian assignment
  PeopleCounter     → Direction-aware line-crossing with dedup
  Visualizer        → Premium HUD overlay on every frame
"""

import sys
import argparse
import json
import time
import math
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

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
#  Module 1 – Detection
# ---------------------------------------------------------------------------
class Detector:
    """YOLOv8 person detector."""

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame: np.ndarray):
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
    """Thin wrapper around DeepOCSORT."""

    def __init__(
        self,
        reid_weights: str = "osnet_x1_0_msmt17.pt",
        device: str = "cpu",
        max_age: int = 120,
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
        return self.tracker.update(dets, frame)


# ---------------------------------------------------------------------------
#  Module 3 – Global Identity Manager (ReID gallery)
# ---------------------------------------------------------------------------
class IdentityManager:
    """
    Robust global identity assignment using OSNet embeddings.

    Design principles
    -----------------
    • Gallery stores N normalised embeddings per global ID (rolling window).
    • When a NEW tracker_id appears, extract an embedding, then:
        – If only 1 new tracker_id this frame: greedy best-match.
        – If ≥2 new tracker_ids this frame: use Hungarian assignment to
          ensure no two new tracks claim the same gallery person.
    • Similarity threshold controls whether a match is accepted.
    • Embeddings are refreshed for mapped tracker_ids on a periodic basis
      using exponential moving average (EMA) for stability.
    • When a person leaves and the tracker drops them, the gallery entry
      persists so they can be re-identified later.
    """

    def __init__(
        self,
        reid_model_path: str,
        device: str = "cpu",
        similarity_threshold: float = 0.65,
        max_embeddings_per_id: int = 20,
        embedding_update_interval: int = 5,
        ema_alpha: float = 0.3,
    ):
        self.reid_model = ReIDDetectMultiBackend(
            weights=Path(reid_model_path), device=torch.device(device), fp16=False
        )
        self.similarity_threshold = similarity_threshold
        self.max_embeddings_per_id = max_embeddings_per_id
        self.embedding_update_interval = embedding_update_interval
        self.ema_alpha = ema_alpha

        # global_id → list[np.ndarray]  (normalised embeddings)
        self.gallery: dict[int, list[np.ndarray]] = {}
        # tracker_id → global_id
        self.tid_to_gid: dict[int, int] = {}
        # global_id → last frame seen
        self.gid_last_seen: dict[int, int] = {}
        # global_id → running EMA embedding (for fast comparison)
        self.gid_ema: dict[int, np.ndarray] = {}
        # tracker_id → last known bbox center (for spatial reasonableness)
        self.tid_last_center: dict[int, tuple[float, float]] = {}
        # Active tracker_ids this frame (to detect disappeared tracks)
        self._active_tids: set[int] = set()
        self._prev_active_tids: set[int] = set()

        self._next_global_id = 1
        self._frame_count = 0

    # ---- embedding helpers ------------------------------------------------
    def _normalise(self, emb: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb

    def _extract_embedding(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
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

    def _gallery_similarity(self, emb: np.ndarray, gid: int) -> float:
        """Compute similarity between an embedding and a gallery entry.
        Uses the AVERAGE of top-3 similarities for robustness."""
        emb_list = self.gallery.get(gid, [])
        if not emb_list:
            return 0.0
        mat = np.stack(emb_list)  # (K, D)
        sims = mat @ emb          # (K,)
        # Average top-3 for more stable matching
        topk = min(3, len(sims))
        top_sims = np.sort(sims)[-topk:]
        return float(np.mean(top_sims))

    def _best_gallery_match(self, emb: np.ndarray, exclude_gids: set | None = None):
        """Return (global_id, similarity) or (None, 0.0)."""
        best_gid, best_sim = None, 0.0
        for gid in self.gallery:
            if exclude_gids and gid in exclude_gids:
                continue
            s = self._gallery_similarity(emb, gid)
            if s > best_sim:
                best_sim = s
                best_gid = gid
        return best_gid, best_sim

    def _add_embedding(self, gid: int, emb: np.ndarray):
        self.gallery.setdefault(gid, []).append(emb)
        if len(self.gallery[gid]) > self.max_embeddings_per_id:
            self.gallery[gid] = self.gallery[gid][-self.max_embeddings_per_id:]
        # Update EMA
        if gid in self.gid_ema:
            self.gid_ema[gid] = self._normalise(
                self.ema_alpha * emb + (1 - self.ema_alpha) * self.gid_ema[gid]
            )
        else:
            self.gid_ema[gid] = emb.copy()

    def _mint_new_id(self) -> int:
        gid = self._next_global_id
        self._next_global_id += 1
        return gid

    # ---- public API -------------------------------------------------------
    def resolve(self, tracks: np.ndarray, frame: np.ndarray) -> list[dict]:
        """
        For every track row, produce a dict:
          { 'bbox': (x1,y1,x2,y2),
            'global_id': int,
            'tracker_id': int,
            'conf': float,
            'similarity': float | None,
            'is_new': bool }
        """
        self._frame_count += 1
        results = []
        new_tracker_entries = []  # (index, tracker_id, bbox, embedding)

        self._prev_active_tids = self._active_tids.copy()
        self._active_tids = set()

        for idx, track in enumerate(tracks):
            bbox = track[:4]
            tracker_id = int(track[4])
            conf = float(track[5]) if len(track) > 5 else 0.0
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            self._active_tids.add(tracker_id)

            # --- Case A: tracker_id already mapped ---
            if tracker_id in self.tid_to_gid:
                gid = self.tid_to_gid[tracker_id]
                # Periodically refresh embedding
                if self._frame_count % self.embedding_update_interval == 0:
                    emb = self._extract_embedding(frame, bbox)
                    if emb is not None:
                        # Verify the embedding still matches THIS global ID
                        # to detect tracker ID switches
                        own_sim = self._gallery_similarity(emb, gid)
                        if own_sim >= self.similarity_threshold * 0.8:
                            self._add_embedding(gid, emb)
                        else:
                            # The tracker may have switched – re-check
                            match_gid, match_sim = self._best_gallery_match(emb)
                            if match_gid is not None and match_sim >= self.similarity_threshold:
                                # Remap to the correct identity
                                if match_gid != gid:
                                    print(f"  [ReID] Correcting tracker {tracker_id}: "
                                          f"G{gid} → G{match_gid} (sim={match_sim:.3f})")
                                    self.tid_to_gid[tracker_id] = match_gid
                                    gid = match_gid
                                self._add_embedding(gid, emb)
                            else:
                                # Keep current assignment but add embedding anyway
                                self._add_embedding(gid, emb)

                self.gid_last_seen[gid] = self._frame_count
                self.tid_last_center[tracker_id] = (cx, cy)
                results.append(dict(
                    bbox=tuple(bbox), global_id=gid, tracker_id=tracker_id,
                    conf=conf, similarity=None, is_new=False))
                continue

            # --- Case B: new tracker_id → need embedding match ---
            emb = self._extract_embedding(frame, bbox)
            if emb is None:
                # Can't extract → mint a throwaway global id
                gid = self._mint_new_id()
                self.tid_to_gid[tracker_id] = gid
                self.gid_last_seen[gid] = self._frame_count
                self.tid_last_center[tracker_id] = (cx, cy)
                results.append(dict(
                    bbox=tuple(bbox), global_id=gid, tracker_id=tracker_id,
                    conf=conf, similarity=None, is_new=True))
                continue

            # Defer to batch assignment
            new_tracker_entries.append((len(results), tracker_id, bbox, emb, conf, cx, cy))
            # Placeholder
            results.append(None)

        # --- Batch assignment for new tracker_ids (Hungarian) ---
        if new_tracker_entries:
            self._assign_new_tracks(new_tracker_entries, results)

        # --- Case C: Cleanup stale tid_to_gid mappings ---
        # If a tracker ID hasn't been seen for max_age frames, we can release it
        # However, the gallery (global IDs) MUST persist.
        tids_to_remove = []
        for tid in self.tid_to_gid:
            if tid not in self._active_tids and tid not in self._prev_active_tids:
                # We could remove it, but let's keep it for a few more frames
                # to handle brief flicker. DeepOCSORT handles max_age internally,
                # so we just need to mirror that.
                pass 

        return results

    def _assign_new_tracks(self, entries, results):
        """
        Assign new tracker_ids to gallery entries using Hungarian algorithm
        (or greedy if scipy is unavailable). This prevents two new tracks from
        claiming the same gallery identity.
        """
        n_new = len(entries)
        gallery_ids = list(self.gallery.keys())
        n_gallery = len(gallery_ids)

        if n_gallery == 0:
            # No gallery yet — all are new persons
            for (result_idx, tracker_id, bbox, emb, conf, cx, cy) in entries:
                gid = self._mint_new_id()
                self._add_embedding(gid, emb)
                self.tid_to_gid[tracker_id] = gid
                self.gid_last_seen[gid] = self._frame_count
                self.tid_last_center[tracker_id] = (cx, cy)
                print(f"  [ReID] Tracker {tracker_id} → NEW Global ID {gid} (empty gallery)")
                results[result_idx] = dict(
                    bbox=tuple(bbox), global_id=gid, tracker_id=tracker_id,
                    conf=conf, similarity=0.0, is_new=True)
            return

        # Build cost matrix: rows = new tracks, cols = gallery IDs
        sim_matrix = np.zeros((n_new, n_gallery), dtype=np.float32)
        for i, (_, _, _, emb, _, _, _) in enumerate(entries):
            for j, gid in enumerate(gallery_ids):
                sim_matrix[i, j] = self._gallery_similarity(emb, gid)

        # Convert to cost (we want to maximise similarity → minimise negative)
        cost_matrix = 1.0 - sim_matrix

        # --- Assignment ---
        claimed_gids = set()

        if HAS_SCIPY and n_new > 1:
            # Pad cost matrix to be square if needed
            dim = max(n_new, n_gallery)
            padded = np.ones((dim, dim), dtype=np.float32) * 1.0  # high cost for dummy
            padded[:n_new, :n_gallery] = cost_matrix
            row_ind, col_ind = linear_sum_assignment(padded)

            for r, c in zip(row_ind, col_ind):
                if r >= n_new:
                    continue
                result_idx, tracker_id, bbox, emb, conf, cx, cy = entries[r]
                if c < n_gallery:
                    assigned_gid = gallery_ids[c]
                    sim = sim_matrix[r, c]
                    if sim >= self.similarity_threshold and assigned_gid not in claimed_gids:
                        # Re-identified
                        gid = assigned_gid
                        self._add_embedding(gid, emb)
                        claimed_gids.add(gid)
                        self.tid_to_gid[tracker_id] = gid
                        self.gid_last_seen[gid] = self._frame_count
                        self.tid_last_center[tracker_id] = (cx, cy)
                        print(f"  [ReID] Tracker {tracker_id} → Re-identified as G{gid} (sim={sim:.3f})")
                        results[result_idx] = dict(
                            bbox=tuple(bbox), global_id=gid, tracker_id=tracker_id,
                            conf=conf, similarity=float(sim), is_new=False)
                        continue

                # New person
                gid = self._mint_new_id()
                self._add_embedding(gid, emb)
                self.tid_to_gid[tracker_id] = gid
                self.gid_last_seen[gid] = self._frame_count
                self.tid_last_center[tracker_id] = (cx, cy)
                best_sim = float(np.max(sim_matrix[r])) if n_gallery > 0 else 0.0
                print(f"  [ReID] Tracker {tracker_id} → NEW Global ID {gid} (best_sim={best_sim:.3f})")
                results[result_idx] = dict(
                    bbox=tuple(bbox), global_id=gid, tracker_id=tracker_id,
                    conf=conf, similarity=float(best_sim), is_new=True)
        else:
            # Greedy assignment (single new track, or no scipy)
            for i, (result_idx, tracker_id, bbox, emb, conf, cx, cy) in enumerate(entries):
                match_gid, sim = self._best_gallery_match(emb, exclude_gids=claimed_gids)

                if match_gid is not None and sim >= self.similarity_threshold:
                    gid = match_gid
                    self._add_embedding(gid, emb)
                    claimed_gids.add(gid)
                    print(f"  [ReID] Tracker {tracker_id} → Re-identified as G{gid} (sim={sim:.3f})")
                    results[result_idx] = dict(
                        bbox=tuple(bbox), global_id=gid, tracker_id=tracker_id,
                        conf=conf, similarity=float(sim), is_new=False)
                else:
                    gid = self._mint_new_id()
                    self._add_embedding(gid, emb)
                    print(f"  [ReID] Tracker {tracker_id} → NEW Global ID {gid} (best_sim={sim:.3f})")
                    results[result_idx] = dict(
                        bbox=tuple(bbox), global_id=gid, tracker_id=tracker_id,
                        conf=conf, similarity=float(sim) if sim else 0.0, is_new=True)

                self.tid_to_gid[tracker_id] = gid
                self.gid_last_seen[gid] = self._frame_count
                self.tid_last_center[tracker_id] = (cx, cy)

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
        # Rebuild EMA from gallery
        for gid, emb_list in self.gallery.items():
            if emb_list:
                self.gid_ema[gid] = self._normalise(np.mean(np.stack(emb_list), axis=0))
        print(f"  [IdentityManager] Loaded gallery ({len(self.gallery)} IDs) from {path}")


# ---------------------------------------------------------------------------
#  Module 4 – People Counter (line-crossing with direction)
# ---------------------------------------------------------------------------
class PeopleCounter:
    """
    Direction-based line-crossing counter.
    • crossing downward (y increases past line) → IN
    • crossing upward  (y decreases past line) → OUT
    Each global_id counted at most once per direction.
    """

    def __init__(self, line_y: int, hysteresis: int = 10):
        self.line_y = line_y
        self.hysteresis = hysteresis    # pixels of dead-zone around line
        self.prev_cy: dict[int, float] = {}
        self.counted_in: set[int] = set()
        self.counted_out: set[int] = set()

    def update(self, resolved: list[dict]) -> tuple[int, int]:
        for r in resolved:
            if r is None:
                continue
            gid = r["global_id"]
            x1, y1, x2, y2 = r["bbox"]
            cy = (y1 + y2) / 2.0

            if gid in self.prev_cy:
                prev = self.prev_cy[gid]
                # Crossed downward → IN
                if prev < self.line_y - self.hysteresis and cy >= self.line_y and gid not in self.counted_in:
                    self.counted_in.add(gid)
                    print(f"  [Counter] Global ID {gid} → IN ↓")
                # Crossed upward → OUT
                elif prev > self.line_y + self.hysteresis and cy <= self.line_y and gid not in self.counted_out:
                    self.counted_out.add(gid)
                    print(f"  [Counter] Global ID {gid} → OUT ↑")

            self.prev_cy[gid] = cy

        return len(self.counted_in), len(self.counted_out)


# ---------------------------------------------------------------------------
#  Module 5 – Visualizer (Premium HUD)
# ---------------------------------------------------------------------------
class Visualizer:
    """Draws bounding boxes, IDs, counting line, and a polished stats overlay."""

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
        return (int(c[2]), int(c[1]), int(c[0]))  # BGR

    @staticmethod
    def _draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
        """Draw a rectangle with rounded corners."""
        x1, y1 = pt1
        x2, y2 = pt2
        r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
        if r <= 0:
            cv2.rectangle(img, pt1, pt2, color, thickness)
            return
        # Top-left corner
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        # Top-right corner
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        # Bottom-right corner
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
        # Bottom-left corner
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        # Fill rectangles
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)

    @staticmethod
    def draw(frame, resolved, line_y, in_count, out_count, total_unique,
             frame_idx, fps_current=0.0, active_in_frame=0):
        h, w = frame.shape[:2]

        # --- Counting line (dashed effect) ---
        dash_len = 20
        for x in range(0, w, dash_len * 2):
            cv2.line(frame, (x, line_y), (min(x + dash_len, w), line_y), (0, 255, 255), 2)
        cv2.putText(frame, "-- COUNTING LINE --", (w // 2 - 100, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # Direction arrows
        cv2.putText(frame, "IN v", (15, line_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        cv2.putText(frame, "^ OUT", (15, line_y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

        # --- Per-person visualisation ---
        for r in resolved:
            if r is None:
                continue
            gid = r["global_id"]
            tid = r["tracker_id"]
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            conf = r["conf"]
            color = Visualizer._color(gid)

            # Bounding box with slight transparency
            overlay_box = frame.copy()
            cv2.rectangle(overlay_box, (x1, y1), (x2, y2), color, 2)
            # Thicker outline for clarity
            cv2.rectangle(overlay_box, (x1, y1), (x2, y2), color, 3)
            cv2.addWeighted(overlay_box, 0.9, frame, 0.1, 0, frame)

            # Center point
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)

            # ID badge at top of bounding box
            label = f"ID:{gid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            badge_y1 = max(0, y1 - th - 12)
            badge_y2 = y1
            badge_x2 = x1 + tw + 12
            # Badge background
            cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), (255, 255, 255), 1)
            cv2.putText(frame, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Confidence & tracker ID below the box
            sub_label = f"T{tid} {conf:.0%}"
            cv2.putText(frame, sub_label, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # --- Top HUD bar ---
        bar_h = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Stats
        # IN count (green)
        cv2.putText(frame, f"IN: {in_count}", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)
        # OUT count (red)
        cv2.putText(frame, f"OUT: {out_count}", (180, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 255), 2)
        # Total unique (yellow)
        cv2.putText(frame, f"UNIQUE: {total_unique}", (360, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Second row
        active = sum(1 for r in resolved if r is not None)
        cv2.putText(frame, f"Frame: {frame_idx}", (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, f"Active: {active}", (180, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        if fps_current > 0:
            cv2.putText(frame, f"FPS: {fps_current:.1f}", (320, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Divider line below HUD
        cv2.line(frame, (0, bar_h), (w, bar_h), (80, 80, 80), 1)

        return frame


# ---------------------------------------------------------------------------
#  Main orchestrator
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Ultimate Person Tracker & Counter")
    p.add_argument("--source", type=str, required=True, help="Input video path")
    p.add_argument("--output", type=str, default="output_ultimate_tracker.mp4")
    p.add_argument("--yolo-model", type=str, default="yolov8n.pt")
    p.add_argument("--reid-model", type=str, default="osnet_x1_0_msmt17.pt")
    p.add_argument("--conf", type=float, default=0.35, help="YOLO detection confidence")
    p.add_argument("--similarity", type=float, default=0.65,
                    help="Cosine similarity threshold for ReID (0.60-0.75 recommended)")
    p.add_argument("--line-ratio", type=float, default=0.50,
                    help="Counting line position as fraction of frame height (0-1)")
    p.add_argument("--max-age", type=int, default=120,
                    help="DeepOCSORT track buffer (frames to keep lost tracks)")
    p.add_argument("--gallery", type=str, default="person_gallery.json",
                    help="Path to save/load the embedding gallery")
    p.add_argument("--show", action="store_true", help="Display live window")
    p.add_argument("--debug", action="store_true", help="Print per-frame debug info")
    p.add_argument("--fresh", action="store_true",
                    help="Start with fresh gallery (ignore saved file)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  ULTIMATE PERSON TRACKER & RE-IDENTIFICATION SYSTEM")
    print("=" * 70)
    print(f"  Source          : {args.source}")
    print(f"  YOLO model      : {args.yolo_model}")
    print(f"  ReID model      : {args.reid_model}")
    print(f"  Similarity thr  : {args.similarity}")
    print(f"  Line ratio      : {args.line_ratio}")
    print(f"  Track buffer    : {args.max_age} frames")
    print(f"  Gallery file    : {args.gallery}")
    print(f"  Fresh gallery   : {args.fresh}")
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
    if not args.fresh:
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

    counter = PeopleCounter(line_y=line_y, hysteresis=10)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"\n  Video: {width}x{height} @ {fps}fps  ({total_frames} frames)")
    print(f"  Counting line at y={line_y}")
    print(f"  Device: {device}\n")

    frame_idx = 0
    t0 = time.time()
    fps_current = 0.0

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
            in_count, out_count = counter.update(resolved)
            total_unique = len(identity_mgr.gallery)

            # 5. Debug log
            if args.debug or frame_idx % 30 == 0:
                for r in resolved:
                    if r is None:
                        continue
                    sim_str = f"{r['similarity']:.3f}" if r["similarity"] is not None else "—"
                    print(
                        f"  [F{frame_idx:>5}] T{r['tracker_id']:>3} → G{r['global_id']:>3}  "
                        f"bbox=({int(r['bbox'][0])},{int(r['bbox'][1])},"
                        f"{int(r['bbox'][2])},{int(r['bbox'][3])})  "
                        f"conf={r['conf']:.2f}  sim={sim_str}"
                    )

            # 6. FPS calculation
            elapsed = time.time() - t0
            if elapsed > 0:
                fps_current = frame_idx / elapsed

            # 7. Visualise
            frame = Visualizer.draw(
                frame, resolved, line_y, in_count, out_count, total_unique,
                frame_idx, fps_current=fps_current
            )

            out.write(frame)
            if args.show:
                cv2.imshow("Ultimate Person Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Progress
            if frame_idx % 60 == 0:
                print(
                    f"  Progress: {frame_idx}/{total_frames} "
                    f"({100 * frame_idx / total_frames:.1f}%)  "
                    f"{fps_current:.1f} fps  |  "
                    f"IN={in_count}  OUT={out_count}  UNIQUE={total_unique}"
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
    print(f"  Wall-clock time  : {elapsed:.1f}s  ({frame_idx / elapsed:.1f} fps)")
    print(f"  IN count         : {in_count}")
    print(f"  OUT count        : {out_count}")
    print(f"  Total unique IDs : {total_unique}")
    print(f"  Output video     : {args.output}")
    print(f"  Gallery saved    : {args.gallery}")
    print("=" * 70)

    # Detailed ID summary
    print("\n  Global ID summary:")
    for gid in sorted(identity_mgr.gallery.keys()):
        n_emb = len(identity_mgr.gallery[gid])
        entered = "✓" if gid in counter.counted_in else "—"
        exited  = "✓" if gid in counter.counted_out else "—"
        print(f"    G{gid:>3}  embeddings={n_emb:>2}  IN={entered}  OUT={exited}")


if __name__ == "__main__":
    main()
