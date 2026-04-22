"""
Track people with a short-term tracker plus a stricter global identity manager.

The local tracker handles continuous motion. A separate global-ID layer handles
re-identification so one unstable local track ID does not immediately become a
new person ID.
"""
import argparse
import os
import pickle
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

# Add boxmot to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Keep Ultralytics settings local to the workspace to avoid permission issues.
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_cfg"))

from ultralytics import YOLO

from boxmot.trackers.strongsort.strong_sort import StrongSORT
from boxmot.utils import WEIGHTS


def normalize(feature):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    return feature / (np.linalg.norm(feature) + 1e-8)


def tlwh_to_xyxy(tlwh, width, height):
    x, y, w, h = tlwh
    x1 = max(int(x), 0)
    y1 = max(int(y), 0)
    x2 = min(int(x + w), width - 1)
    y2 = min(int(y + h), height - 1)
    return x1, y1, x2, y2


def color_for_id(track_id):
    rng = np.random.default_rng(track_id)
    return tuple(int(x) for x in rng.integers(64, 255, size=3))


class GlobalIdentity:
    def __init__(self, global_id, feature, tlwh, frame_idx, local_track_id, confidence):
        feature = normalize(feature)
        tlwh = np.asarray(tlwh, dtype=np.float32)
        self.global_id = int(global_id)
        self.ema_feature = feature.copy()
        self.gallery = deque([feature.copy()], maxlen=60)
        self.last_tlwh = tlwh.copy()
        self.mean_width = float(tlwh[2])
        self.mean_height = float(tlwh[3])
        self.first_frame = int(frame_idx)
        self.last_frame = int(frame_idx)
        self.last_local_track_id = int(local_track_id)
        self.last_confidence = float(confidence)
        self.observations = 1

    def update(self, feature, tlwh, frame_idx, local_track_id, confidence):
        feature = normalize(feature)
        tlwh = np.asarray(tlwh, dtype=np.float32)
        self.ema_feature = normalize(0.85 * self.ema_feature + 0.15 * feature)
        self.gallery.append(feature.copy())
        self.last_tlwh = tlwh.copy()
        self.mean_width = 0.8 * self.mean_width + 0.2 * float(tlwh[2])
        self.mean_height = 0.8 * self.mean_height + 0.2 * float(tlwh[3])
        self.last_frame = int(frame_idx)
        self.last_local_track_id = int(local_track_id)
        self.last_confidence = float(confidence)
        self.observations += 1

    def distance(self, feature):
        feature = normalize(feature)
        samples = list(self.gallery)
        samples.append(self.ema_feature)
        matrix = np.vstack(samples)
        distances = 1.0 - np.dot(matrix, feature)
        k = min(5, len(distances))
        best_k = np.partition(distances, k - 1)[:k]
        return float(best_k.mean())

    def to_state(self):
        return {
            "global_id": self.global_id,
            "ema_feature": self.ema_feature,
            "gallery": list(self.gallery),
            "last_tlwh": self.last_tlwh,
            "mean_width": self.mean_width,
            "mean_height": self.mean_height,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "last_local_track_id": self.last_local_track_id,
            "last_confidence": self.last_confidence,
            "observations": self.observations,
        }

    @classmethod
    def from_state(cls, state):
        obj = cls(
            global_id=state["global_id"],
            feature=state["ema_feature"],
            tlwh=state["last_tlwh"],
            frame_idx=state["last_frame"],
            local_track_id=state.get("last_local_track_id", -1),
            confidence=state.get("last_confidence", 1.0),
        )
        obj.ema_feature = normalize(state["ema_feature"])
        obj.gallery = deque(
            [normalize(feature) for feature in state.get("gallery", [state["ema_feature"]])],
            maxlen=60,
        )
        obj.last_tlwh = np.asarray(state["last_tlwh"], dtype=np.float32)
        obj.mean_width = float(state.get("mean_width", obj.last_tlwh[2]))
        obj.mean_height = float(state.get("mean_height", obj.last_tlwh[3]))
        obj.first_frame = int(state.get("first_frame", obj.first_frame))
        obj.last_frame = int(state.get("last_frame", obj.last_frame))
        obj.last_local_track_id = int(state.get("last_local_track_id", obj.last_local_track_id))
        obj.last_confidence = float(state.get("last_confidence", obj.last_confidence))
        obj.observations = int(state.get("observations", len(obj.gallery)))
        return obj


class GlobalIdentityManager:
    def __init__(
        self,
        reid_threshold=0.18,
        min_confidence=0.5,
        min_box_height=80.0,
        short_gap_frames=20,
        short_gap_spatial_factor=3.0,
        ambiguity_margin=0.04,
        min_size_ratio=0.55,
        max_size_ratio=1.8,
        # --- new: stability gate ---
        stable_min_observations=5,   # frames a track must be seen before counting as "real"
        stable_min_frames_span=8,    # first_frame..last_frame span must be >= this
        # --- new: post-hoc merge ---
        merge_window_frames=60,      # look back this many frames for merge candidates
        merge_threshold=0.14,        # distance threshold for merging two IDs
    ):
        self.reid_threshold = reid_threshold
        self.min_confidence = min_confidence
        self.min_box_height = min_box_height
        self.short_gap_frames = short_gap_frames
        self.short_gap_spatial_factor = short_gap_spatial_factor
        self.ambiguity_margin = ambiguity_margin
        self.min_size_ratio = min_size_ratio
        self.max_size_ratio = max_size_ratio
        self.stable_min_observations = stable_min_observations
        self.stable_min_frames_span = stable_min_frames_span
        self.merge_window_frames = merge_window_frames
        self.merge_threshold = merge_threshold

        self.identities = {}
        self.local_to_global = {}
        self.next_global_id = 1
        # canonical_id[gid] = the ID this gid was merged into (for redirect)
        self.merged_into: dict[int, int] = {}

    def update_from_tracks(self, tracks, frame_idx, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        assigned_globals = {}
        outputs = []

        active_tracks = [
            track for track in tracks
            if track.is_confirmed() and track.time_since_update <= 1 and len(track.features) > 0
        ]

        for track in active_tracks:
            local_id = int(track.track_id)
            tlwh = np.asarray(track.to_tlwh(), dtype=np.float32)
            feature = normalize(track.features[-1])
            confidence = float(track.conf)
            class_id = int(track.class_id)

            global_id = self._resolve_global_id(
                local_id=local_id,
                feature=feature,
                tlwh=tlwh,
                confidence=confidence,
                frame_idx=frame_idx,
                assigned_globals=assigned_globals,
            )

            # Follow any merge redirects
            global_id = self._canonical(global_id)

            identity = self.identities[global_id]
            identity.update(feature, tlwh, frame_idx, local_id, confidence)
            self.local_to_global[local_id] = global_id
            assigned_globals[global_id] = local_id

            # Post-hoc merge: try to collapse this ID into an older one
            self._try_merge(global_id, frame_idx)
            global_id = self._canonical(global_id)

            x1, y1, x2, y2 = tlwh_to_xyxy(tlwh, frame_w, frame_h)
            outputs.append(
                np.array([x1, y1, x2, y2, global_id, confidence, class_id], dtype=np.float64)
            )

        if outputs:
            return np.asarray(outputs)
        return np.empty((0, 7), dtype=np.float64)

    # ---- stability & merge helpers ----------------------------------------

    def _canonical(self, gid: int) -> int:
        """Follow merge chain to the root ID."""
        visited = set()
        while gid in self.merged_into:
            if gid in visited:
                break  # cycle guard
            visited.add(gid)
            gid = self.merged_into[gid]
        return gid

    def _is_stable(self, identity) -> bool:
        """True if this identity has enough evidence to count as a real person."""
        span = identity.last_frame - identity.first_frame
        return (
            identity.observations >= self.stable_min_observations
            and span >= self.stable_min_frames_span
        )

    def _try_merge(self, new_gid: int, frame_idx: int):
        """
        After a new ID accumulates a few observations, check whether it is
        actually the same person as an older ID that recently disappeared.
        If so, merge the newer ID into the older one.
        """
        new_id = self.identities.get(new_gid)
        if new_id is None or new_id.observations < 3:
            return  # not enough data yet

        best_dist, best_gid = 1.0, None
        for gid, identity in self.identities.items():
            if gid == new_gid:
                continue
            if gid in self.merged_into:
                continue  # already merged away
            gap = frame_idx - identity.last_frame
            if gap <= 0 or gap > self.merge_window_frames:
                continue
            # Only merge into older, more established IDs
            if identity.observations < new_id.observations:
                continue
            dist = identity.distance(new_id.ema_feature)
            if dist < best_dist:
                best_dist = dist
                best_gid = gid

        if best_gid is not None and best_dist < self.merge_threshold:
            # Absorb new_gid into best_gid
            survivor = self.identities[best_gid]
            victim = self.identities[new_gid]
            for emb in victim.gallery:
                survivor.gallery.append(emb)
            survivor.ema_feature = normalize(
                0.5 * survivor.ema_feature + 0.5 * victim.ema_feature
            )
            survivor.observations += victim.observations
            survivor.last_frame = max(survivor.last_frame, victim.last_frame)
            survivor.last_tlwh = victim.last_tlwh
            self.merged_into[new_gid] = best_gid
            for lid, gid in list(self.local_to_global.items()):
                if gid == new_gid:
                    self.local_to_global[lid] = best_gid
            print(f"  [Merge] G{new_gid} → G{best_gid} (dist={best_dist:.3f})")

    def stable_unique_ids(self) -> int:
        """Count only IDs with enough evidence — the real people count."""
        return sum(
            1 for gid, identity in self.identities.items()
            if gid not in self.merged_into and self._is_stable(identity)
        )

    def active_ids_count(self, current_frame: int, active_window: int = 5) -> int:
        """Count IDs seen within the last active_window frames."""
        return sum(
            1 for gid, identity in self.identities.items()
            if gid not in self.merged_into
            and (current_frame - identity.last_frame) <= active_window
        )

    def total_unique_ids(self):
        """Total IDs ever created minus merged-away ones."""
        return sum(1 for gid in self.identities if gid not in self.merged_into)

    def get_stats(self):
        stable = self.stable_unique_ids()
        total = self.total_unique_ids()
        return {
            "stable_unique": stable,
            "total_ids": total,
            "merged_away": len(self.merged_into),
            "total_features": sum(len(identity.gallery) for identity in self.identities.values()),
        }

    def save(self, path):
        data = {
            "next_global_id": self.next_global_id,
            "identities": {gid: identity.to_state() for gid, identity in self.identities.items()},
        }
        with open(path, "wb") as handle:
            pickle.dump(data, handle)
        print(f"Saved {len(self.identities)} global identities to {path}")

    def load(self, path):
        if not os.path.exists(path):
            print(f"Gallery file not found: {path}")
            return
        try:
            with open(path, "rb") as handle:
                data = pickle.load(handle)
            identities = data.get("identities")
            if not isinstance(identities, dict):
                print(f"Skipping incompatible gallery file: {path}")
                return
            self.identities = {
                int(gid): GlobalIdentity.from_state(state)
                for gid, state in identities.items()
            }
            self.next_global_id = int(data.get("next_global_id", len(self.identities) + 1))
            print(f"Loaded {len(self.identities)} global identities from {path}")
        except Exception as exc:
            print(f"Could not load gallery {path}: {exc}")

    def _resolve_global_id(self, local_id, feature, tlwh, confidence, frame_idx, assigned_globals):
        if local_id in self.local_to_global:
            global_id = self.local_to_global[local_id]
            # Follow merge chain to canonical ID
            global_id = self._canonical(global_id)
            if global_id in self.identities and global_id not in assigned_globals:
                return global_id

        matched_global_id = self._match_existing_identity(
            feature=feature,
            tlwh=tlwh,
            confidence=confidence,
            frame_idx=frame_idx,
            assigned_globals=assigned_globals,
        )
        if matched_global_id is not None:
            return matched_global_id

        global_id = self.next_global_id
        self.next_global_id += 1
        self.identities[global_id] = GlobalIdentity(
            global_id=global_id,
            feature=feature,
            tlwh=tlwh,
            frame_idx=frame_idx,
            local_track_id=local_id,
            confidence=confidence,
        )
        return global_id

    def _match_existing_identity(self, feature, tlwh, confidence, frame_idx, assigned_globals):
        if confidence < self.min_confidence or tlwh[3] < self.min_box_height:
            return None

        candidates = []
        for global_id, identity in self.identities.items():
            if global_id in assigned_globals:
                continue
            # Skip merged-away IDs
            if global_id in self.merged_into:
                continue

            gap = frame_idx - identity.last_frame
            if gap <= 0:
                continue

            if identity.observations < 3 and gap > self.short_gap_frames:
                continue

            if not self._passes_geometry_gate(identity, tlwh, gap):
                continue

            distance = identity.distance(feature)
            threshold = self.reid_threshold
            if gap <= self.short_gap_frames:
                threshold = min(self.reid_threshold + 0.04, 0.24)

            if distance < threshold:
                candidates.append((distance, global_id))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 1 and (candidates[1][0] - candidates[0][0]) < self.ambiguity_margin:
            return None
        return candidates[0][1]

    def _passes_geometry_gate(self, identity, tlwh, gap):
        det_w = float(tlwh[2])
        det_h = float(tlwh[3])
        if det_w <= 1 or det_h <= 1:
            return False

        width_ratio = det_w / max(identity.mean_width, 1.0)
        height_ratio = det_h / max(identity.mean_height, 1.0)
        if not (self.min_size_ratio <= width_ratio <= self.max_size_ratio):
            return False
        if not (self.min_size_ratio <= height_ratio <= self.max_size_ratio):
            return False

        if gap <= self.short_gap_frames:
            prev = identity.last_tlwh
            prev_center = prev[:2] + prev[2:] / 2
            det_center = tlwh[:2] + tlwh[2:] / 2
            center_distance = float(np.linalg.norm(det_center - prev_center))
            spatial_limit = self.short_gap_spatial_factor * max(det_h, float(prev[3]), 1.0)
            if center_distance > spatial_limit:
                return False

        return True


def main():
    parser = argparse.ArgumentParser(description="Track with stricter persistent ReID")
    parser.add_argument("--source", type=str, required=True, help="Video file path")
    parser.add_argument("--yolo-model", type=str, default="yolov8m.pt", help="YOLO model")
    parser.add_argument("--reid-model", type=str, default="osnet_x1_0_msmt17.pt", help="ReID model")
    parser.add_argument(
        "--reid-threshold",
        type=float,
        default=0.18,
        help="Global-ID matching threshold (lower=stricter)",
    )
    parser.add_argument("--gallery", type=str, default="gallery.pkl", help="Path to identity gallery file")
    parser.add_argument("--load-gallery", action="store_true", help="Load an existing gallery before tracking")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--max-age", type=int, default=70, help="Max frames to keep lost local tracks")
    parser.add_argument("--classes", nargs="+", type=int, default=[0], help="YOLO class filter")
    parser.add_argument("--output", type=str, default="output_persistent_reid.mp4", help="Output video path")
    parser.add_argument("--show", action="store_true", help="Display video while processing")
    args = parser.parse_args()

    print("=" * 80)
    print("Persistent Re-Identification Tracker")
    print("=" * 80)
    print(f"Video: {args.source}")
    print(f"YOLO Model: {args.yolo_model}")
    print(f"ReID Model: {args.reid_model}")
    print(f"ReID Threshold: {args.reid_threshold}")
    print(f"Max Age: {args.max_age} frames")
    print(f"Classes: {args.classes}")
    print("=" * 80)

    print("Loading YOLO model...")
    yolo = YOLO(args.yolo_model)

    device = "cpu"
    if torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability(0)
            sm_tag = f"sm_{capability[0]}{capability[1]}"
            if sm_tag in torch.cuda.get_arch_list():
                device = "cuda:0"
            else:
                print(f"CUDA device detected but current PyTorch build does not support {sm_tag}; using CPU.")
        except Exception as exc:
            print(f"Could not validate CUDA support ({exc}); using CPU.")

    print("Initializing local tracker...")
    tracker = StrongSORT(
        model_weights=WEIGHTS / args.reid_model,
        device=device,
        fp16=False,
        max_age=args.max_age,
        max_dist=0.2,
        max_iou_dist=0.7,
        n_init=3,
        nn_budget=100,
        mc_lambda=0.995,
        ema_alpha=0.9,
    )

    identity_manager = GlobalIdentityManager(reid_threshold=args.reid_threshold)
    if args.load_gallery and args.gallery and Path(args.gallery).exists():
        identity_manager.load(args.gallery)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.source}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    colors = {}
    frame_idx = 0
    print("\nProcessing video...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = yolo(frame, conf=args.conf, classes=args.classes, verbose=False)[0]
            if len(results.boxes) > 0:
                dets = []
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = float(box.cls[0].cpu().numpy())
                    dets.append([x1, y1, x2, y2, conf, cls])
                dets = np.asarray(dets, dtype=np.float32)
            else:
                dets = np.empty((0, 6), dtype=np.float32)

            tracker.update(dets, frame)
            global_tracks = identity_manager.update_from_tracks(
                tracker.tracker.tracks,
                frame_idx,
                frame.shape,
            )

            for track in global_tracks:
                x1, y1, x2, y2, global_id, conf, cls = track
                global_id = int(global_id)
                if global_id not in colors:
                    colors[global_id] = color_for_id(global_id)
                color = colors[global_id]

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ID: {global_id}"
                (label_width, label_height), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2,
                )
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1) - label_height - 10),
                    (int(x1) + label_width, int(y1)),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            stable_ids = identity_manager.stable_unique_ids()
            active_now = identity_manager.active_ids_count(frame_idx)
            total_ids = identity_manager.total_unique_ids()
            # Line 1: active people on screen right now
            cv2.putText(
                frame,
                f"Frame: {frame_idx}/{total_frames}  |  On screen: {active_now}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2,
            )
            # Line 2: stable unique people seen so far (the real count)
            cv2.putText(
                frame,
                f"Stable unique people: {stable_ids}  (raw IDs: {total_ids})",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2,
            )
            out.write(frame)

            if args.show:
                cv2.imshow("Persistent ReID Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_idx % 30 == 0:
                progress = 100.0 * frame_idx / max(total_frames, 1)
                print(
                    f"Progress: {progress:.1f}% ({frame_idx}/{total_frames}) | "
                    f"On screen: {active_now} | Stable: {stable_ids} | Raw IDs: {total_ids}"
                )

    finally:
        if args.gallery:
            identity_manager.save(args.gallery)
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"\nDone! Output saved to: {args.output}")
    stats = identity_manager.get_stats()
    print(f"Stable unique people  : {stats['stable_unique']}")
    print(f"Total raw IDs created : {stats['total_ids']}")
    print(f"IDs merged away       : {stats['merged_away']}")
    print(f"Total features stored : {stats['total_features']}")


if __name__ == "__main__":
    main()
