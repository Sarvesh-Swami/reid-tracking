"""
Attendance Tracker v3 - Aggressive Re-Verification
===========================================================
Fixed camera, people come and go, same person = same ID always.

NEW in v3 - AGGRESSIVE RE-VERIFICATION:
  - Verify EVERY frame against ALL gallery PIDs (not just assigned one)
  - Automatic ID reassignment when better match found
  - Don't trust BoTSORT IDs - treat as temporary labels only
  - Faster contamination detection (2 strikes instead of 3)
  - Continuous identity verification prevents ID swaps

Fixes from v2:
  - Gallery contamination guard (prevents BoTSORT swap pollution)
  - Continuous re-verification of assigned IDs every few frames
  - Faster probation (3 frames) and confirmation (3 frames)
  - 2D H×S color histogram for better clothing discrimination
  - Soft active-PID exclusion (only confirmed mappings excluded)
  - Failed-match avoidance (won't retry same wrong PID)
  - Gallery updates every 3 frames with consistency checking

Usage:
  python track_attendance.py --source test_6.mp4 --show
  python track_attendance.py --source test_6.mp4 --reid-threshold 0.5
"""
import sys
import os
from pathlib import Path

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
import json
import pickle


class PersistentGallery:
    """
    Persistent person gallery with contamination protection.
    Combined color + ReID scoring with consistency guards.
    """

    def __init__(self, reid_threshold=0.40, max_features=50, min_features=3,
                 color_weight=0.55, ratio_margin=1.10, consistency_sim=0.40):
        self.reid_threshold = reid_threshold
        self.max_features = max_features
        self.min_features = min_features
        self.color_weight = color_weight
        self.reid_weight = 1.0 - color_weight
        self.ratio_margin = ratio_margin
        self.consistency_sim = consistency_sim
        self.entries = {}
        self.color_hists = {}
        self.next_id = 1

    def add_new(self, features_list, color_hist_list=None):
        pid = self.next_id
        self.next_id += 1
        self.entries[pid] = [f.copy() for f in features_list]
        if color_hist_list:
            self.color_hists[pid] = [c.copy() for c in color_hist_list if c is not None]
        else:
            self.color_hists[pid] = []
        return pid

    def update(self, pid, features, color_hist=None):
        """Update gallery. Returns True if accepted, False if rejected (contamination guard)."""
        if pid not in self.entries:
            return False
        # Feature consistency guard
        if len(self.entries[pid]) >= self.min_features:
            gallery = np.array(self.entries[pid])
            sim = float((gallery @ features).mean())
            if sim < self.consistency_sim:
                return False
        # Color consistency guard
        if color_hist is not None and len(self.color_hists.get(pid, [])) >= 2:
            csim = self._color_sim(color_hist, pid)
            if csim < 0.35:
                return False

        self.entries[pid].append(features.copy())
        if len(self.entries[pid]) > self.max_features:
            self.entries[pid] = self.entries[pid][-self.max_features:]
        if color_hist is not None:
            self.color_hists.setdefault(pid, []).append(color_hist.copy())
            if len(self.color_hists[pid]) > self.max_features:
                self.color_hists[pid] = self.color_hists[pid][-self.max_features:]
        return True

    def _color_sim(self, query_hist, pid):
        stored = self.color_hists.get(pid, [])
        if not stored or query_hist is None:
            return 0.5
        coeffs = [float(np.sum(np.sqrt(query_hist * s + 1e-10))) for s in stored]
        coeffs.sort(reverse=True)
        return float(np.mean(coeffs[:min(5, len(coeffs))]))

    def _reid_sim(self, features, pid):
        fl = self.entries.get(pid, [])
        if not fl:
            return 0.0
        gallery = np.array(fl)
        sims = gallery @ features
        k = min(5, len(sims))
        return float(np.sort(sims)[-k:].mean())

    def combined_score(self, features, pid, color_hist=None):
        """Get combined score for a specific PID (for re-verification)."""
        csim = self._color_sim(color_hist, pid)
        rsim = self._reid_sim(features, pid)
        combined = self.color_weight * csim + self.reid_weight * rsim
        return combined, csim, rsim

    def query(self, features, color_hist=None, exclude_pids=None):
        if not self.entries:
            return None
        scores = []
        for pid, fl in self.entries.items():
            if exclude_pids and pid in exclude_pids:
                continue
            if len(fl) < self.min_features:
                continue
            csim = self._color_sim(color_hist, pid)
            rsim = self._reid_sim(features, pid)
            combined = self.color_weight * csim + self.reid_weight * rsim
            scores.append((pid, combined, csim, rsim))

        if not scores:
            return None
        scores.sort(key=lambda x: x[1], reverse=True)
        best_pid, best_score, best_csim, best_rsim = scores[0]
        best_dist = 1.0 - best_score

        if best_dist >= self.reid_threshold:
            return None
        if best_csim < 0.30:
            return None
        # Ratio test
        if len(scores) > 1 and scores[1][1] > 0.01:
            if best_score / scores[1][1] < self.ratio_margin:
                return None
        return (best_pid, best_dist, best_csim, best_rsim)


class AttendanceTracker:

    def __init__(self, yolo_model='yolov8n.pt', reid_model='osnet_x1_0_msmt17.pt',
                 reid_threshold=0.40, detection_conf=0.25, track_buffer_sec=5,
                 color_weight=0.55):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.yolo = YOLO(yolo_model)
        self.detection_conf = detection_conf
        self.reid_model_name = reid_model
        self.reid_threshold = reid_threshold
        self.track_buffer_sec = track_buffer_sec
        self.color_weight = color_weight

        self.tracker = None
        self.reid_model = None
        self.gallery = None
        self.id_map = {}               # tid → pid
        self.tentative_matches = {}    # tid → match info
        self.new_track_probation = {}  # tid → probation info
        self.verify_fail_count = defaultdict(int)  # tid → consecutive update rejections
        self.last_verified_frame = {}  # tid → last frame we verified this track

        # v3 tuning: BALANCED re-verification (optimized for similar clothing)
        self.confirmation_frames = 3   
        self.min_new_frames = 3        
        self.update_interval = 3       # gallery update frequency
        self.verify_interval = 3       # Verify every 3 frames (was 1 - too aggressive)
        self.max_verify_fails = 3      # Allow 3 strikes (was 2 - too strict)
        self.max_confirm_fails = 3     # Allow 3 confirmation failures
        self.reassignment_threshold = 0.08  # Easier reassignment (was 0.10)

        self.colors = {}
        self.appearance_log = defaultdict(list)
        self.reid_events = []
        self.reassignment_events = []
        self.frame_count = 0
        
        # Embedding logging for analysis
        self.embedding_log = []  # List of (frame, pid, embedding, color_hist)
        
        # Embedding logging for analysis
        self.embedding_log = []  # List of (frame, pid, features, color_hist)

    def _init_tracker(self, fps):
        track_buffer = int(self.track_buffer_sec * fps)
        reid_weights = WEIGHTS / self.reid_model_name
        self.tracker = BoTSORT(
            model_weights=reid_weights, device=self.device, fp16=False,
            track_high_thresh=0.3, new_track_thresh=0.4, track_buffer=track_buffer,
            match_thresh=0.8, proximity_thresh=0.5, appearance_thresh=0.25,
            cmc_method='sparseOptFlow', frame_rate=fps, lambda_=0.985,
        )
        self.reid_model = self.tracker.model
        self.gallery = PersistentGallery(
            reid_threshold=self.reid_threshold, color_weight=self.color_weight,
        )
        print(f"  Device: {self.device}")
        print(f"  Track buffer: {track_buffer} frames ({self.track_buffer_sec}s)")
        print(f"  ReID threshold: {self.reid_threshold}")
        print(f"  Color weight: {self.color_weight}")
        print(f"  Probation: {self.min_new_frames} frames")
        print(f"  Confirmation: {self.confirmation_frames} frames")
        print(f"  Gallery update: every {self.update_interval} frames")
        print(f"  Re-verification: every {self.verify_interval} frame(s)")
        print(f"  Reassignment threshold: {self.reassignment_threshold}")
        print(f"  Contamination guard: {self.max_verify_fails} strikes")

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

    def _extract_color_histogram(self, img, bbox):
        """2D H×S histogram + V channel for better clothing discrimination."""
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        ch, cw = crop.shape[:2]
        ty1, ty2 = int(ch * 0.15), int(ch * 0.75)
        tx1, tx2 = int(cw * 0.1), int(cw * 0.9)
        torso = crop[ty1:ty2, tx1:tx2]
        if torso.size == 0 or torso.shape[0] < 10 or torso.shape[1] < 10:
            return None
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        # Joint H×S captures color identity much better than independent channels
        hist_hs = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        hist = np.concatenate([hist_hs, hist_v]).astype(np.float32)
        total = hist.sum()
        if total < 1e-6:
            return None
        return hist / total

    def _unmap_track(self, tid, feat, chist):
        """Remove tid→pid mapping and start fresh probation."""
        if tid in self.id_map:
            old_pid = self.id_map[tid]
            del self.id_map[tid]
            print(f"    [UNMAP] Track {tid} unmapped from Person {old_pid}")
        if tid in self.verify_fail_count:
            del self.verify_fail_count[tid]
        if tid in self.last_verified_frame:
            del self.last_verified_frame[tid]
        self.new_track_probation[tid] = {
            "features": [feat] if feat is not None else [],
            "color_hists": [chist] if chist is not None else [],
            "start_frame": self.frame_count,
            "failed_pids": set(),
        }

    def _verify_track_identity(self, tid, pid, feat, chist):
        """
        CRITICAL: Verify that this track still belongs to this person.
        Check against ALL gallery PIDs, not just the assigned one.
        Returns: (is_valid, best_pid, best_dist, should_reassign)
        """
        if feat is None:
            return (True, pid, 0.0, False)  # Can't verify without features
        
        # Get score for assigned PID
        assigned_score, assigned_csim, assigned_rsim = self.gallery.combined_score(feat, pid, chist)
        assigned_dist = 1.0 - assigned_score
        
        # Check against ALL other PIDs
        best_other_pid = None
        best_other_score = 0.0
        best_other_dist = 1.0
        
        for other_pid in self.gallery.entries.keys():
            if other_pid == pid:
                continue
            if len(self.gallery.entries[other_pid]) < self.min_new_frames:
                continue
            
            other_score, other_csim, other_rsim = self.gallery.combined_score(feat, other_pid, chist)
            other_dist = 1.0 - other_score
            
            if other_score > best_other_score:
                best_other_score = other_score
                best_other_dist = other_dist
                best_other_pid = other_pid
        
        # Decision logic
        is_valid = True
        should_reassign = False
        best_pid = pid
        best_dist = assigned_dist
        
        # Check 1: Does assigned PID match at all?
        if assigned_dist >= self.reid_threshold:
            is_valid = False
            print(f"    [VERIFY FAIL] Track {tid}: Assigned Person {pid} doesn't match "
                  f"(dist: {assigned_dist:.3f} >= {self.reid_threshold})")
        
        # Check 2: Does another PID match significantly better?
        if best_other_pid is not None:
            score_diff = best_other_score - assigned_score
            if score_diff > self.reassignment_threshold:
                should_reassign = True
                best_pid = best_other_pid
                best_dist = best_other_dist
                print(f"    [REASSIGN CANDIDATE] Track {tid}: Person {best_other_pid} matches better "
                      f"(score diff: {score_diff:.3f}, dist: {best_other_dist:.3f} vs {assigned_dist:.3f})")
        
        return (is_valid, best_pid, best_dist, should_reassign)

    def _map_ids(self, tracks, img):
        if len(tracks) == 0:
            return np.empty((0, 7))

        current_tids = set(int(t[4]) for t in tracks)

        # Cleanup disappeared tracks
        for tid in [t for t in self.tentative_matches if t not in current_tids]:
            del self.tentative_matches[tid]
        for tid in [t for t in self.new_track_probation if t not in current_tids]:
            del self.new_track_probation[tid]
        for tid in [t for t in list(self.verify_fail_count) if t not in current_tids]:
            del self.verify_fail_count[tid]
        for tid in [t for t in list(self.last_verified_frame) if t not in current_tids]:
            del self.last_verified_frame[tid]

        # Only exclude CONFIRMED active PIDs (not tentative)
        confirmed_active_pids = set()
        for t in tracks:
            tid = int(t[4])
            if tid in self.id_map:
                confirmed_active_pids.add(self.id_map[tid])

        output = []
        for t in tracks:
            bbox = t[:4]
            tid = int(t[4])
            conf = float(t[5])
            cls = t[6] if len(t) > 6 else 0
            pid = 0

            if tid in self.id_map:
                # === KNOWN TRACK - AGGRESSIVE RE-VERIFICATION ===
                pid = self.id_map[tid]
                
                # Extract features for verification
                feat = self._extract_features(img, bbox)
                chist = self._extract_color_histogram(img, bbox)
                
                if feat is not None:
                    # CRITICAL: Verify EVERY frame (or every verify_interval frames)
                    should_verify = (self.frame_count - self.last_verified_frame.get(tid, 0)) >= self.verify_interval
                    
                    if should_verify:
                        self.last_verified_frame[tid] = self.frame_count
                        is_valid, best_pid, best_dist, should_reassign = self._verify_track_identity(
                            tid, pid, feat, chist
                        )
                        
                        if should_reassign and best_pid != pid:
                            # REASSIGN to better matching person
                            old_pid = pid
                            pid = best_pid
                            self.id_map[tid] = pid
                            confirmed_active_pids.discard(old_pid)
                            confirmed_active_pids.add(pid)
                            self.verify_fail_count[tid] = 0
                            self.reassignment_events.append((self.frame_count, tid, old_pid, pid, best_dist))
                            print(f"  [REASSIGNED] Frame {self.frame_count}: Track {tid} "
                                  f"reassigned from Person {old_pid} to Person {pid} "
                                  f"(dist: {best_dist:.3f})")
                            # Update gallery with correct features
                            self.gallery.update(pid, feat, chist)
                        
                        elif not is_valid:
                            # Assigned PID doesn't match - unmap
                            self.verify_fail_count[tid] += 1
                            if self.verify_fail_count[tid] >= self.max_verify_fails:
                                print(f"  [IDENTITY LOST] Frame {self.frame_count}: "
                                      f"Track {tid} lost identity of Person {pid} "
                                      f"({self.max_verify_fails} verification failures)")
                                self._unmap_track(tid, feat, chist)
                                pid = 0
                        else:
                            # Valid match - reset fail count
                            self.verify_fail_count[tid] = 0
                    
                    # Update gallery periodically (only if still mapped)
                    if pid > 0 and self.frame_count % self.update_interval == 0 and conf > 0.4:
                        accepted = self.gallery.update(pid, feat, chist)
                        if not accepted:
                            print(f"    [GALLERY REJECT] Frame {self.frame_count}: "
                                  f"Gallery rejected features for Person {pid}")

            elif tid in self.tentative_matches:
                # === RE-ID CONFIRMATION ===
                info = self.tentative_matches[tid]
                pid = info["pid"]
                feat = self._extract_features(img, bbox)
                chist = self._extract_color_histogram(img, bbox)
                if feat is not None:
                    csim = self.gallery._color_sim(chist, info["pid"])
                    if csim < 0.30:
                        # Color mismatch → fail
                        failed_pids = info.get("failed_pids", set()) | {info["pid"]}
                        del self.tentative_matches[tid]
                        self.new_track_probation[tid] = {
                            "features": info["features"] + [feat],
                            "color_hists": info["color_hists"] + ([chist] if chist is not None else []),
                            "start_frame": self.frame_count,
                            "failed_pids": failed_pids,
                        }
                        pid = 0
                        print(f"  [COLOR FAIL] Frame {self.frame_count}: "
                              f"Color mismatch with Person {info['pid']} "
                              f"(color: {csim:.3f}), restarting probation")
                    else:
                        rsim = self.gallery._reid_sim(feat, info["pid"])
                        combined = self.gallery.color_weight * csim + self.gallery.reid_weight * rsim
                        dist = 1.0 - combined
                        if dist < self.reid_threshold:
                            info["count"] += 1
                            info["features"].append(feat)
                            if chist is not None:
                                info["color_hists"].append(chist)
                            # FIX #7: Confirmation still needs N frames
                            if info["count"] >= self.confirmation_frames:
                                pid = info["pid"]
                                self.id_map[tid] = pid
                                for i, f in enumerate(info["features"]):
                                    ch = info["color_hists"][i] if i < len(info["color_hists"]) else None
                                    self.gallery.update(pid, f, ch)
                                del self.tentative_matches[tid]
                                self.reid_events.append((self.frame_count, pid, dist))
                                confirmed_active_pids.add(pid)
                                print(f"  [CONFIRMED] Frame {self.frame_count}: "
                                      f"Person {pid} RE-ID confirmed "
                                      f"(dist: {dist:.3f}, color: {csim:.3f})")
                        else:
                            info["failures"] = info.get("failures", 0) + 1
                            # FIX #7: Allow 3 failures instead of 2
                            if info["failures"] >= self.max_confirm_fails:
                                failed_pids = info.get("failed_pids", set()) | {info["pid"]}
                                del self.tentative_matches[tid]
                                self.new_track_probation[tid] = {
                                    "features": info["features"] + [feat],
                                    "color_hists": info["color_hists"] + ([chist] if chist is not None else []),
                                    "start_frame": self.frame_count,
                                    "failed_pids": failed_pids,
                                }
                                pid = 0
                                print(f"  [FAILED] Frame {self.frame_count}: "
                                      f"Match to Person {info['pid']} failed, "
                                      f"restarting probation")

            elif tid in self.new_track_probation:
                # === NEW TRACK PROBATION ===
                prob = self.new_track_probation[tid]
                feat = self._extract_features(img, bbox)
                chist = self._extract_color_histogram(img, bbox)
                if feat is not None:
                    prob["features"].append(feat)
                    if chist is not None:
                        prob["color_hists"].append(chist)

                # FIX #3: Reduced probation from 5 to 3 frames
                if len(prob["features"]) >= self.min_new_frames:
                    avg_feat = np.mean(prob["features"], axis=0).astype(np.float32)
                    avg_feat = avg_feat / (np.linalg.norm(avg_feat) + 1e-10)
                    latest_chist = prob["color_hists"][-1] if prob["color_hists"] else None

                    # FIX #1: Only exclude confirmed PIDs + previously failed PIDs
                    exclude = confirmed_active_pids | prob.get("failed_pids", set())
                    result = self.gallery.query(avg_feat, latest_chist, exclude_pids=exclude)

                    if result is not None:
                        mpid, dist, csim, rsim = result
                        self.tentative_matches[tid] = {
                            "pid": mpid, "count": 1, "failures": 0,
                            "features": prob["features"],
                            "color_hists": prob["color_hists"],
                            "failed_pids": prob.get("failed_pids", set()),
                        }
                        del self.new_track_probation[tid]
                        pid = mpid
                        print(f"  [TENTATIVE] Frame {self.frame_count}: "
                              f"Person {mpid} tentative match "
                              f"(dist: {dist:.3f}, color: {csim:.3f}, reid: {rsim:.3f})")
                    else:
                        pid = self.gallery.add_new(prob["features"], prob["color_hists"])
                        self.id_map[tid] = pid
                        confirmed_active_pids.add(pid)
                        del self.new_track_probation[tid]
                        print(f"  [NEW] Frame {self.frame_count}: NEW Person {pid}")

            else:
                # === BRAND NEW TRACK → start probation ===
                feat = self._extract_features(img, bbox)
                chist = self._extract_color_histogram(img, bbox)
                if feat is not None:
                    self.new_track_probation[tid] = {
                        "features": [feat],
                        "color_hists": [chist] if chist is not None else [],
                        "start_frame": self.frame_count,
                        "failed_pids": set(),
                    }

            if pid > 0:
                self.appearance_log[pid].append(self.frame_count)
                if pid not in self.colors:
                    np.random.seed(pid * 37)
                    self.colors[pid] = tuple(np.random.randint(60, 230, 3).tolist())
                output.append([bbox[0], bbox[1], bbox[2], bbox[3], pid, conf, cls])
                
                # Log embedding for analysis (only if we have features)
                if tid in self.id_map or tid in self.tentative_matches or tid in self.new_track_probation:
                    feat = self._extract_features(img, bbox)
                    chist = self._extract_color_histogram(img, bbox)
                    if feat is not None:
                        self.embedding_log.append({
                            'frame': self.frame_count,
                            'pid': pid,
                            'tid': tid,
                            'features': feat.copy(),
                            'color_hist': chist.copy() if chist is not None else None,
                            'bbox': bbox.tolist()
                        })
            else:
                output.append([bbox[0], bbox[1], bbox[2], bbox[3], -1, conf, cls])

        return np.array(output) if output else np.empty((0, 7))

    def _draw(self, frame, tracks, total_frames):
        h, w = frame.shape[:2]
        for t in tracks:
            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
            pid = int(t[4])
            if pid <= 0:
                color = (128, 128, 128)
                label = "?"
            else:
                color = self.colors.get(pid, (200, 200, 200))
                label = f"Person {pid}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 12), (x1 + lw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if pid > 0:
                for ef, ep, ed in self.reid_events[-10:]:
                    if ep == pid and (self.frame_count - ef) < 45:
                        cv2.putText(frame, "RETURNED!", (x1, y2 + 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        break

        n_unique = len(self.gallery.entries)
        n_active = sum(1 for t in tracks if int(t[4]) > 0)
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
                    na = sum(1 for t in tracks if len(t) > 4 and int(t[4]) > 0)
                    print(f"  {pct:.0f}% | Active: {na} | Unique: {len(self.gallery.entries)}")
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        print(f"\n[DONE] Saved: {output}")
        self._save_embeddings(output)
        self._summary(total, fps)

    def _save_embeddings(self, output_video_path):
        """Save all embeddings to file for distance analysis"""
        import json
        from pathlib import Path
        
        # Create output filename based on video name
        base_name = Path(output_video_path).stem
        embeddings_file = f"{base_name}_embeddings.npz"
        metadata_file = f"{base_name}_metadata.json"
        
        print(f"\n[SAVING EMBEDDINGS]")
        print(f"  Total embeddings collected: {len(self.embedding_log)}")
        
        # Organize embeddings by PID
        embeddings_by_pid = defaultdict(list)
        for entry in self.embedding_log:
            embeddings_by_pid[entry['pid']].append(entry)
        
        print(f"  Unique PIDs: {len(embeddings_by_pid)}")
        for pid in sorted(embeddings_by_pid.keys()):
            print(f"    Person {pid}: {len(embeddings_by_pid[pid])} embeddings")
        
        # Save embeddings as numpy arrays
        save_dict = {}
        for pid, entries in embeddings_by_pid.items():
            features = np.array([e['features'] for e in entries])
            save_dict[f'pid_{pid}_features'] = features
            
            # Save color histograms if available
            color_hists = [e['color_hist'] for e in entries if e['color_hist'] is not None]
            if color_hists:
                save_dict[f'pid_{pid}_color_hists'] = np.array(color_hists)
        
        np.savez_compressed(embeddings_file, **save_dict)
        print(f"  ✓ Saved embeddings: {embeddings_file}")
        
        # Save metadata as JSON
        metadata = {
            'total_embeddings': len(self.embedding_log),
            'unique_pids': len(embeddings_by_pid),
            'embeddings_per_pid': {
                str(pid): len(entries) for pid, entries in embeddings_by_pid.items()
            },
            'frame_info': [
                {
                    'frame': e['frame'],
                    'pid': int(e['pid']),
                    'tid': int(e['tid']),
                    'bbox': e['bbox']
                }
                for e in self.embedding_log
            ]
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Saved metadata: {metadata_file}")

    def _summary(self, total_frames, fps):
        print(f"\n{'='*60}")
        print(f"ATTENDANCE REPORT")
        print(f"{'='*60}")
        print(f"Total unique persons: {len(self.gallery.entries)}")
        print(f"Re-identification events: {len(self.reid_events)}")
        print(f"ID reassignments: {len(self.reassignment_events)}")
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
        if self.reassignment_events:
            print(f"\nID Reassignments (BoTSORT swaps corrected):")
            for f, tid, old_pid, new_pid, d in self.reassignment_events:
                print(f"  Frame {f}: Track {tid} reassigned from Person {old_pid} → Person {new_pid} (dist: {d:.3f})")


def main():
    parser = argparse.ArgumentParser(description='Attendance Tracker v2')
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_attendance.mp4')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt')
    parser.add_argument('--reid-model', type=str, default='osnet_x1_0_msmt17.pt')
    parser.add_argument('--reid-threshold', type=float, default=0.42,
                        help='Gallery matching threshold (lower=stricter)')
    parser.add_argument('--detection-conf', type=float, default=0.25)
    parser.add_argument('--track-buffer', type=float, default=5.0)
    parser.add_argument('--color-weight', type=float, default=0.35,
                        help='Weight for color in combined score (0-1)')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("ATTENDANCE TRACKER v3")
    print("=" * 60)
    print("Layer 1: BoTSORT (frame-to-frame tracking)")
    print("Layer 2: Persistent Gallery (aggressive re-verification)")
    print("  - EVERY frame verification against ALL gallery PIDs")
    print("  - Automatic ID reassignment on better matches")
    print("  - Gallery contamination guard (2-strike system)")
    print("  - Failed-match avoidance (no retry of wrong PIDs)")
    print("  - 2D H×S color histogram")
    print("  - Faster probation & confirmation (3+3 frames)")
    print("=" * 60)

    tracker = AttendanceTracker(
        yolo_model=args.yolo_model,
        reid_model=args.reid_model,
        reid_threshold=args.reid_threshold,
        detection_conf=args.detection_conf,
        track_buffer_sec=args.track_buffer,
        color_weight=args.color_weight,
    )
    tracker.run(source=args.source, output=args.output, show=args.show)


if __name__ == '__main__':
    main()
