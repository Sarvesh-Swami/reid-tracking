"""
Attendance Tracker v4 - Identity Consensus & Spatial-Temporal Reasoning
======================================================================
Fixed camera, people come and go, same person = same ID always.

NEW in v4:
  - VIEWPOINT CLUSTERING: Stores multiple appearance prototypes (Front/Back/Side)
  - SPATIAL-TEMPORAL REASONING: Factors in last known location and time
  - IDENTITY CONSENSUS (STICKY IDs): Requires multiple frames of evidence to swap IDs
  - ADAPTIVE LEARNING: Relaxed consistency guard when tracker continuity is high
  - HYPOTHESIS DISPLAY: Shows tentative IDs immediately to prevent flickering

Fixes from v3:
  - Resolved "Consistency Guard Catch-22" preventing back-profile learning
  - Reduced ID hopping by adding reassignment voting
  - Improved return matching by using spatial exit/entry zones
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
    Persistent person gallery with Viewpoint Clustering and Adaptive Learning.
    """

    def __init__(self, reid_threshold=0.42, max_features=100, min_features=3,
                 color_weight=0.35, ratio_margin=1.10, consistency_sim=0.45):
        self.reid_threshold = reid_threshold
        self.max_features = max_features
        self.min_features = min_features
        self.color_weight = color_weight
        self.reid_weight = 1.0 - color_weight
        self.ratio_margin = ratio_margin
        self.consistency_sim = consistency_sim
        
        self.entries = {}        # pid -> list of features
        self.prototypes = {}     # pid -> list of cluster centers (Front, Back, etc)
        self.color_hists = {}    # pid -> list of color histograms
        self.next_id = 1

    def add_new(self, features_list, color_hist_list=None):
        pid = self.next_id
        self.next_id += 1
        self.entries[pid] = [f.copy() for f in features_list]
        self.prototypes[pid] = [np.mean(features_list, axis=0)]
        if color_hist_list:
            self.color_hists[pid] = [c.copy() for c in color_hist_list if c is not None]
        else:
            self.color_hists[pid] = []
        return pid

    def update(self, pid, features, color_hist=None, reliable=False):
        """
        Update gallery. 
        If reliable=True (tracker continuity), we relax the consistency guard.
        """
        if pid not in self.entries:
            return False
            
        # Feature consistency guard
        if not reliable and len(self.entries[pid]) >= self.min_features:
            # Check against ALL prototypes (not just average)
            max_sim = 0
            for proto in self.prototypes[pid]:
                sim = float(proto @ features)
                max_sim = max(max_sim, sim)
            
            if max_sim < self.consistency_sim:
                # This might be a new viewpoint (e.g. back profile)
                # We only allow it if reliable=True
                return False

        self.entries[pid].append(features.copy())
        
        # Periodic prototype update (clustering)
        if len(self.entries[pid]) % 10 == 0:
            self._update_prototypes(pid)
            
        if len(self.entries[pid]) > self.max_features:
            self.entries[pid] = self.entries[pid][-self.max_features:]
            
        if color_hist is not None:
            self.color_hists.setdefault(pid, []).append(color_hist.copy())
            if len(self.color_hists[pid]) > self.max_features:
                self.color_hists[pid] = self.color_hists[pid][-self.max_features:]
        return True

    def _update_prototypes(self, pid):
        """Simple clustering to find major viewpoints (Front, Back, etc)"""
        feats = np.array(self.entries[pid])
        if len(feats) < 5:
            self.prototypes[pid] = [np.mean(feats, axis=0)]
            return

        # Simple greedy clustering
        protos = [feats[0]]
        for f in feats[1:]:
            sims = [float(f @ p) for p in protos]
            if max(sims) < 0.75: # New cluster if similarity is low
                protos.append(f)
        
        # Refine clusters by averaging members
        refined = []
        for p in protos:
            members = [f for f in feats if (f @ p) > 0.70]
            if members:
                center = np.mean(members, axis=0)
                refined.append(center / np.linalg.norm(center))
        
        self.prototypes[pid] = refined

    def _color_sim(self, query_hist, pid):
        stored = self.color_hists.get(pid, [])
        if not stored or query_hist is None:
            return 0.5
        coeffs = [float(np.sum(np.sqrt(query_hist * s + 1e-10))) for s in stored]
        coeffs.sort(reverse=True)
        return float(np.mean(coeffs[:min(5, len(coeffs))]))

    def _reid_sim(self, features, pid):
        protos = self.prototypes.get(pid, [])
        if not protos:
            return 0.0
        # Match against the BEST prototype (Max-of-Max)
        sims = [float(features @ p) for p in protos]
        return max(sims)

    def combined_score(self, features, pid, color_hist=None):
        csim = self._color_sim(color_hist, pid)
        rsim = self._reid_sim(features, pid)
        combined = self.color_weight * csim + self.reid_weight * rsim
        return combined, csim, rsim

    def query(self, features, color_hist=None, exclude_pids=None, spatial_weights=None):
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
            
            # Apply spatial weight if provided
            s_weight = 1.0
            if spatial_weights and pid in spatial_weights:
                s_weight = spatial_weights[pid]
            
            combined = (self.color_weight * csim + self.reid_weight * rsim) * s_weight
            scores.append((pid, combined, csim, rsim))

        if not scores:
            return None
            
        scores.sort(key=lambda x: x[1], reverse=True)
        best_pid, best_score, best_csim, best_rsim = scores[0]
        best_dist = 1.0 - best_score

        if best_dist >= self.reid_threshold:
            return None
        if best_csim < 0.25: # Relaxed from 0.30
            return None
            
        # Ratio test
        if len(scores) > 1 and scores[1][1] > 0.01:
            if best_score / scores[1][1] < self.ratio_margin:
                return None
        return (best_pid, best_dist, best_csim, best_rsim)


class AttendanceTracker:

    def __init__(self, yolo_model='yolov8n.pt', reid_model='osnet_x1_0_msmt17.pt',
                 reid_threshold=0.42, detection_conf=0.25, track_buffer_sec=10,
                 color_weight=0.35):
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
        
        # Sticky Identity / Voting
        self.reassignment_votes = defaultdict(lambda: defaultdict(int)) # tid -> pid -> count
        self.verify_fail_count = defaultdict(int)
        self.last_verified_frame = {}
        
        # Spatial-Temporal Memory
        self.spatial_history = {} # pid -> {'last_pos': (x,y), 'last_frame': f}
        
        # Tuning
        self.confirmation_frames = 5   
        self.min_new_frames = 3        
        self.verify_interval = 2       
        self.max_verify_fails = 4      
        self.reassignment_threshold_frames = 10 # Need 10 frames of better match to swap
        self.reassignment_score_diff = 0.12     # Score difference required for a "vote"

        self.colors = {}
        self.appearance_log = defaultdict(list)
        self.reid_events = []
        self.reassignment_events = []
        self.frame_count = 0
        self.embedding_log = []

    def _init_tracker(self, fps):
        track_buffer = int(self.track_buffer_sec * fps)
        reid_weights = WEIGHTS / self.reid_model_name
        self.tracker = BoTSORT(
            model_weights=reid_weights, device=self.device, fp16=False,
            track_high_thresh=0.4, new_track_thresh=0.4, track_buffer=track_buffer,
            match_thresh=0.8, proximity_thresh=0.6, appearance_thresh=0.35, # Strict BoTSORT
            cmc_method='sparseOptFlow', frame_rate=fps, lambda_=0.985,
        )
        self.reid_model = self.tracker.model
        self.gallery = PersistentGallery(
            reid_threshold=self.reid_threshold, color_weight=self.color_weight,
        )

    @torch.no_grad()
    def _extract_features(self, img, bbox):
        x1, y1, x2, y2 = map(int, bbox[:4])
        h, w = img.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if (x2 - x1) < 20 or (y2 - y1) < 40: return None
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: return None
        features = self.reid_model([crop])
        if isinstance(features, torch.Tensor): features = features.cpu().numpy()
        features = features.flatten().astype(np.float32)
        norm = np.linalg.norm(features)
        return features / norm if norm > 1e-6 else None

    def _extract_color_histogram(self, img, bbox):
        x1, y1, x2, y2 = map(int, bbox[:4])
        h, w = img.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: return None
        ch, cw = crop.shape[:2]
        torso = crop[int(ch*0.15):int(ch*0.75), int(cw*0.1):int(cw*0.9)]
        if torso.size == 0: return None
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist_hs = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        hist = np.concatenate([hist_hs, hist_v]).astype(np.float32)
        total = hist.sum()
        return hist / total if total > 1e-6 else None

    def _get_spatial_weights(self, current_bbox):
        """Calculate score multipliers based on distance from last seen location."""
        weights = {}
        cx = (current_bbox[0] + current_bbox[2]) / 2
        cy = (current_bbox[1] + current_bbox[3]) / 2
        
        for pid, history in self.spatial_history.items():
            lx, ly = history['last_pos']
            dt = self.frame_count - history['last_frame']
            
            # Simple Euclidean distance
            dist = np.sqrt((cx - lx)**2 + (cy - ly)**2)
            
            # Decay weight over distance and time
            spatial_bonus = 1.0
            if dt < 300: # 10 seconds
                if dist < 200: # Close
                    spatial_bonus = 1.25
                elif dist > 800: # Very far
                    spatial_bonus = 0.85
            weights[pid] = spatial_bonus
        return weights

    def _unmap_track(self, tid, feat, chist):
        if tid in self.id_map:
            old_pid = self.id_map[tid]
            del self.id_map[tid]
            print(f"    [UNMAP] Track {tid} unmapped from Person {old_pid}")
        self.reassignment_votes[tid].clear()
        self.new_track_probation[tid] = {
            "features": [feat] if feat is not None else [],
            "color_hists": [chist] if chist is not None else [],
            "start_frame": self.frame_count,
            "failed_pids": set(),
        }

    def _map_ids(self, tracks, img):
        if len(tracks) == 0: return np.empty((0, 7))

        current_tids = set(int(t[4]) for t in tracks)
        # Cleanup
        for tid in list(self.tentative_matches): 
            if tid not in current_tids: del self.tentative_matches[tid]
        for tid in list(self.new_track_probation):
            if tid not in current_tids: del self.new_track_probation[tid]
        for tid in list(self.reassignment_votes):
            if tid not in current_tids: del self.reassignment_votes[tid]

        confirmed_active_pids = set(pid for tid, pid in self.id_map.items() if tid in current_tids)
        
        output = []
        for t in tracks:
            bbox, tid, conf = t[:4], int(t[4]), float(t[5])
            cls = t[6] if len(t) > 6 else 0
            pid = 0
            is_hypothetical = False

            feat = self._extract_features(img, bbox)
            chist = self._extract_color_histogram(img, bbox)

            if tid in self.id_map:
                # === KNOWN TRACK ===
                pid = self.id_map[tid]
                if feat is not None:
                    # Adaptive Verification & Learning
                    should_verify = (self.frame_count - self.last_verified_frame.get(tid, 0)) >= self.verify_interval
                    if should_verify:
                        self.last_verified_frame[tid] = self.frame_count
                        
                        # Compare against assigned PID
                        score, csim, rsim = self.gallery.combined_score(feat, pid, chist)
                        
                        # Compare against ALL other PIDs
                        best_other_pid = None
                        best_other_score = 0.0
                        for other_pid in self.gallery.entries.keys():
                            if other_pid == pid: continue
                            oscore, _, _ = self.gallery.combined_score(feat, other_pid, chist)
                            if oscore > best_other_score:
                                best_other_score = oscore
                                best_other_pid = other_pid
                        
                        # Voting Logic (Sticky ID)
                        if best_other_pid is not None and (best_other_score - score) > self.reassignment_score_diff:
                            self.reassignment_votes[tid][best_other_pid] += 1
                            if self.reassignment_votes[tid][best_other_pid] >= self.reassignment_threshold_frames:
                                # CONSENSUS REACHED: Reassign
                                old_pid = pid
                                pid = best_other_pid
                                self.id_map[tid] = pid
                                print(f"  [REASSIGNED] Frame {self.frame_count}: Track {tid} {old_pid} -> {pid} (Consensus)")
                                self.reassignment_votes[tid].clear()
                        else:
                            # Reset votes if evidence is not consistent
                            for p in list(self.reassignment_votes[tid]):
                                self.reassignment_votes[tid][p] = max(0, self.reassignment_votes[tid][p] - 1)

                        # Identity Lost Logic
                        if (1.0 - score) >= self.reid_threshold:
                            self.verify_fail_count[tid] += 1
                            if self.verify_fail_count[tid] >= self.max_verify_fails:
                                print(f"  [LOST] Frame {self.frame_count}: Track {tid} lost Person {pid}")
                                self._unmap_track(tid, feat, chist)
                                pid = 0
                        else:
                            self.verify_fail_count[tid] = 0
                            # Update gallery (Reliable=True because BoTSORT kept the ID)
                            self.gallery.update(pid, feat, chist, reliable=True)

            elif tid in self.tentative_matches:
                # === TENTATIVE MATCH ===
                info = self.tentative_matches[tid]
                pid = info["pid"]
                is_hypothetical = True
                if feat is not None:
                    score, csim, rsim = self.gallery.combined_score(feat, pid, chist)
                    if (1.0 - score) < self.reid_threshold:
                        info["count"] += 1
                        if info["count"] >= self.confirmation_frames:
                            self.id_map[tid] = pid
                            del self.tentative_matches[tid]
                            is_hypothetical = False
                            print(f"  [CONFIRMED] Frame {self.frame_count}: Person {pid} returned")
                    else:
                        info["failures"] += 1
                        if info["failures"] >= 3:
                            self._unmap_track(tid, feat, chist)
                            pid = 0

            elif tid in self.new_track_probation:
                # === PROBATION ===
                prob = self.new_track_probation[tid]
                if feat is not None:
                    prob["features"].append(feat)
                    prob["color_hists"].append(chist)
                
                if len(prob["features"]) >= self.min_new_frames:
                    avg_feat = np.mean(prob["features"], axis=0)
                    avg_feat /= np.linalg.norm(avg_feat)
                    
                    # Use spatial weights to help matching
                    s_weights = self._get_spatial_weights(bbox)
                    res = self.gallery.query(avg_feat, prob["color_hists"][-1], 
                                           exclude_pids=confirmed_active_pids,
                                           spatial_weights=s_weights)
                    
                    if res:
                        mpid, dist, csim, rsim = res
                        self.tentative_matches[tid] = {"pid": mpid, "count": 1, "failures": 0}
                        pid = mpid
                        is_hypothetical = True
                        del self.new_track_probation[tid]
                    else:
                        pid = self.gallery.add_new(prob["features"], prob["color_hists"])
                        self.id_map[tid] = pid
                        del self.new_track_probation[tid]
                        print(f"  [NEW] Frame {self.frame_count}: Person {pid}")

            else:
                # === NEW TRACK ===
                self.new_track_probation[tid] = {"features": [feat] if feat is not None else [], 
                                                "color_hists": [chist] if chist is not None else [],
                                                "start_frame": self.frame_count}

            if pid > 0:
                # Update spatial history
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                self.spatial_history[pid] = {'last_pos': (cx, cy), 'last_frame': self.frame_count}
                
                self.appearance_log[pid].append(self.frame_count)
                if pid not in self.colors:
                    np.random.seed(pid * 37)
                    self.colors[pid] = tuple(np.random.randint(60, 230, 3).tolist())
                
                # Hypothesis label
                label_pid = pid
                if is_hypothetical:
                    label_pid = -pid # Use negative to flag for drawing
                output.append([bbox[0], bbox[1], bbox[2], bbox[3], label_pid, conf, cls])
            else:
                output.append([bbox[0], bbox[1], bbox[2], bbox[3], -1, conf, cls])

        return np.array(output) if output else np.empty((0, 7))

    def _draw(self, frame, tracks, total_frames):
        for t in tracks:
            x1, y1, x2, y2 = map(int, t[:4])
            pid_raw = int(t[4])
            is_hypo = pid_raw < 0
            pid = abs(pid_raw) if pid_raw != -1 else -1
            
            if pid == -1:
                color, label = (128, 128, 128), "?"
            else:
                color = self.colors.get(pid, (200, 200, 200))
                label = f"Person {pid}" + ("?" if is_hypo else "")
            
            if is_hypo:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1) # Thin box for hypo
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self, source, output='output_v4.mp4', show=False):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened(): return
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._init_tracker(fps)
        out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                self.frame_count += 1
                
                results = self.yolo(frame, conf=self.detection_conf, classes=[0], verbose=False)[0]
                dets = np.array([[*b.xyxy[0].cpu().numpy(), float(b.conf[0]), float(b.cls[0])] for b in results.boxes]) if len(results.boxes) > 0 else np.empty((0, 6))
                
                raw_tracks = self.tracker.update(dets, frame)
                tracks = self._map_ids(raw_tracks, frame)
                self._draw(frame, tracks, total)
                out.write(frame)
                
                if show:
                    cv2.imshow('Attendance Tracker v4', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                if self.frame_count % (fps * 5) == 0:
                    print(f"  {self.frame_count/total*100:.0f}% | Unique: {len(self.gallery.entries)}")
        finally:
            cap.release(); out.release(); cv2.destroyAllWindows()
            print(f"\n[DONE] Saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_v4.mp4')
    parser.add_argument('--reid-threshold', type=float, default=0.42)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    tracker = AttendanceTracker(reid_threshold=args.reid_threshold)
    tracker.run(source=args.source, output=args.output, show=args.show)

if __name__ == '__main__':
    main()
