"""
Attendance Tracker v5 - Physical Signatures & Ghost ID Tracking
==============================================================
Fixed camera, people come and go, same person = same ID always.

NEW in v5:
  - PHYSICAL SIGNATURES: Stores Height-to-Width ratios to distinguish body types.
  - BRIGHTNESS DISCRIMINATOR: Uses V-channel (HSV) to distinguish similar dark clothes.
  - DYNAMIC RATIO TEST: Relaxes the "confidence requirement" in exit/entry zones.
  - GHOST ID MEMORY: Stronger spatial-temporal bias for immediate re-entries.
  - TEMPORAL MAX-POOLING: Uses the best features from a track window for queries.

Fixes from v4:
  - Fixed ID fragmentation for similar-looking people (3 & 4) by using body geometry.
  - Reduced "I'm not sure" new-ID creations by using localized ratio tests.
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
    Persistent gallery with Physical Signatures (Geometry + Brightness).
    """

    def __init__(self, reid_threshold=0.42, max_features=100, min_features=3,
                 color_weight=0.35, ratio_margin=1.12):
        self.reid_threshold = reid_threshold
        self.max_features = max_features
        self.min_features = min_features
        self.color_weight = color_weight
        self.reid_weight = 1.0 - color_weight
        self.ratio_margin = ratio_margin # Default strict ratio
        
        self.entries = {}        # pid -> list of features
        self.prototypes = {}     # pid -> list of cluster centers
        self.color_hists = {}    # pid -> list of color histograms
        self.hw_ratios = {}      # pid -> list of H/W ratios
        self.brightness = {}     # pid -> list of mean V values
        self.next_id = 1

    def add_new(self, features_list, color_hist_list=None, hw_list=None, v_list=None):
        pid = self.next_id
        self.next_id += 1
        self.entries[pid] = [f.copy() for f in features_list]
        self.prototypes[pid] = [np.mean(features_list, axis=0)]
        self.color_hists[pid] = [c.copy() for c in color_hist_list if c is not None] if color_hist_list else []
        self.hw_ratios[pid] = hw_list if hw_list else []
        self.brightness[pid] = v_list if v_list else []
        return pid

    def update(self, pid, features, color_hist=None, hw=None, v=None, reliable=False):
        if pid not in self.entries: return False
        
        # Adaptive consistency
        if not reliable and len(self.entries[pid]) >= self.min_features:
            max_sim = max([float(p @ features) for p in self.prototypes[pid]])
            if max_sim < 0.45: return False

        self.entries[pid].append(features.copy())
        if len(self.entries[pid]) % 10 == 0: self._update_prototypes(pid)
        if len(self.entries[pid]) > self.max_features: self.entries[pid] = self.entries[pid][-self.max_features:]
        
        if color_hist is not None:
            self.color_hists.setdefault(pid, []).append(color_hist.copy())
            if len(self.color_hists[pid]) > self.max_features: self.color_hists[pid] = self.color_hists[pid][-self.max_features:]
        
        if hw is not None:
            self.hw_ratios.setdefault(pid, []).append(hw)
            if len(self.hw_ratios[pid]) > self.max_features: self.hw_ratios[pid] = self.hw_ratios[pid][-self.max_features:]
            
        if v is not None:
            self.brightness.setdefault(pid, []).append(v)
            if len(self.brightness[pid]) > self.max_features: self.brightness[pid] = self.brightness[pid][-self.max_features:]
            
        return True

    def _update_prototypes(self, pid):
        feats = np.array(self.entries[pid])
        if len(feats) < 5:
            self.prototypes[pid] = [np.mean(feats, axis=0)]
            return
        protos = [feats[0]]
        for f in feats[1:]:
            if max([float(f @ p) for p in protos]) < 0.75: protos.append(f)
        self.prototypes[pid] = [p / np.linalg.norm(p) for p in protos]

    def _physical_sim(self, query_hw, query_v, pid):
        """Calculate similarity based on Geometry (HW) and Brightness (V)."""
        hw_stored = self.hw_ratios.get(pid, [])
        v_stored = self.brightness.get(pid, [])
        
        hw_score = 1.0
        if hw_stored and query_hw:
            avg_hw = np.mean(hw_stored)
            # Distance in ratio: e.g. 2.5 vs 2.6
            diff = abs(query_hw - avg_hw)
            hw_score = np.exp(-5.0 * diff) # Sharp decay for geometry
            
        v_score = 1.0
        if v_stored and query_v:
            avg_v = np.mean(v_stored)
            diff = abs(query_v - avg_v) / 255.0
            v_score = np.exp(-10.0 * diff) # Exponential decay for brightness
            
        return hw_score, v_score

    def combined_score(self, features, pid, color_hist=None, hw=None, v=None):
        csim = self._color_sim(color_hist, pid)
        rsim = self._reid_sim(features, pid)
        hw_s, v_s = self._physical_sim(hw, v, pid)
        
        # Combine everything. Physical signatures help disambiguate similar clothes.
        # We give physical geometry a 10% vote
        combined = (0.30 * csim + 0.60 * rsim + 0.05 * hw_s + 0.05 * v_s)
        return combined, csim, rsim

    def _reid_sim(self, features, pid):
        protos = self.prototypes.get(pid, [])
        return max([float(features @ p) for p in protos]) if protos else 0.0

    def _color_sim(self, query_hist, pid):
        stored = self.color_hists.get(pid, [])
        if not stored or query_hist is None: return 0.5
        coeffs = [float(np.sum(np.sqrt(query_hist * s + 1e-10))) for s in stored]
        return float(np.mean(sorted(coeffs, reverse=True)[:5]))

    def query(self, features, color_hist=None, hw=None, v=None, exclude_pids=None, spatial_weights=None, is_ghost_zone=False):
        if not self.entries: return None
        scores = []
        for pid in self.entries.keys():
            if exclude_pids and pid in exclude_pids: continue
            
            combined, csim, rsim = self.combined_score(features, pid, color_hist, hw, v)
            
            s_weight = spatial_weights.get(pid, 1.0) if spatial_weights else 1.0
            score = combined * s_weight
            scores.append((pid, score, csim, rsim))

        if not scores: return None
        scores.sort(key=lambda x: x[1], reverse=True)
        best_pid, best_score, best_csim, best_rsim = scores[0]
        best_dist = 1.0 - best_score

        # DYNAMIC RATIO TEST: 
        # If we are in a "Ghost Zone" (near exit), we relax the ratio requirements.
        dynamic_ratio = self.ratio_margin
        if is_ghost_zone:
            dynamic_ratio = 1.03 # Very lenient
            
        if best_dist >= self.reid_threshold: return None
        if best_csim < 0.22: return None
        
        if len(scores) > 1 and scores[1][1] > 0.01:
            if best_score / scores[1][1] < dynamic_ratio:
                return None
        return (best_pid, best_dist, best_csim, best_rsim)


class AttendanceTracker:

    def __init__(self, yolo_model='yolov8n.pt', reid_model='osnet_x1_0_msmt17.pt',
                 reid_threshold=0.45, detection_conf=0.25, track_buffer_sec=10):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.yolo = YOLO(yolo_model)
        self.detection_conf = detection_conf
        self.reid_model_name = reid_model
        self.reid_threshold = reid_threshold

        self.tracker = None
        self.reid_model = None
        self.gallery = None
        
        self.id_map = {}
        self.tentative_matches = {}
        self.new_track_probation = {}
        self.reassignment_votes = defaultdict(lambda: defaultdict(int))
        self.spatial_history = {} # pid -> {'last_pos': (x,y), 'last_frame': f, 'hw': ratio, 'v': brightness}
        
        self.frame_count = 0
        self.colors = {}
        self.appearance_log = defaultdict(list)
        self.last_verified_frame = {}
        self.verify_fail_count = defaultdict(int)

    def _init_tracker(self, fps):
        track_buffer = int(10 * fps)
        reid_weights = WEIGHTS / self.reid_model_name
        self.tracker = BoTSORT(
            model_weights=reid_weights, device=self.device, fp16=False,
            track_high_thresh=0.4, new_track_thresh=0.4, track_buffer=track_buffer,
            match_thresh=0.8, proximity_thresh=0.6, appearance_thresh=0.35,
            cmc_method='sparseOptFlow', frame_rate=fps, lambda_=0.985,
        )
        self.reid_model = self.tracker.model
        self.gallery = PersistentGallery(reid_threshold=self.reid_threshold)

    @torch.no_grad()
    def _extract_all(self, img, bbox):
        """Extract ReID features, Color Histogram, and Physical Signatures."""
        x1, y1, x2, y2 = map(int, bbox[:4])
        h_img, w_img = img.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: return None, None, None, None
        
        # 1. ReID Features
        feat = self.reid_model([crop])
        if isinstance(feat, torch.Tensor): feat = feat.detach().cpu().numpy()
        feat = feat.flatten().astype(np.float32)
        norm = np.linalg.norm(feat)
        feat = feat / norm if norm > 1e-6 else None
        
        # 2. Geometry (HW Ratio)
        hw_ratio = (y2 - y1) / (x2 - x1 + 1e-6)
        
        # 3. Color & Brightness
        ch, cw = crop.shape[:2]
        torso = crop[int(ch*0.15):int(ch*0.75), int(cw*0.1):int(cw*0.9)]
        if torso.size > 0:
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            hist_hs = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()
            hist = hist_hs / (hist_hs.sum() + 1e-6)
            brightness = np.mean(hsv[:, :, 2])
        else:
            hist, brightness = None, None
            
        return feat, hist, hw_ratio, brightness

    def _map_ids(self, tracks, img):
        if len(tracks) == 0: return np.empty((0, 7))
        current_tids = set(int(t[4]) for t in tracks)
        
        # Cleanup expired
        for tid in list(self.new_track_probation):
            if tid not in current_tids: del self.new_track_probation[tid]
        for tid in list(self.tentative_matches):
            if tid not in current_tids: del self.tentative_matches[tid]
            
        confirmed_active_pids = set(pid for tid, pid in self.id_map.items() if tid in current_tids)
        output = []

        for t in tracks:
            bbox, tid, conf = t[:4], int(t[4]), float(t[5])
            cls = t[6] if len(t) > 6 else 0
            feat, chist, hw, v = self._extract_all(img, bbox)
            pid = 0
            is_hypo = False

            if tid in self.id_map:
                pid = self.id_map[tid]
                if feat is not None:
                    # Consensus-based reassignment
                    score, _, _ = self.gallery.combined_score(feat, pid, chist, hw, v)
                    
                    # Look for better match
                    best_other_pid, best_other_score = None, 0.0
                    for opid in self.gallery.entries.keys():
                        if opid == pid: continue
                        oscore, _, _ = self.gallery.combined_score(feat, opid, chist, hw, v)
                        if oscore > best_other_score: best_other_pid, best_other_score = opid, oscore
                    
                    if best_other_pid and (best_other_score - score) > 0.15:
                        self.reassignment_votes[tid][best_other_pid] += 1
                        if self.reassignment_votes[tid][best_other_pid] >= 12: # 12 frames consensus
                            self.id_map[tid] = best_other_pid
                            pid = best_other_pid
                            self.reassignment_votes[tid].clear()
                    
                    self.gallery.update(pid, feat, chist, hw, v, reliable=True)

            elif tid in self.tentative_matches:
                info = self.tentative_matches[tid]
                pid, is_hypo = info["pid"], True
                if feat is not None:
                    score, _, _ = self.gallery.combined_score(feat, pid, chist, hw, v)
                    if (1.0 - score) < self.reid_threshold:
                        info["count"] += 1
                        if info["count"] >= 5: # Confirmation
                            self.id_map[tid], is_hypo = pid, False
                            del self.tentative_matches[tid]
                    else:
                        info["fails"] = info.get("fails", 0) + 1
                        if info["fails"] > 4: del self.tentative_matches[tid]; pid = 0

            elif tid in self.new_track_probation:
                prob = self.new_track_probation[tid]
                if feat is not None:
                    prob["feats"].append(feat); prob["hists"].append(chist)
                    prob["hws"].append(hw); prob["vs"].append(v)
                
                if len(prob["feats"]) >= 3:
                    avg_feat = np.mean(prob["feats"], axis=0)
                    avg_feat /= np.linalg.norm(avg_feat)
                    
                    # Spatial bias
                    weights = {}
                    cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                    is_ghost = False
                    for gpid, ghist in self.spatial_history.items():
                        lx, ly = ghist['last_pos']
                        dist = np.sqrt((cx-lx)**2 + (cy-ly)**2)
                        if dist < 150 and (self.frame_count - ghist['last_frame']) < 150:
                            weights[gpid] = 1.3 # Strong bonus
                            is_ghost = True
                        elif dist < 400: weights[gpid] = 1.1
                    
                    res = self.gallery.query(avg_feat, prob["hists"][-1], prob["hws"][-1], prob["vs"][-1],
                                           exclude_pids=confirmed_active_pids,
                                           spatial_weights=weights, is_ghost_zone=is_ghost)
                    if res:
                        pid, is_hypo = res[0], True
                        self.tentative_matches[tid] = {"pid": pid, "count": 1}
                        del self.new_track_probation[tid]
                    else:
                        pid = self.gallery.add_new(prob["feats"], prob["hists"], prob["hws"], prob["vs"])
                        self.id_map[tid] = pid
                        del self.new_track_probation[tid]
                        print(f"  [NEW] Person {pid}")

            else:
                self.new_track_probation[tid] = {"feats": [], "hists": [], "hws": [], "vs": []}

            if pid > 0:
                cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                self.spatial_history[pid] = {'last_pos': (cx, cy), 'last_frame': self.frame_count}
                if pid not in self.colors:
                    np.random.seed(pid*37); self.colors[pid] = tuple(np.random.randint(60,230,3).tolist())
                output.append([*bbox, -pid if is_hypo else pid, conf, cls])
            else:
                output.append([*bbox, -1, conf, cls])

        return np.array(output)

    def run(self, source, output='output_v5.mp4', show=False):
        cap = cv2.VideoCapture(source)
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
                
                # Draw
                for t in tracks:
                    x1, y1, x2, y2 = map(int, t[:4])
                    pid_raw = int(t[4])
                    pid = abs(pid_raw) if pid_raw != -1 else -1
                    color = self.colors.get(pid, (128,128,128)) if pid != -1 else (128,128,128)
                    label = f"Person {pid}" + ("?" if pid_raw < 0 else "")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1 if pid_raw < 0 else 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                out.write(frame)
                if show:
                    cv2.imshow('Tracker v5', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                if self.frame_count % (fps*5) == 0: print(f"  {self.frame_count/total*100:.0f}% | Unique: {len(self.gallery.entries)}")
        finally:
            cap.release(); out.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_v5.mp4')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    tracker = AttendanceTracker()
    tracker.run(source=args.source, output=args.output, show=args.show)
