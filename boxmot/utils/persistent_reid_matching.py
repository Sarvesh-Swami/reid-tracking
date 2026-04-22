"""
Persistent ReID Matching Module
Extends NearestNeighborDistanceMetric to maintain a persistent gallery
"""
import os
import pickle

import numpy as np


class PersistentNearestNeighborDistanceMetric:
    """
    Enhanced distance metric that maintains features for ALL tracks (active + deleted)
    Enables re-identification and ID reassignment when persons reappear
    """
    
    def __init__(self, metric, matching_threshold, budget=None, persistent_budget=500):
        if metric == "euclidean":
            self._metric = self._nn_euclidean_distance
        elif metric == "cosine":
            self._metric = self._nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        
        self.matching_threshold = matching_threshold
        self.budget = budget  # Budget for active tracks
        self.persistent_budget = persistent_budget  # Budget for deleted tracks
        
        self.samples = {}  # Active track features
        self.persistent_samples = {}  # ALL track features (never deleted)
        self.deleted_ids = set()  # IDs that are currently inactive
        self.track_metadata = {}
        self.min_persistent_samples = 3
        self.robust_k = 5
        self.min_reid_confidence = 0.45
        self.min_reid_box_height = 80.0
        self.max_size_ratio = 1.8
        self.min_size_ratio = 0.55
        self.short_gap_frames = 15
        self.short_gap_spatial_factor = 3.0
        self.ambiguity_margin = 0.04
        
    def partial_fit(self, features, targets, active_targets):
        """
        Update the distance metric with new data.
        Unlike the original, this KEEPS features for deleted tracks.
        """
        print(f"  [Gallery Update] Adding {len(features)} features for targets: {targets}")
        print(f"  [Gallery Update] Active targets: {active_targets}")
        
        # Add features to both active and persistent galleries
        for feature, target in zip(features, targets):
            # Active gallery (for current tracking)
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
            
            # Persistent gallery (never deleted, for re-identification)
            self.persistent_samples.setdefault(target, []).append(feature)
            if self.persistent_budget is not None:
                self.persistent_samples[target] = self.persistent_samples[target][-self.persistent_budget:]
        
        # Identify deleted tracks
        all_known_ids = set(self.samples.keys())
        newly_deleted = all_known_ids - set(active_targets)
        
        if newly_deleted:
            print(f"  [Gallery Update] Newly deleted IDs: {newly_deleted}")
            for track_id in newly_deleted:
                num_features = len(self.persistent_samples.get(track_id, []))
                print(f"  [Gallery Update]    ID {track_id}: {num_features} features saved for future re-ID")
        
        self.deleted_ids.update(newly_deleted)
        
        # Remove from active samples (but keep in persistent!)
        self.samples = {k: self.samples[k] for k in active_targets}
        
        # Remove from deleted set if reactivated
        reactivated = self.deleted_ids & set(active_targets)
        if reactivated:
            print(f"  [Gallery Update] Reactivated IDs: {reactivated}")
        self.deleted_ids = self.deleted_ids - set(active_targets)
        
        print(f"  [Gallery Update] Current state: {len(self.samples)} active, {len(self.deleted_ids)} deleted, {len(self.persistent_samples)} total")
    
    def distance(self, features, targets):
        """
        Compute distance between features and targets (active tracks only)
        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            if target in self.samples:
                cost_matrix[i, :] = self._metric(self.samples[target], features)
            else:
                cost_matrix[i, :] = 1.0  # Max distance if no samples
        return cost_matrix
    
    def find_reid_matches(self, features, reid_threshold=None):
        """
        Find matches in the persistent gallery (including deleted tracks)
        Returns: list of (feature_idx, matched_id, distance) tuples
        """
        if reid_threshold is None:
            reid_threshold = self.matching_threshold
        
        candidates = []
        
        print(f"  [ReID Search] Searching {len(features)} features against {len(self.deleted_ids)} deleted IDs")
        print(f"  [ReID Search] Deleted IDs: {sorted(list(self.deleted_ids))}")
        print(f"  [ReID Search] Threshold: {reid_threshold}")
        
        for feat_idx, feature in enumerate(features):
            matched_id, distance = self.find_matching_deleted_id(
                feature,
                threshold=reid_threshold,
            )
            if matched_id is not None:
                candidates.append((feat_idx, matched_id, distance))
                print(f"  [ReID Search] Candidate accepted!")

        # Enforce a one-to-one mapping so the same deleted ID is not reused by
        # multiple detections in the same frame.
        matches = []
        used_features = set()
        used_ids = set()
        for feat_idx, matched_id, distance in sorted(candidates, key=lambda x: x[2]):
            if feat_idx in used_features or matched_id in used_ids:
                continue
            matches.append((feat_idx, matched_id, distance))
            used_features.add(feat_idx)
            used_ids.add(matched_id)

        print(f"  [ReID Search] Found {len(matches)} matches")
        return matches
    
    def find_matching_deleted_id(
        self,
        feature,
        detection_tlwh=None,
        detection_confidence=None,
        frame_idx=None,
        threshold=None,
    ):
        """
        Find a matching ID for a single feature from the deleted IDs.
        Used by the Tracker class for persistent re-identification.
        """
        if threshold is None:
            threshold = self.matching_threshold

        if len(self.deleted_ids) == 0:
            return None, 1.0

        if (
            detection_confidence is not None
            and detection_confidence < self.min_reid_confidence
        ):
            return None, 1.0

        if detection_tlwh is not None and detection_tlwh[3] < self.min_reid_box_height:
            return None, 1.0

        candidates = self._collect_reid_candidates(
            feature=feature,
            detection_tlwh=detection_tlwh,
            frame_idx=frame_idx,
        )
        if not candidates:
            return None, 1.0

        best = candidates[0]
        print(
            f"  [ReID Search] Best deleted-ID match = ID {best['track_id']}, "
            f"distance = {best['distance']:.3f} (threshold: {threshold})"
        )

        if best["distance"] >= threshold:
            return None, 1.0

        if len(candidates) > 1:
            second = candidates[1]
            if second["distance"] - best["distance"] < self.ambiguity_margin:
                print(
                    f"  [ReID Search] Rejecting ambiguous match: "
                    f"ID {best['track_id']} ({best['distance']:.3f}) vs "
                    f"ID {second['track_id']} ({second['distance']:.3f})"
                )
                return None, 1.0

        return best["track_id"], best["distance"]
        return None, 1.0
    
    def reactivate_id(self, track_id):
        """Reactivate a deleted ID"""
        if track_id in self.deleted_ids:
            self.deleted_ids.remove(track_id)
            # Restore to active samples
            if track_id in self.persistent_samples:
                self.samples[track_id] = self.persistent_samples[track_id].copy()

    def update_track_metadata(self, tracks, frame_idx):
        """Persist the latest geometry for confirmed active tracks."""
        for track in tracks:
            if not track.is_confirmed():
                continue

            tlwh = np.asarray(track.to_tlwh(), dtype=np.float32)
            meta = self.track_metadata.setdefault(
                int(track.track_id),
                {
                    "last_tlwh": tlwh,
                    "last_frame": frame_idx,
                    "mean_height": float(tlwh[3]),
                    "mean_width": float(tlwh[2]),
                    "updates": 0,
                },
            )
            meta["updates"] += 1
            alpha = 0.8
            meta["last_tlwh"] = tlwh
            meta["last_frame"] = frame_idx
            meta["mean_height"] = alpha * meta["mean_height"] + (1 - alpha) * float(tlwh[3])
            meta["mean_width"] = alpha * meta["mean_width"] + (1 - alpha) * float(tlwh[2])

    def total_unique_ids(self):
        """Return the number of stable identities stored in the gallery."""
        return len(self.persistent_samples)
    
    @staticmethod
    def _nn_euclidean_distance(x, y):
        """Euclidean distance"""
        distances = np.zeros((len(y),))
        for i, yi in enumerate(y):
            distances[i] = np.linalg.norm(np.asarray(x) - yi, axis=1).min()
        return distances
    
    @staticmethod
    def _nn_cosine_distance(x, y):
        """Cosine distance"""
        distances = np.zeros((len(y),))
        for i, yi in enumerate(y):
            # Normalize
            x_norm = np.asarray(x) / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
            yi_norm = yi / (np.linalg.norm(yi) + 1e-8)
            # Cosine similarity
            similarities = np.dot(x_norm, yi_norm)
            # Convert to distance
            distances[i] = 1.0 - similarities.max()
        return distances

    def _robust_track_distance(self, samples, feature):
        """Use several nearest samples instead of a single best sample.

        A pure min-distance rule is too optimistic and causes false matches
        when any one stored embedding happens to look similar.
        """
        samples = np.asarray(samples)
        feature = np.asarray(feature)

        if self._metric == self._nn_cosine_distance:
            samples_norm = samples / (np.linalg.norm(samples, axis=1, keepdims=True) + 1e-8)
            feature_norm = feature / (np.linalg.norm(feature) + 1e-8)
            distances = 1.0 - np.dot(samples_norm, feature_norm)
        else:
            distances = np.linalg.norm(samples - feature, axis=1)

        k = min(self.robust_k, len(distances))
        best_k = np.partition(distances, k - 1)[:k]
        return float(best_k.mean())

    def _collect_reid_candidates(self, feature, detection_tlwh=None, frame_idx=None):
        candidates = []
        for track_id in sorted(self.deleted_ids):
            if track_id not in self.persistent_samples:
                continue

            sample_count = len(self.persistent_samples[track_id])
            if sample_count < self.min_persistent_samples:
                continue

            meta = self.track_metadata.get(track_id)
            if detection_tlwh is not None and meta is not None:
                if not self._passes_geometry_gate(meta, detection_tlwh, frame_idx):
                    continue

            distance = self._robust_track_distance(
                self.persistent_samples[track_id], feature
            )
            candidates.append({"track_id": track_id, "distance": distance})

        candidates.sort(key=lambda item: item["distance"])
        return candidates

    def _passes_geometry_gate(self, meta, detection_tlwh, frame_idx):
        det_w = float(detection_tlwh[2])
        det_h = float(detection_tlwh[3])
        if det_w <= 1 or det_h <= 1:
            return False

        width_ratio = det_w / max(meta["mean_width"], 1.0)
        height_ratio = det_h / max(meta["mean_height"], 1.0)
        if not (self.min_size_ratio <= width_ratio <= self.max_size_ratio):
            return False
        if not (self.min_size_ratio <= height_ratio <= self.max_size_ratio):
            return False

        if frame_idx is None:
            return True

        frame_gap = frame_idx - meta.get("last_frame", frame_idx)
        if frame_gap <= self.short_gap_frames:
            prev = np.asarray(meta["last_tlwh"], dtype=np.float32)
            prev_center = prev[:2] + prev[2:] / 2
            det_center = np.asarray(detection_tlwh[:2], dtype=np.float32) + np.asarray(detection_tlwh[2:], dtype=np.float32) / 2
            center_distance = float(np.linalg.norm(det_center - prev_center))
            spatial_limit = self.short_gap_spatial_factor * max(det_h, prev[3], 1.0)
            if center_distance > spatial_limit:
                return False

        return True
    
    def save_gallery(self, path):
        """Save the persistent gallery to a file"""
        data = {
            'persistent_samples': self.persistent_samples,
            'deleted_ids': self.deleted_ids,
            'track_metadata': self.track_metadata,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  [Gallery Save] Saved {len(self.persistent_samples)} IDs to {path}")

    def load_gallery(self, path):
        """Load the persistent gallery from a file"""
        if not os.path.exists(path):
            print(f"  [Gallery Load] Warning: File {path} not found")
            return
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            loaded_samples = data.get('persistent_samples', {})
            self.persistent_samples.update(loaded_samples)
            self.track_metadata.update(data.get('track_metadata', {}))
            
            # When loading, we treat ALL loaded IDs as deleted initially 
            # so they can be re-identified in the current session
            for track_id in loaded_samples.keys():
                if track_id not in self.samples:
                    self.deleted_ids.add(track_id)
                
            print(f"  [Gallery Load] Loaded {len(loaded_samples)} IDs from {path}")
        except Exception as e:
            print(f"  [Gallery Load] Error loading gallery: {e}")

    def get_stats(self):
        """Get statistics about the gallery"""
        return {
            'active_ids': len(self.samples),
            'deleted_ids': len(self.deleted_ids),
            'total_ids_ever': len(self.persistent_samples),
            'total_features': sum(len(v) for v in self.persistent_samples.values()),
            'metadata_ids': len(self.track_metadata),
        }
