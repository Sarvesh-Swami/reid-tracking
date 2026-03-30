"""
Persistent ReID Matching Module
Extends NearestNeighborDistanceMetric to maintain a persistent gallery
"""
import numpy as np
from collections import defaultdict


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
            print(f"  [Gallery Update] 📝 Newly deleted IDs: {newly_deleted}")
            for track_id in newly_deleted:
                num_features = len(self.persistent_samples.get(track_id, []))
                print(f"  [Gallery Update]    ID {track_id}: {num_features} features saved for future re-ID")
        
        self.deleted_ids.update(newly_deleted)
        
        # Remove from active samples (but keep in persistent!)
        self.samples = {k: self.samples[k] for k in active_targets}
        
        # Remove from deleted set if reactivated
        reactivated = self.deleted_ids & set(active_targets)
        if reactivated:
            print(f"  [Gallery Update] ♻️ Reactivated IDs: {reactivated}")
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
        
        matches = []
        
        print(f"  [ReID Search] Searching {len(features)} features against {len(self.deleted_ids)} deleted IDs")
        print(f"  [ReID Search] Deleted IDs: {sorted(list(self.deleted_ids))}")
        print(f"  [ReID Search] Threshold: {reid_threshold}")
        
        for feat_idx, feature in enumerate(features):
            best_id = None
            best_distance = float('inf')
            
            # Search through ALL persistent samples (including deleted)
            for track_id in self.deleted_ids:
                if track_id not in self.persistent_samples:
                    print(f"  [ReID Search] Warning: ID {track_id} in deleted_ids but not in persistent_samples")
                    continue
                
                distance = np.min(self._metric(self.persistent_samples[track_id], [feature]))
                
                if distance < best_distance:
                    best_distance = distance
                    best_id = track_id
            
            if best_id is not None:
                print(f"  [ReID Search] Feature {feat_idx}: Best match = ID {best_id}, distance = {best_distance:.3f} (threshold: {reid_threshold})")
                if best_distance < reid_threshold:
                    matches.append((feat_idx, best_id, best_distance))
                    print(f"  [ReID Search] ✅ Match accepted!")
                else:
                    print(f"  [ReID Search] ❌ Match rejected (distance too high)")
        
        print(f"  [ReID Search] Found {len(matches)} matches")
        return matches
    
    def reactivate_id(self, track_id):
        """Reactivate a deleted ID"""
        if track_id in self.deleted_ids:
            self.deleted_ids.remove(track_id)
            # Restore to active samples
            if track_id in self.persistent_samples:
                self.samples[track_id] = self.persistent_samples[track_id].copy()
    
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
    
    def get_stats(self):
        """Get statistics about the gallery"""
        return {
            'active_ids': len(self.samples),
            'deleted_ids': len(self.deleted_ids),
            'total_ids_ever': len(self.persistent_samples),
            'total_features': sum(len(v) for v in self.persistent_samples.values())
        }
