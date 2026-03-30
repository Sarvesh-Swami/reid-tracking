"""
DeepOCSORT with Persistent Re-Identification
Maintains a gallery of all seen persons to reassign IDs when they reappear
"""
import numpy as np
from collections import defaultdict
from ..deepocsort.deep_ocsort import DeepOCSort, KalmanBoxTracker


class PersistentGallery:
    """
    Maintains a persistent gallery of ReID features for all tracks ever seen.
    Allows re-identification and ID reassignment when persons reappear.
    """
    def __init__(self, max_features_per_id=100, reid_threshold=0.5):
        self.gallery = defaultdict(list)  # {track_id: [features]}
        self.max_features_per_id = max_features_per_id
        self.reid_threshold = reid_threshold
        self.deleted_ids = set()  # Track IDs that have been deleted
        
    def add_features(self, track_id, features):
        """Add features for a track ID"""
        if isinstance(features, list):
            self.gallery[track_id].extend(features)
        else:
            self.gallery[track_id].append(features)
        
        # Keep only the most recent features
        if len(self.gallery[track_id]) > self.max_features_per_id:
            self.gallery[track_id] = self.gallery[track_id][-self.max_features_per_id:]
    
    def mark_deleted(self, track_id):
        """Mark a track as deleted (available for reassignment)"""
        self.deleted_ids.add(track_id)
    
    def find_best_match(self, feature):
        """
        Find the best matching deleted track ID for a new detection
        Returns: (best_id, distance) or (None, None) if no good match
        """
        if not self.deleted_ids or not self.gallery:
            return None, None
        
        best_id = None
        best_distance = float('inf')
        
        # Normalize query feature
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        
        # Search through deleted tracks
        for track_id in self.deleted_ids:
            if track_id not in self.gallery:
                continue
            
            # Get all features for this ID
            gallery_features = np.array(self.gallery[track_id])
            
            # Compute cosine distance to all features
            # Normalize gallery features
            norms = np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-8
            gallery_features_norm = gallery_features / norms
            
            # Cosine similarity
            similarities = np.dot(gallery_features_norm, feature)
            max_similarity = np.max(similarities)
            distance = 1.0 - max_similarity  # Convert to distance
            
            if distance < best_distance:
                best_distance = distance
                best_id = track_id
        
        # Only return if below threshold
        if best_distance < self.reid_threshold:
            return best_id, best_distance
        
        return None, None
    
    def reactivate_id(self, track_id):
        """Remove ID from deleted set (it's active again)"""
        self.deleted_ids.discard(track_id)
    
    def get_all_ids(self):
        """Get all track IDs ever seen"""
        return set(self.gallery.keys())


class PersistentDeepOCSort(DeepOCSort):
    """
    DeepOCSORT with persistent re-identification capability.
    Maintains a gallery of all seen persons and reassigns IDs when they reappear.
    """
    
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        per_class=False,
        det_thresh=0.0,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        use_byte=False,
        # New parameters for persistent ReID
        enable_persistent_reid=True,
        reid_threshold=0.5,  # Lower = stricter matching
        max_gallery_size=100,  # Features per ID
    ):
        super().__init__(
            model_weights=model_weights,
            device=device,
            fp16=fp16,
            per_class=per_class,
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            delta_t=delta_t,
            asso_func=asso_func,
            inertia=inertia,
            use_byte=use_byte,
        )
        
        self.enable_persistent_reid = enable_persistent_reid
        self.persistent_gallery = PersistentGallery(
            max_features_per_id=max_gallery_size,
            reid_threshold=reid_threshold
        )
        
        # Track the maximum ID ever used
        self.max_id_ever = 0
    
    def update(self, dets, img, tag='persistent'):
        """
        Update with persistent re-identification
        """
        # Store features for active tracks before update
        if self.enable_persistent_reid:
            for trk in self.trackers:
                if hasattr(trk, 'emb') and trk.emb is not None:
                    self.persistent_gallery.add_features(trk.id, trk.emb)
        
        # Get detections and extract features
        if dets.shape[0] > 0:
            bboxes = dets[:, :4]
            scores = dets[:, 4]
            classes = dets[:, 5]
            
            # Extract ReID features for all detections
            if not self.embedding_off:
                features = self._get_features(bboxes, img)
            else:
                features = None
        else:
            features = None
        
        # Call parent update
        outputs = super().update(dets, img, tag)
        
        # After update, check for deleted tracks and mark them
        if self.enable_persistent_reid:
            current_ids = set([int(out[4]) for out in outputs]) if len(outputs) > 0 else set()
            all_active_ids = set([trk.id for trk in self.trackers])
            
            # Update max_id_ever
            if len(all_active_ids) > 0:
                self.max_id_ever = max(self.max_id_ever, max(all_active_ids))
        
        return outputs
    
    def _initiate_track_with_reid(self, bbox, cls, score, emb):
        """
        Initiate a new track, but first check if it matches a deleted track
        """
        if not self.enable_persistent_reid or emb is None:
            # Normal track creation
            return self._create_new_track(bbox, cls, score, emb)
        
        # Try to find a match in the persistent gallery
        matched_id, distance = self.persistent_gallery.find_best_match(emb)
        
        if matched_id is not None:
            # Reuse the old ID
            print(f"🔄 Re-identified person! Reassigning ID {matched_id} (distance: {distance:.3f})")
            self.persistent_gallery.reactivate_id(matched_id)
            
            # Create track with the old ID
            trk = KalmanBoxTracker(
                bbox=bbox,
                cls=cls,
                delta_t=self.delta_t,
                emb=emb,
                alpha=self.alpha_fixed_emb,
                new_kf=not self.new_kf_off
            )
            trk.id = matched_id  # Assign the old ID
            trk.conf = score
            return trk
        else:
            # Create new track with new ID
            return self._create_new_track(bbox, cls, score, emb)
    
    def _create_new_track(self, bbox, cls, score, emb):
        """Create a completely new track with a new ID"""
        trk = KalmanBoxTracker(
            bbox=bbox,
            cls=cls,
            delta_t=self.delta_t,
            emb=emb,
            alpha=self.alpha_fixed_emb,
            new_kf=not self.new_kf_off
        )
        trk.conf = score
        
        # Assign new ID
        self.max_id_ever += 1
        trk.id = self.max_id_ever
        
        return trk
    
    def _mark_track_deleted(self, track_id):
        """Mark a track as deleted in the persistent gallery"""
        if self.enable_persistent_reid:
            self.persistent_gallery.mark_deleted(track_id)
            print(f"📝 Track ID {track_id} deleted, features saved for future re-identification")
