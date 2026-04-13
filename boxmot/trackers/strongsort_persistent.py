"""
StrongSORT with Persistent Re-Identification
Maintains a gallery of all seen persons to reassign IDs when they reappear
"""
import numpy as np
import torch
import cv2

from ..utils.persistent_reid_matching import PersistentNearestNeighborDistanceMetric
from .strongsort.sort.detection import Detection
from .strongsort.sort.tracker import Tracker
from ..appearance.reid_multibackend import ReIDDetectMultiBackend
from ..utils.ops import xyxy2xywh


class PersistentTracker(Tracker):
    """
    Extended Tracker with persistent re-identification
    """
    
    def __init__(self, *args, enable_reid=True, reid_threshold=0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_reid = enable_reid
        self.reid_threshold = reid_threshold
        self.next_new_id = 1  # Track the next ID to assign
        self.frame_count = 0  # Track frame count for stats
        
    def sync_next_id(self):
        """Update next_new_id based on features in the metric gallery"""
        if hasattr(self.metric, 'persistent_samples') and self.metric.persistent_samples:
            max_id = max(self.metric.persistent_samples.keys())
            if max_id >= self.next_new_id:
                self.next_new_id = max_id + 1
                print(f"  [Tracker] Synced next_new_id to {self.next_new_id}")
        
    def update(self, detections, classes, confidences):
        """
        Enhanced update with persistent re-identification
        """
        self.frame_count += 1
        
        # First, try to match with deleted tracks using ReID
        if self.enable_reid and isinstance(self.metric, PersistentNearestNeighborDistanceMetric):
            if len(detections) > 0:
                features = [det.feature for det in detections]
                reid_matches = self.metric.find_reid_matches(features, self.reid_threshold)
                
                print(f"[Frame {self.frame_count}] Checking {len(detections)} detections against {len(self.metric.deleted_ids)} deleted IDs")
                
                # Process ReID matches
                matched_indices = set()
                for feat_idx, matched_id, distance in reid_matches:
                    print(f"Re-identified! Reassigning ID {matched_id} (distance: {distance:.3f})")
                    
                    # Reactivate the old ID
                    self.metric.reactivate_id(matched_id)
                    
                    # Create a new track with the old ID
                    detection = detections[feat_idx]
                    class_id = classes[feat_idx].item()
                    conf = confidences[feat_idx].item()
                    
                    # Import Track class
                    from .strongsort.sort.track import Track, TrackState
                    
                    new_track = Track(
                        detection.to_xyah(),
                        matched_id,  # Use the old ID!
                        class_id,
                        conf,
                        self.n_init,
                        self.max_age,
                        self.ema_alpha,
                        detection.feature,
                    )
                    
                    # Mark as confirmed immediately (it's a re-identification)
                    new_track.state = TrackState.Confirmed
                    new_track.hits = self.n_init  # Ensure it's confirmed
                    
                    self.tracks.append(new_track)
                    matched_indices.add(feat_idx)
                    
                    # Update next_new_id if necessary
                    if matched_id >= self.next_new_id:
                        self.next_new_id = matched_id + 1
                
                # Filter out matched detections
                if matched_indices:
                    detections = [det for i, det in enumerate(detections) if i not in matched_indices]
                    classes = classes[[i for i in range(len(classes)) if i not in matched_indices]]
                    confidences = confidences[[i for i in range(len(confidences)) if i not in matched_indices]]
                    print(f"  → {len(matched_indices)} detections matched to old IDs, {len(detections)} remain for new tracking")
        
        # Continue with normal tracking
        if len(detections) > 0:
            super().update(detections, classes, confidences)
        else:
            # No new detections, just update existing tracks
            for track in self.tracks:
                track.mark_missed()
            self.tracks = [t for t in self.tracks if not t.is_deleted()]
    
    def _initiate_track(self, detection, class_id, conf):
        """Override to use sequential IDs"""
        from .strongsort.sort.track import Track
        
        self.tracks.append(
            Track(
                detection.to_xyah(),
                self.next_new_id,  # Use sequential ID
                class_id,
                conf,
                self.n_init,
                self.max_age,
                self.ema_alpha,
                detection.feature,
            )
        )
        self.next_new_id += 1


class StrongSORTPersistent:
    """
    StrongSORT with persistent re-identification capability.
    Maintains a gallery of all seen persons and reassigns IDs when they reappear.
    """
    
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        max_dist=0.2,
        max_iou_dist=0.7,
        max_age=70,
        max_unmatched_preds=7,
        n_init=3,
        nn_budget=100,
        mc_lambda=0.995,
        ema_alpha=0.9,
        # New parameters for persistent ReID
        enable_persistent_reid=True,
        reid_threshold=0.4,  # Lower = stricter matching (0.3-0.5 recommended)
        persistent_budget=500,  # Features to keep per ID
    ):
        self.model = ReIDDetectMultiBackend(
            weights=model_weights, device=device, fp16=fp16
        )
        
        self.max_dist = max_dist
        self.enable_persistent_reid = enable_persistent_reid
        
        # Use persistent metric if enabled
        if enable_persistent_reid:
            metric = PersistentNearestNeighborDistanceMetric(
                "cosine", 
                self.max_dist, 
                nn_budget,
                persistent_budget=persistent_budget
            )
            print(f"Persistent ReID enabled (threshold: {reid_threshold})")
        else:
            from ..utils.matching import NearestNeighborDistanceMetric
            metric = NearestNeighborDistanceMetric("cosine", self.max_dist, nn_budget)
        
        self.tracker = PersistentTracker(
            metric,
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
            max_unmatched_preds=max_unmatched_preds,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
            enable_reid=enable_persistent_reid,
            reid_threshold=reid_threshold,
        )
    
    def update(self, dets, img):
        """Update tracker with new detections"""
        assert isinstance(dets, np.ndarray), f"Unsupported 'dets' input format '{type(dets)}'"
        assert isinstance(img, np.ndarray), f"Unsupported 'img' input format '{type(img)}'"
        assert len(dets.shape) == 2, f"Unsupported 'dets' dimensions"
        assert dets.shape[1] == 6, f"Unsupported 'dets' 2nd dimension length"
        
        xyxys = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        
        classes = clss
        xywhs = xyxy2xywh(xyxys)
        self.height, self.width = img.shape[:2]
        
        # Generate detections with ReID features
        features = self._get_features(xywhs, img)
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        detections = [
            Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confs)
        ]
        
        # Update tracker
        self.tracker.predict()
        self.tracker.update(detections, clss, confs)
        
        # Output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            
            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            outputs.append(
                np.array([x1, y1, x2, y2, track_id, conf, class_id], dtype=np.float64)
            )
        
        outputs = np.asarray(outputs)
        
        # Print stats periodically
        if hasattr(self.tracker.metric, 'get_stats') and hasattr(self.tracker, 'frame_count'):
            if self.tracker.frame_count % 100 == 0:  # Every 100 frames
                stats = self.tracker.metric.get_stats()
                print(f"[Gallery] Stats: {stats}")
        
        return outputs

    def load_gallery(self, path):
        """Load persistent gallery from disk"""
        if hasattr(self.tracker.metric, 'load_gallery'):
            self.tracker.metric.load_gallery(path)
            self.tracker.sync_next_id()

    def save_gallery(self, path):
        """Save persistent gallery to disk"""
        if hasattr(self.tracker.metric, 'save_gallery'):
            self.tracker.metric.save_gallery(path)
    
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.0
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.0
        return bbox_tlwh
    
    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2
    
    def _tlwh_to_xyxy(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2
    
    @torch.no_grad()
    def _get_features(self, bbox_xywh, img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features
