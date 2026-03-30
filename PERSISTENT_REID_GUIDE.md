# Persistent Re-Identification Guide

## 🎯 Problem Solved

Standard tracking algorithms (DeepOCSORT, StrongSORT, BoTSORT) **permanently delete track IDs** when a person leaves the frame for more than `max_age` frames. When the same person reappears, they get a **new ID**.

This implementation adds **Persistent Re-Identification** that:
- ✅ Maintains a gallery of ReID features for ALL persons ever seen
- ✅ Reassigns the SAME ID when a person reappears
- ✅ Works even after long occlusions or when person leaves and returns

## 🚀 Quick Start

### Run with Persistent ReID:

```bash
python track_persistent_reid.py --source test_3.mp4 --reid-threshold 0.4 --show
```

### Key Parameters:

- `--reid-threshold`: Lower = stricter matching (recommended: 0.3-0.5)
  - `0.3`: Very strict (fewer false positives, may miss some re-IDs)
  - `0.4`: Balanced (recommended)
  - `0.5`: Lenient (more re-IDs, but risk of false matches)

- `--max-age`: How long to keep tracks alive (default: 70 frames)
  - Higher = tracks survive longer occlusions
  - Lower = faster deletion, more opportunities for re-ID

- `--reid-model`: Choose ReID model for feature extraction
  - `osnet_x0_25_msmt17.pt`: Lightweight, fast
  - `osnet_x1_0_msmt17.pt`: Better accuracy
  - `lmbn_n_cuhk03_d.pt`: LightMBN, good balance

## 📊 How It Works

### 1. Feature Gallery
```
Active Tracks:     ID 1, ID 2, ID 3
Deleted Tracks:    ID 4, ID 5, ID 6  ← Features kept in persistent gallery
```

### 2. Re-Identification Process
```
New Detection → Extract ReID Features → Search Persistent Gallery
                                       ↓
                                  Match Found?
                                  ↓         ↓
                                YES        NO
                                 ↓          ↓
                          Reassign Old ID   Create New ID
```

### 3. Feature Matching
- Uses **cosine distance** between ReID embeddings
- Compares new detection against ALL deleted track features
- Reassigns ID if distance < `reid_threshold`

## 🔧 Technical Implementation

### Key Components:

1. **PersistentNearestNeighborDistanceMetric** (`boxmot/utils/persistent_reid_matching.py`)
   - Extends standard distance metric
   - Maintains two galleries:
     - `samples`: Active tracks (for normal tracking)
     - `persistent_samples`: ALL tracks (never deleted)
   - Tracks deleted IDs for re-identification

2. **StrongSORTPersistent** (`boxmot/trackers/strongsort_persistent.py`)
   - Enhanced StrongSORT tracker
   - Before normal tracking, checks new detections against deleted tracks
   - Reactivates old IDs when matches found

3. **PersistentTracker** (in same file)
   - Extended Tracker class
   - Handles ID reassignment logic
   - Maintains sequential ID counter

## 📈 Performance Tuning

### For Better Re-Identification:

1. **Lower reid_threshold** (0.3-0.35)
   - More confident matches only
   - Fewer false positives

2. **Better ReID model**
   ```bash
   --reid-model osnet_x1_0_msmt17.pt  # More accurate features
   ```

3. **Higher persistent_budget** (in code)
   ```python
   tracker = StrongSORTPersistent(
       ...
       persistent_budget=1000,  # Keep more features per ID
   )
   ```

4. **Lower detection confidence**
   ```bash
   --conf 0.3  # Detect people earlier/more reliably
   ```

### For Faster Processing:

1. **Smaller ReID model**
   ```bash
   --reid-model osnet_x0_25_msmt17.pt  # Lightweight
   ```

2. **Lower persistent_budget**
   ```python
   persistent_budget=100  # Fewer features to search
   ```

## 🎨 Example Scenarios

### Scenario 1: Person Leaves and Returns
```
Frame 1-100:   Person A (ID 1) visible
Frame 101-200: Person A leaves frame → Track deleted after max_age
Frame 201-300: Person A returns → ReID matches → ID 1 reassigned! ✅
```

### Scenario 2: Multiple People
```
Frame 1-50:    Person A (ID 1), Person B (ID 2)
Frame 51-100:  Person A leaves, Person B stays
Frame 101-150: Person C appears (ID 3)
Frame 151-200: Person A returns → ID 1 reassigned! ✅
```

### Scenario 3: Long Occlusion
```
Frame 1-50:    Person A (ID 1) visible
Frame 51-500:  Person A occluded/gone (450 frames!)
Frame 501+:    Person A returns → ID 1 reassigned! ✅
```

## 🔍 Debugging

### Check Gallery Stats:
The script prints stats every 100 frames:
```
📊 Gallery stats: {
    'active_ids': 3,           # Currently tracked
    'deleted_ids': 5,          # Available for re-ID
    'total_ids_ever': 8,       # All IDs ever seen
    'total_features': 2400     # Total features stored
}
```

### Watch for Re-ID Messages:
```
🔄 Re-identified! Reassigning ID 5 (distance: 0.234)
```

### Adjust Threshold:
- Too many false re-IDs? → Lower threshold (0.3)
- Missing re-IDs? → Raise threshold (0.5)

## 📝 Code Integration

### Use in Your Own Script:

```python
from boxmot.trackers.strongsort_persistent import StrongSORTPersistent
from boxmot.utils import WEIGHTS

# Initialize tracker
tracker = StrongSORTPersistent(
    model_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',
    device='cuda:0',
    fp16=False,
    max_age=70,
    enable_persistent_reid=True,      # Enable persistent ReID
    reid_threshold=0.4,                # Matching threshold
    persistent_budget=500,             # Features per ID
)

# Use like normal tracker
tracks = tracker.update(detections, frame)
# tracks: [[x1, y1, x2, y2, id, conf, cls], ...]
```

### Disable Persistent ReID:
```python
tracker = StrongSORTPersistent(
    ...
    enable_persistent_reid=False,  # Acts like normal StrongSORT
)
```

## ⚠️ Limitations

1. **Memory Usage**: Grows with number of unique persons
   - Solution: Set `persistent_budget` to limit features per ID

2. **Search Time**: Increases with deleted tracks
   - Solution: Periodic cleanup of very old IDs (not implemented)

3. **Appearance Changes**: May fail if person changes clothes/appearance
   - Solution: Lower threshold for stricter matching

4. **Similar Appearances**: May confuse similar-looking people
   - Solution: Lower threshold, better ReID model

## 🎯 Comparison

| Feature | Standard Trackers | Persistent ReID |
|---------|------------------|-----------------|
| Track during occlusion | ✅ (up to max_age) | ✅ (up to max_age) |
| Reassign ID after deletion | ❌ | ✅ |
| Memory usage | Low | Medium |
| Processing speed | Fast | Slightly slower |
| ID consistency | Good (short-term) | Excellent (long-term) |

## 🚀 Next Steps

1. **Test on your video**:
   ```bash
   python track_persistent_reid.py --source test_3.mp4 --show
   ```

2. **Tune reid_threshold** based on results

3. **Try different ReID models** for better accuracy

4. **Integrate into your pipeline** using the code examples above

## 📚 References

- StrongSORT: [Paper](https://arxiv.org/abs/2202.13514)
- OSNet ReID: [Paper](https://arxiv.org/abs/1905.00953)
- Deep Person ReID: [GitHub](https://github.com/KaiyangZhou/deep-person-reid)
