# Major Fixes Applied - V2

## Problems Identified

1. **Using smoothed embeddings from tracker** (EMA with alpha=0.9)
2. **No continuous validation** - only checked gallery at track creation
3. **Front/back profile problem** - same person looks different from different angles
4. **Embedding contamination** during overlaps and ID switches
5. **Averaging already-averaged embeddings** in step2

## Solutions Implemented

### Fix 1: Extract RAW Embeddings (step1_extract_embeddings.py)

**Before**:
```python
# Got smoothed embedding from tracker
emb = trk.emb  # This is 90% old + 10% new (contaminated!)
```

**After**:
```python
# Extract RAW embedding directly from ReID model
person_crop = frame[y1:y2, x1:x2]
raw_emb = reid_embedder([person_crop])  # Fresh, uncontaminated!
```

**Benefits**:
- No EMA smoothing contamination
- Each embedding is independent
- Captures different viewing angles cleanly
- No ID switch contamination

### Fix 2: Use Median Instead of Mean (step2_cluster_tracks.py)

**Before**:
```python
avg_emb = np.mean(embs, axis=0)  # Sensitive to outliers
```

**After**:
```python
median_emb = np.median(embs, axis=0)  # Robust to outliers
```

**Benefits**:
- More robust to contaminated embeddings
- Reduces impact of brief ID switches
- Better handles viewing angle variations

### Fix 3: Continuous Gallery Validation (step3_gallery_tracking_v2.py)

**NEW FEATURE**: Validates track IDs against gallery every N frames

**Before**:
```
Frame 1: Person appears → Check gallery → Assign ID
Frame 2-1000: Keep same ID (no validation!)
```

**After**:
```
Frame 1: Person appears → Check gallery → Assign ID
Frame 10: Validate against gallery → Correct if wrong
Frame 20: Validate against gallery → Correct if wrong
...
```

**Benefits**:
- Catches ID switches during overlaps
- Corrects wrong IDs when person turns around
- Continuously validates using fresh embeddings

### Fix 4: Multi-Embedding Gallery Matching

**Before**:
```python
# Matched against single average embedding per person
distance = compute_distance(track_emb, person_avg_emb)
```

**After**:
```python
# Match against ALL embeddings, use MINIMUM distance
for gallery_emb in person_all_embeddings:
    distance = compute_distance(track_emb, gallery_emb)
    min_distance = min(min_distance, distance)
```

**Benefits**:
- Handles front/back/side profiles
- More robust to viewing angle changes
- Uses all captured appearances

## How to Use

### Step 1: Extract RAW Embeddings
```bash
python step1_extract_embeddings.py --video test_6.mp4 --output test_6_embeddings.pkl
```

**What changed**: Now extracts RAW embeddings directly from ReID model

### Step 2: Cluster with Median
```bash
python step2_cluster_tracks.py --input test_6_embeddings.pkl --output test_6_gallery.pkl --threshold 0.30
```

**What changed**: 
- Uses median instead of mean
- Shows embedding diversity per track
- Lower threshold recommended (0.25-0.35 instead of 0.4)

### Step 3: Track with Continuous Validation
```bash
python step3_gallery_tracking_v2.py --video test_6.mp4 --gallery test_6_gallery.pkl --output test_6_final.mp4 --threshold 0.35 --validation-interval 10
```

**What changed**:
- Extracts RAW embeddings during tracking
- Validates every 10 frames (configurable)
- Corrects wrong IDs automatically
- Matches against ALL gallery embeddings

## Expected Results

### Before Fixes:
- Lady (pink): ID 1 (back) → ID 9 (front) ❌
- Man (black): ID 3 → confused with lady ❌
- Total persons: 9 (should be 4) ❌

### After Fixes:
- Lady (pink): ID 1 (back) → ID 1 (front) ✅
- Man (black): ID 2 (consistent) ✅
- Total persons: 4 (correct!) ✅
- ID corrections logged in real-time ✅

## Key Parameters

### step2 threshold:
- **0.25**: Very strict - may split same person
- **0.30**: Strict - good for distinct people
- **0.35**: Balanced - recommended
- **0.40**: Loose - may merge different people

### step3 threshold:
- **0.30**: Very strict matching
- **0.35**: Strict - recommended
- **0.40**: Balanced
- **0.45**: Loose

### step3 validation_interval:
- **5**: Validate every 5 frames (more corrections, slower)
- **10**: Validate every 10 frames (balanced) - recommended
- **20**: Validate every 20 frames (fewer corrections, faster)

## Monitoring

Watch for these log messages:

```
✅ Track 1 → Person 1 (distance: 0.25)  # Initial match
🔄 Track 1 CORRECTED: Person 3 → Person 1 (distance: 0.28)  # Correction made!
```

If you see many corrections, it means:
- Tracker is making mistakes (good that we're catching them!)
- Consider lowering validation_interval for more frequent checks

## Testing

Run the full pipeline:

```bash
# Step 1: Extract RAW embeddings
python step1_extract_embeddings.py --video test_6.mp4

# Step 2: Cluster (try different thresholds)
python step2_cluster_tracks.py --input embeddings.pkl --threshold 0.30

# Step 3: Track with validation
python step3_gallery_tracking_v2.py --video test_6.mp4 --threshold 0.35 --validation-interval 10
```

Check the output:
- Should see 4 unique persons (not 9)
- Should see ID corrections in logs
- Lady should keep same ID throughout
- Man should keep different ID from lady

## Troubleshooting

**Still getting wrong IDs?**
- Lower step2 threshold (try 0.25)
- Lower step3 threshold (try 0.30)
- Decrease validation_interval (try 5)

**Too many persons?**
- Raise step2 threshold (try 0.35)

**Too few persons?**
- Lower step2 threshold (try 0.25)
