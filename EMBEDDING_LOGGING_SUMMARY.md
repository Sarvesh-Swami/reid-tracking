# Embedding Logging - Quick Summary

## What Was Added

The system now saves all embeddings during tracking for later analysis.

## How to Use

### Step 1: Run Tracking
```bash
python track_attendance.py --source test_6.mp4 --output output_v3.mp4
```

**Output files:**
- `output_v3.mp4` - Tracked video
- `output_v3_embeddings.npz` - All embeddings saved
- `output_v3_metadata.json` - Frame info, PIDs, bboxes

### Step 2: Analyze Distances
```bash
python analyze_embeddings.py output_v3
```

**This shows:**
- Intra-person distances (same person, should be LOW)
- Inter-person distances (different persons, should be HIGH)
- Recommended threshold based on data
- Why current threshold is creating multiple IDs

## What You'll Learn

### Example Output

```
Person 1: Mean intra-distance: 0.21 ✓ (consistent)
Person 2: Mean intra-distance: 0.38 ⚠ (varies a lot)

Person 1 vs Person 2: Mean inter-distance: 0.65 ✓ (clearly different)
Person 2 vs Person 3: Mean inter-distance: 0.35 ⚠ (similar clothes!)

Recommended threshold: 0.38
Current threshold: 0.35

⚠ 234 same-person pairs are being rejected!
   This causes the system to create new IDs for the same person
```

**Interpretation:**
- Person 2's appearance varies (front/back profiles)
- Persons 2 and 3 wear similar clothes (low inter-distance)
- Current threshold 0.35 is too strict
- Should use 0.38 instead

## Why This Helps

### Problem: "Why am I getting 13 IDs instead of 4?"

**Analysis shows:**
```
Person 2 vs Person 3: Mean 0.35
Person 2 vs Person 4: Mean 0.33
Person 3 vs Person 4: Mean 0.32

⚠ These are LOWER than Person 2's intra-distance (0.38)!
```

**Diagnosis:** Persons 2, 3, 4 are more similar to each other than Person 2 is to themselves (front vs back)!

**Solution:** 
- Can't use simple threshold
- Need to rely more on ReID features (less on color)
- Need temporal reasoning

## Code Changes Made

### 1. Added Embedding Logging

In `track_attendance.py`:
```python
# Added to __init__
self.embedding_log = []

# Added in _map_ids when pid > 0
self.embedding_log.append({
    'frame': self.frame_count,
    'pid': pid,
    'tid': tid,
    'features': feat.copy(),
    'color_hist': chist.copy() if chist is not None else None,
    'bbox': bbox.tolist()
})
```

### 2. Added Save Function

```python
def _save_embeddings(self, output_video_path):
    # Saves embeddings to .npz file
    # Saves metadata to .json file
```

### 3. Created Analysis Script

`analyze_embeddings.py`:
- Loads saved embeddings
- Calculates intra/inter distances
- Recommends optimal threshold
- Shows why current threshold fails

## Files Created

1. **EMBEDDING_ANALYSIS_GUIDE.md** - Detailed guide
2. **analyze_embeddings.py** - Analysis script
3. **EMBEDDING_LOGGING_SUMMARY.md** - This file

## Quick Example

```bash
# Run tracking
python track_attendance.py --source test_6.mp4 --output output_v3.mp4

# Analyze
python analyze_embeddings.py output_v3

# Output shows:
# Recommended threshold: 0.38
# Current threshold: 0.35 (too strict)

# Re-run with better threshold
python track_attendance.py --source test_6.mp4 --reid-threshold 0.38 --output output_v3_better.mp4

# Analyze again
python analyze_embeddings.py output_v3_better

# Compare: Should have fewer unique persons now
```

## What the Analysis Reveals

### Scenario 1: Person 1 Works, Others Don't

```
Person 1 intra: 0.20 ✓
Person 1 vs others: 0.65+ ✓ (clearly different)

Person 2 intra: 0.38 ⚠ (high variance)
Person 2 vs Person 3: 0.35 ⚠ (similar!)
```

**Reason:** Person 1 has distinct clothes, others are similar

### Scenario 2: Front/Back Profile Problem

```
Person 2 intra: 0.42 ⚠
  - Front profile embeddings: cluster around 0.15
  - Back profile embeddings: cluster around 0.20
  - Front vs Back: 0.55 ⚠ (very different!)
```

**Reason:** ReID model struggles with front/back

### Scenario 3: Threshold Too Strict

```
Intra 95th percentile: 0.40
Current threshold: 0.35

⚠ Rejecting 15% of valid same-person matches!
```

**Reason:** Threshold doesn't account for appearance variation

## Benefits

1. **Understand the problem** - See actual distance distributions
2. **Data-driven tuning** - Use real data to set thresholds
3. **Identify issues** - See which persons are confused
4. **Validate changes** - Compare before/after analysis

## Next Steps

1. Run tracking to generate embeddings
2. Analyze to understand distances
3. Adjust thresholds based on recommendations
4. Re-run and compare results
5. Iterate until satisfied

See **EMBEDDING_ANALYSIS_GUIDE.md** for detailed explanations.
