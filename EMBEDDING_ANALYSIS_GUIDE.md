# Embedding Analysis Guide

## What This Does

The system now saves all embeddings (feature vectors) with their assigned person IDs during tracking. You can then analyze these embeddings to understand:

1. **Intra-person distances** - How different are embeddings of the SAME person?
2. **Inter-person distances** - How different are embeddings of DIFFERENT persons?
3. **Optimal thresholds** - What threshold should you use for matching?

## Step 1: Run Tracking with Embedding Logging

```bash
python track_attendance.py --source test_6.mp4 --output output_v3.mp4
```

This will create:
- `output_v3.mp4` - Tracked video
- `output_v3_embeddings.npz` - All embeddings (numpy compressed)
- `output_v3_metadata.json` - Metadata (frames, PIDs, bboxes)

## Step 2: Analyze Embeddings

```bash
python analyze_embeddings.py output_v3
```

This will calculate and display:

### Intra-Person Distances (Same Person)

```
Person 1: 50 embeddings, 1225 pairs
  Min distance:  0.0234
  Max distance:  0.4521
  Mean distance: 0.2145
  Median:        0.2089
```

**What this means:**
- Person 1 has 50 embeddings collected throughout the video
- Comparing all pairs gives 1225 distance measurements
- Mean 0.2145 = On average, Person 1's embeddings are 0.2145 apart

**Good values:**
- Low mean (< 0.30) = Consistent appearance
- Low max (< 0.50) = No extreme variations

**Bad values:**
- High mean (> 0.40) = Inconsistent appearance (front/back, lighting changes)
- High max (> 0.60) = Very different appearances within same person

### Inter-Person Distances (Different Persons)

```
Person 1 vs Person 2: 2500 pairs
  Min distance:  0.3421
  Max distance:  0.7234
  Mean distance: 0.5123
  Median:        0.5089
```

**What this means:**
- Comparing Person 1's 50 embeddings with Person 2's 50 embeddings
- 2500 pairwise distances calculated
- Mean 0.5123 = On average, they are 0.5123 apart

**Good values:**
- High mean (> 0.50) = Clearly different persons
- High min (> 0.40) = Always distinguishable

**Bad values:**
- Low mean (< 0.40) = Similar appearance (similar clothes)
- Low min (< 0.30) = Some embeddings are very similar (confusion possible)

### Summary Statistics

```
All INTRA-person distances (same person):
  Count: 3675
  Mean:   0.2345
  Median: 0.2289

All INTER-person distances (different persons):
  Count: 15000
  Mean:   0.5234
  Median: 0.5189
```

**Ideal scenario:**
- Intra mean << Inter mean (large gap)
- Example: Intra 0.20, Inter 0.60 = Easy to distinguish

**Problem scenario:**
- Intra mean ≈ Inter mean (overlap)
- Example: Intra 0.35, Inter 0.40 = Hard to distinguish

### Threshold Recommendations

```
Intra-person 95th percentile: 0.3456
  (95% of same-person pairs have distance < 0.3456)

Inter-person 5th percentile: 0.4123
  (5% of different-person pairs have distance < 0.4123)

✓ Good separation! Recommended threshold: 0.3790
  (Midpoint between intra-95% and inter-5%)
```

**What this means:**
- 95% of same-person pairs are below 0.3456
- 95% of different-person pairs are above 0.4123
- There's a gap! Use threshold 0.3790

**If you see:**
```
⚠ Poor separation! Intra-95% (0.4521) >= Inter-5% (0.3234)
  Same-person and different-person distances overlap significantly
  This explains why the system creates multiple IDs
```

**This means:**
- Same-person and different-person distances overlap
- No perfect threshold exists
- Similar clothing is causing confusion

### Current Threshold Analysis

```
Current reid_threshold: 0.3500

  Same-person pairs rejected: 234/3675 (6.4%)
  Different-person pairs accepted: 45/15000 (0.3%)

  ⚠ 234 same-person pairs are being rejected!
     This causes the system to create new IDs for the same person
```

**What this means:**
- With threshold 0.35, you're rejecting 6.4% of valid same-person matches
- This is why Person 2, 3, 4 keep getting new IDs
- You're also accepting 0.3% of wrong matches (different persons)

## Interpreting Results

### Scenario 1: Good Separation

```
Intra mean: 0.20, Intra 95%: 0.30
Inter mean: 0.60, Inter 5%: 0.45

Recommended threshold: 0.375
```

**Action:** Use threshold 0.375, system should work well

### Scenario 2: Moderate Overlap

```
Intra mean: 0.30, Intra 95%: 0.42
Inter mean: 0.50, Inter 5%: 0.38

Recommended threshold: 0.40 (compromise)
```

**Action:** 
- Use threshold 0.40
- Expect some errors
- Consider increasing color_weight

### Scenario 3: Severe Overlap (Your Case)

```
Intra mean: 0.35, Intra 95%: 0.48
Inter mean: 0.42, Inter 5%: 0.32

⚠ Poor separation!
```

**Action:**
- Lower threshold to 0.30 (accept more false positives)
- Increase color_weight to 0.70 (rely more on clothing)
- Lower reassignment_threshold to 0.05 (easier reassignment)
- Accept that some errors will occur

## What to Look For

### Person 1 Works, Others Don't

```
Person 1 vs Person 2: Mean 0.65 ✓ (clearly different)
Person 1 vs Person 3: Mean 0.68 ✓ (clearly different)
Person 1 vs Person 4: Mean 0.70 ✓ (clearly different)

Person 2 vs Person 3: Mean 0.38 ⚠ (similar!)
Person 2 vs Person 4: Mean 0.35 ⚠ (similar!)
Person 3 vs Person 4: Mean 0.33 ⚠ (similar!)
```

**Diagnosis:** Persons 2, 3, 4 wear similar clothes, Person 1 is distinct

**Solution:**
- Lower threshold won't help (will cause wrong assignments)
- Need to rely more on ReID features (body shape, gait)
- Decrease color_weight to 0.40
- Add temporal/spatial reasoning

### High Intra-Person Variance

```
Person 2: 
  Mean: 0.42 ⚠ (high!)
  Max:  0.68 ⚠ (very high!)
```

**Diagnosis:** Person 2's appearance varies a lot (front/back, lighting)

**Solution:**
- This is expected for front/back profiles
- Need more lenient threshold
- Or collect more diverse features during probation

## Using the Results

### 1. Adjust reid_threshold

Based on recommended threshold:
```bash
python track_attendance.py --source test_6.mp4 --reid-threshold 0.38
```

### 2. Adjust color_weight

If inter-person distances are low (similar clothes):
```bash
python track_attendance.py --source test_6.mp4 --color-weight 0.40
```

If intra-person distances are high (appearance varies):
```bash
python track_attendance.py --source test_6.mp4 --color-weight 0.65
```

### 3. Adjust reassignment_threshold

If many same-person pairs are rejected:
```python
# Edit track_attendance.py line ~184
self.reassignment_threshold = 0.05  # Lower = easier reassignment
```

## Files Generated

### output_v3_embeddings.npz

Binary file containing:
- `pid_1_features`: (N, 512) array of Person 1's embeddings
- `pid_1_color_hists`: (N, 272) array of Person 1's color histograms
- `pid_2_features`: (M, 512) array of Person 2's embeddings
- etc.

**Load in Python:**
```python
import numpy as np
data = np.load('output_v3_embeddings.npz')
person1_features = data['pid_1_features']
print(person1_features.shape)  # (50, 512)
```

### output_v3_metadata.json

JSON file containing:
```json
{
  "total_embeddings": 234,
  "unique_pids": 13,
  "embeddings_per_pid": {
    "1": 50,
    "2": 35,
    ...
  },
  "frame_info": [
    {
      "frame": 3,
      "pid": 1,
      "tid": 1,
      "bbox": [x1, y1, x2, y2]
    },
    ...
  ]
}
```

## Example Workflow

```bash
# 1. Run tracking
python track_attendance.py --source test_6.mp4 --output output_v3.mp4

# 2. Analyze embeddings
python analyze_embeddings.py output_v3

# 3. Read the output, note recommended threshold

# 4. Re-run with better threshold
python track_attendance.py --source test_6.mp4 --reid-threshold 0.38 --output output_v3_tuned.mp4

# 5. Analyze again
python analyze_embeddings.py output_v3_tuned

# 6. Compare results
```

## Understanding Distance Metrics

### Cosine Distance

```
distance = 1 - cosine_similarity
distance = 1 - (A · B) / (||A|| ||B||)
```

**Range:** 0.0 to 2.0
- 0.0 = Identical vectors
- 1.0 = Orthogonal vectors
- 2.0 = Opposite vectors

**Typical values:**
- Same person: 0.1 - 0.4
- Different persons: 0.4 - 0.8
- Very different: 0.8 - 1.2

### What Affects Distances

**Intra-person (same person):**
- Pose changes (front/back)
- Lighting changes
- Occlusions
- Camera angle
- Distance from camera

**Inter-person (different persons):**
- Clothing similarity
- Body shape similarity
- Height similarity
- Pose similarity

## Troubleshooting

### "No separation between intra and inter"

**Cause:** People wear very similar clothes

**Solutions:**
1. Lower color_weight (rely less on clothing)
2. Accept some errors (lower threshold)
3. Add temporal reasoning (track positions)
4. Use better ReID model

### "High intra-person variance"

**Cause:** Front/back profiles very different

**Solutions:**
1. Collect more features during probation
2. Use higher threshold (more lenient)
3. Use pose-aware ReID model

### "Many same-person pairs rejected"

**Cause:** Threshold too strict

**Solutions:**
1. Increase reid_threshold
2. Decrease reassignment_threshold
3. Increase confirmation_frames

## Summary

The embedding analysis tells you:
1. **Why** the system is creating multiple IDs
2. **What** threshold to use
3. **How** to tune the parameters

Use it to understand the distance distributions and make informed decisions about thresholds.
