# Merge Duplicate IDs - Guide

## What This Does

Post-processes the tracking results to merge duplicate person IDs based on:
1. **Low embedding distance** (< 0.30 by default)
2. **No temporal overlap** (never appear at same time)

## Quick Start

```bash
# Run tracking first
python track_attendance.py --source test_6.mp4 --output output_v3.mp4

# Analyze to see the problem
python analyze_embeddings.py output_v3

# Merge duplicates
python merge_duplicate_ids.py output_v3
```

## How It Works

### Step 1: Find Merge Candidates

For each pair of persons:
1. Calculate mean embedding distance
2. Check if they ever appear together
3. If distance < 0.30 AND no overlap → Merge candidate

### Step 2: Create Merge Map

```
Person 2 + Person 10 → Person 2 (distance 0.25)
Person 4 + Person 5 → Person 4 (distance 0.27)
Person 3 + Person 12 + Person 13 → Person 3 (distances 0.36-0.43)
```

### Step 3: Generate Report

Creates `output_v3_merge_report.json` with:
- All merge candidates
- Final merge map
- Statistics

## Example Output

```
======================================================================
FINDING MERGE CANDIDATES
======================================================================

Person 2 vs Person 10:
  Mean distance: 0.2510
  Min distance:  0.0484
  Temporal overlap: False
  Person 2 frames: 193-372
  Person 10 frames: 927-1059
  ✓ MERGE CANDIDATE!

Person 4 vs Person 5:
  Mean distance: 0.2650
  Min distance:  0.0921
  Temporal overlap: False
  Person 4 frames: 267-367
  Person 5 frames: 302-314
  ✓ MERGE CANDIDATE!

======================================================================
CREATING MERGE MAP
======================================================================

Merge Groups:
  Person 2 ← [2, 10]
    (Merging 2, 10 into Person 2)
  Person 4 ← [4, 5]
    (Merging 4, 5 into Person 4)
  Person 3 ← [3, 12, 13]
    (Merging 3, 12, 13 into Person 3)

======================================================================
MERGE SUMMARY
======================================================================

Original person count: 13
Persons merged: 5
Final person count: 8

Merge Details:
  Person 5 → Person 4
  Person 10 → Person 2
  Person 12 → Person 3
  Person 13 → Person 3

Final Person IDs: [1, 2, 3, 4, 6, 7, 8, 9, 11]
```

## Command Line Options

### Basic Usage
```bash
python merge_duplicate_ids.py output_v3
```
Uses default threshold 0.30

### Custom Threshold
```bash
python merge_duplicate_ids.py output_v3 0.35
```
Uses threshold 0.35 (more aggressive merging)

### Lower Threshold (Conservative)
```bash
python merge_duplicate_ids.py output_v3 0.25
```
Only merges very similar persons

## Interpreting Results

### Good Merge
```
Person 2 vs Person 10:
  Mean distance: 0.2510 ✓ (very low)
  Temporal overlap: False ✓ (never together)
  Person 2 frames: 193-372
  Person 10 frames: 927-1059 (appears later)
  ✓ MERGE CANDIDATE!
```
→ Clearly the same person returning

### Rejected Merge
```
Person 1 vs Person 2:
  Mean distance: 0.7143 ✗ (high)
  Temporal overlap: False
```
→ Too different, not merged

### Prevented Merge
```
Person 3 vs Person 7:
  Mean distance: 0.2800 ✓ (low)
  Temporal overlap: True ✗ (appear together)
```
→ Similar but appear together, not merged

## Files Generated

### output_v3_merge_report.json

```json
{
  "merge_candidates": [
    {
      "from_pid": 10,
      "to_pid": 2,
      "distance": 0.2510,
      "min_distance": 0.0484,
      "reason": "Low distance (0.251) + No overlap"
    }
  ],
  "merge_map": {
    "10": 2,
    "5": 4,
    "12": 3,
    "13": 3
  },
  "summary": {
    "total_merges": 4,
    "original_pids": 13,
    "final_pids": 9
  }
}
```

## Using the Merge Map

### Option 1: Manual Post-Processing

Load the merge map and apply to your results:

```python
import json

# Load merge map
with open('output_v3_merge_report.json', 'r') as f:
    report = json.load(f)
    merge_map = {int(k): v for k, v in report['merge_map'].items()}

# Apply to your person IDs
def apply_merge(pid):
    return merge_map.get(pid, pid)

# Example
old_pid = 10
new_pid = apply_merge(old_pid)  # Returns 2
```

### Option 2: Re-run Tracking

Use the insights to adjust parameters:

```bash
# Based on merge analysis, use lower threshold
python track_attendance.py --source test_6.mp4 --reid-threshold 0.25 --output output_v3_rerun.mp4
```

## Tuning the Threshold

### Threshold 0.25 (Conservative)
- Only merges very similar persons
- Fewer merges, more IDs remain
- Use if you want to be safe

### Threshold 0.30 (Default)
- Balanced approach
- Merges obvious duplicates
- Recommended starting point

### Threshold 0.35 (Aggressive)
- Merges more persons
- May merge some different persons
- Use if you have many duplicates

### Threshold 0.40 (Very Aggressive)
- Merges many persons
- Higher risk of wrong merges
- Use only if desperate

## Expected Results for Your Video

Based on the embedding analysis:

### Likely Merges (Threshold 0.30)
```
Person 2 + Person 10 → Person 2 (distance 0.25)
Person 4 + Person 5 → Person 4 (distance 0.27)
```
**Result: 13 → 11 persons**

### Possible Merges (Threshold 0.35)
```
Person 3 + Person 12 → Person 3 (distance 0.36)
Person 7 + Person 12 → Person 7 (distance 0.39)
```
**Result: 11 → 9 persons**

### Aggressive Merges (Threshold 0.40)
```
Person 3 + Person 13 → Person 3 (distance 0.43)
Person 4 + Person 8 → Person 4 (distance 0.43)
```
**Result: 9 → 7 persons**

## Limitations

### What This Script Does
- ✓ Identifies duplicate IDs
- ✓ Creates merge mapping
- ✓ Generates report

### What This Script Does NOT Do
- ✗ Re-draw bounding boxes with new IDs
- ✗ Modify the original video
- ✗ Update the tracking system

### To Get Video with Merged IDs

You need to:
1. Use the merge insights to adjust tracking parameters
2. Re-run tracking with better thresholds
3. Or implement a video re-labeling tool

## Troubleshooting

### "NO MERGE CANDIDATES FOUND"

**Cause:** Threshold too strict or all persons appear together

**Solutions:**
```bash
# Try higher threshold
python merge_duplicate_ids.py output_v3 0.35

# Check the analysis
python analyze_embeddings.py output_v3
```

### "Too Many Merges"

**Cause:** Threshold too loose

**Solutions:**
```bash
# Use lower threshold
python merge_duplicate_ids.py output_v3 0.25

# Check merge report for suspicious merges
```

### "Merging Different Persons"

**Cause:** Similar clothing + no temporal overlap

**Check:**
- Look at the distance values
- If distance > 0.35, probably wrong merge
- Review the frames where they appear

## Workflow

```
1. Run tracking
   ↓
2. Analyze embeddings
   ↓
3. See duplicate IDs (e.g., 13 instead of 4)
   ↓
4. Run merge script
   ↓
5. Review merge report
   ↓
6. If merges look good:
   - Use insights to adjust tracking parameters
   - Re-run tracking
   ↓
7. If merges look wrong:
   - Adjust threshold
   - Re-run merge script
```

## Integration with Tracking

To integrate merges into the tracking system, you would need to:

1. **During tracking:** Apply merge map in real-time
2. **After tracking:** Post-process all frame assignments
3. **Video output:** Re-draw boxes with merged IDs

This requires modifying `track_attendance.py` to accept a merge map parameter.

## Summary

The merge script:
- Identifies obvious duplicate IDs
- Creates a merge mapping
- Helps you understand which persons are being split
- Provides data to improve tracking parameters

Use it to:
- Understand why you're getting 13 IDs instead of 4
- Decide what threshold to use for re-tracking
- Validate that merges make sense

It's a diagnostic and planning tool, not an automatic fix.
