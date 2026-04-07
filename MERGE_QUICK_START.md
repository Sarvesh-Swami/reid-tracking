# Quick Start: Merge Duplicate IDs

## TL;DR

```bash
# 1. You already ran tracking and got 13 IDs
python track_attendance.py --source test_6.mp4 --output output_v3.mp4

# 2. Merge duplicates
python merge_duplicate_ids.py output_v3

# 3. Check the report
cat output_v3_merge_report.json
```

## What to Expect

Based on your embedding analysis:

### Definite Merges (Distance < 0.30)
```
Person 2 + Person 10 → Person 2 (distance 0.25)
  - Person 2: frames 193-372
  - Person 10: frames 927-1059
  - Same person returning later ✓

Person 4 + Person 5 → Person 4 (distance 0.27)
  - Person 4: frames 267-367
  - Person 5: frames 302-314 (brief)
  - Same person, brief appearance ✓
```

**Result: 13 → 11 persons**

### Possible Merges (Distance 0.30-0.40)
```
Person 3 + Person 12 + Person 13 → Person 3
  - Distances: 0.36, 0.43
  - High intra-variance for Person 3
  - Likely same person with appearance changes
```

**Result: 11 → 9 persons**

### Questionable (Distance > 0.40)
```
Person 7 + Person 8 (distance 0.50)
  - Moderate distance
  - Could be same or different
  - Need manual review
```

## Expected Output

```
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

## What This Tells You

### Person 1 (No Merges)
- Distinct appearance
- No duplicates
- Working correctly ✓

### Person 2 (Merged with 10)
- Same person appearing twice
- Front/back or lighting change
- System split into 2 IDs

### Person 3 (Merged with 12, 13)
- Same person with high variance
- Multiple appearance changes
- System split into 3 IDs

### Person 4 (Merged with 5)
- Same person
- Brief appearance created duplicate
- System split into 2 IDs

## Next Steps

### Option 1: Accept 8-9 Persons

If the merge report shows 8-9 final persons:
- This is closer to reality than 13
- Some splits may be unavoidable
- Consider this "good enough"

### Option 2: Re-run with Better Threshold

Based on merge insights:
```bash
# Use lower threshold (more lenient)
python track_attendance.py --source test_6.mp4 --reid-threshold 0.25 --output output_v3_rerun.mp4

# Then check
python analyze_embeddings.py output_v3_rerun
```

### Option 3: Adjust Other Parameters

```bash
# Rely less on color (since clothing is similar)
python track_attendance.py --source test_6.mp4 --color-weight 0.40 --output output_v3_rerun.mp4

# Or combine
python track_attendance.py --source test_6.mp4 --reid-threshold 0.25 --color-weight 0.40 --output output_v3_rerun.mp4
```

## Understanding the Merges

### Why Person 2 and 10 are Same
```
Distance: 0.2510 (very low)
Person 2 intra-distance: 0.2765

0.2510 < 0.2765 → Person 10 is MORE similar to Person 2 than Person 2 is to themselves!
```

### Why Person 4 and 5 are Same
```
Distance: 0.2650 (very low)
Person 5: Only 11 embeddings (brief appearance)
Person 4 intra-distance: 0.2934

0.2650 < 0.2934 → Person 5 is MORE similar to Person 4 than Person 4 is to themselves!
```

### Why Person 3, 12, 13 Might be Same
```
Person 3 intra-distance: 0.3483 (high variance)
Person 3 vs 12: 0.3589 (almost same as intra)
Person 3 vs 13: 0.4286 (higher but still close)

High intra-variance + close inter-distances = Likely same person with appearance changes
```

## Files Created

- `output_v3_merge_report.json` - Detailed merge information
- Console output - Summary and statistics

## Limitations

This script:
- ✓ Identifies duplicates
- ✓ Creates merge mapping
- ✗ Does NOT create new video with merged IDs

To get video with merged IDs, you need to re-run tracking with adjusted parameters.

## Quick Decision Tree

```
Merge report shows 8-9 persons?
├─ Yes → Good! Much better than 13
│   ├─ Accept as-is
│   └─ Or re-run with reid-threshold 0.25
│
└─ No (still 11-12 persons) → Try aggressive merge
    └─ python merge_duplicate_ids.py output_v3 0.35
```

## Summary

The merge script confirms what the embedding analysis showed:
- **Person 1:** Works perfectly (distinct)
- **Persons 2, 3, 4:** Split into multiple IDs due to appearance changes
- **Merging reduces 13 → 8-9 persons**
- **Still not perfect 4, but much better**

The fundamental issue remains: Similar clothing + front/back differences make perfect tracking impossible with current ReID model.

See **MERGE_IDS_GUIDE.md** for detailed documentation.
