# Quick Start: Embedding Analysis

## TL;DR

```bash
# 1. Run tracking (saves embeddings automatically)
python track_attendance.py --source test_6.mp4 --output output_v3.mp4

# 2. Analyze embeddings
python analyze_embeddings.py output_v3

# 3. Look at the output, find "Recommended threshold"

# 4. Re-run with better threshold
python track_attendance.py --source test_6.mp4 --reid-threshold <RECOMMENDED> --output output_v3_tuned.mp4
```

## What to Look For in Analysis Output

### Good News ✓
```
✓ Good separation! Recommended threshold: 0.3790
```
→ Use this threshold, should work well

### Bad News ⚠
```
⚠ Poor separation! Intra-95% (0.4521) >= Inter-5% (0.3234)
  Same-person and different-person distances overlap significantly
  This explains why the system creates multiple IDs
```
→ Similar clothing problem, need different approach

### Critical Warning
```
⚠ 234 same-person pairs are being rejected!
   This causes the system to create new IDs for the same person
```
→ Threshold too strict, increase it

## Quick Fixes

### If Analysis Shows "Poor Separation"

**Problem:** People wear similar clothes

**Fix 1:** Rely less on color
```bash
python track_attendance.py --source test_6.mp4 --color-weight 0.40
```

**Fix 2:** Lower threshold (accept some errors)
```bash
python track_attendance.py --source test_6.mp4 --reid-threshold 0.30
```

**Fix 3:** Easier reassignment
```python
# Edit track_attendance.py line ~184
self.reassignment_threshold = 0.05  # was 0.10
```

### If Analysis Shows "High Intra-Person Variance"

**Problem:** Front/back profiles very different

**Fix:** More lenient threshold
```bash
python track_attendance.py --source test_6.mp4 --reid-threshold 0.40
```

### If Analysis Shows "Many Rejections"

**Problem:** Threshold too strict

**Fix:** Increase threshold
```bash
python track_attendance.py --source test_6.mp4 --reid-threshold 0.38
```

## Files Generated

After running tracking:
- `output_v3_embeddings.npz` - All embeddings (binary)
- `output_v3_metadata.json` - Frame info (text)

These files contain all the feature vectors extracted during tracking, organized by person ID.

## What the Numbers Mean

### Intra-Person Distance (Same Person)
- **Low (< 0.30):** Good, consistent appearance
- **Medium (0.30-0.40):** OK, some variation
- **High (> 0.40):** Problem, appearance varies too much

### Inter-Person Distance (Different Persons)
- **High (> 0.50):** Good, clearly different
- **Medium (0.40-0.50):** OK, somewhat different
- **Low (< 0.40):** Problem, similar appearance

### Ideal Scenario
```
Intra mean: 0.20 (low)
Inter mean: 0.60 (high)
Gap: 0.40 (large)
→ Easy to distinguish people
```

### Problem Scenario
```
Intra mean: 0.35 (medium)
Inter mean: 0.40 (medium)
Gap: 0.05 (small)
→ Hard to distinguish people
```

## Interpreting Recommendations

### Example 1: Clear Recommendation
```
Intra-person 95th percentile: 0.3456
Inter-person 5th percentile: 0.4123
✓ Good separation! Recommended threshold: 0.3790
```
→ Use 0.38, should work perfectly

### Example 2: Overlap Warning
```
Intra-person 95th percentile: 0.4521
Inter-person 5th percentile: 0.3234
⚠ Poor separation!
  Compromise threshold: 0.3878
```
→ Use 0.39, but expect some errors

### Example 3: Current Threshold Analysis
```
Current reid_threshold: 0.3500
  Same-person pairs rejected: 234/3675 (6.4%)
  Different-person pairs accepted: 45/15000 (0.3%)
```
→ 6.4% false negatives (creating duplicate IDs)
→ 0.3% false positives (wrong assignments)
→ Increase threshold to reduce false negatives

## Decision Tree

```
Run analysis
    ↓
Good separation?
    ├─ Yes → Use recommended threshold ✓
    └─ No → Check what's wrong
        ↓
        High intra-person variance?
        ├─ Yes → Increase threshold
        └─ No → Low inter-person distance?
            ├─ Yes → Similar clothes
            │   ├─ Lower color_weight
            │   ├─ Lower threshold
            │   └─ Lower reassignment_threshold
            └─ No → Check current threshold
                ↓
                Many rejections?
                ├─ Yes → Increase threshold
                └─ No → Many acceptances?
                    ├─ Yes → Decrease threshold
                    └─ No → Use recommended threshold
```

## Common Patterns

### Pattern 1: One Person Works, Others Don't
```
Person 1 vs others: 0.65+ (high)
Person 2 vs Person 3: 0.35 (low)
```
→ Person 1 has distinct clothes
→ Persons 2, 3 have similar clothes
→ Lower color_weight to 0.40

### Pattern 2: All High Intra-Variance
```
All persons: Intra mean > 0.35
```
→ Front/back profile problem
→ Increase threshold to 0.40

### Pattern 3: All Low Inter-Distance
```
All pairs: Inter mean < 0.45
```
→ Everyone wears similar clothes
→ Lower color_weight to 0.35
→ Lower threshold to 0.30

## Summary

1. **Run tracking** → Generates embeddings
2. **Analyze embeddings** → Shows distances
3. **Read recommendations** → Optimal threshold
4. **Adjust parameters** → Based on analysis
5. **Re-run tracking** → Better results
6. **Analyze again** → Verify improvement

See **EMBEDDING_ANALYSIS_GUIDE.md** for detailed explanations.
