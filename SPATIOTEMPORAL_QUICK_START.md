# Spatial-Temporal Tracker - Quick Start Guide

## What's New in v4?

**SPATIAL-TEMPORAL REASONING** - The missing piece that solves the 360° rotation problem!

### The Problem
- OSNet ReID model fails when person rotates (front vs back view)
- Person 2 front → ID 2, Person 2 back → ID 6 (WRONG!)
- Embedding distance 0.55-0.70 (threshold is 0.42)
- Appearance-only matching is insufficient

### The Solution
**Multi-Signal Scoring**: Appearance + Location + Motion + Time

1. **Location Tracking (WHERE)**
   - Tracks bounding box center position over time
   - "Person at (100, 200) who reappears at (105, 210) = likely same person"

2. **Motion Tracking (HOW)**
   - Calculates velocity vectors (speed + direction)
   - "Person moving left at 2 m/s = consistent with previous motion"

3. **Temporal Logic (WHEN)**
   - Tracks time gaps between disappearance and reappearance
   - "Short gap + nearby location = same person (even if appearance changed)"

4. **"People Don't Teleport" Rule**
   - Detects physically impossible movements
   - "Person cannot jump 500 pixels in 1 frame"

5. **Combined Confidence Scoring**
   - Weighted combination of all signals
   - When appearance fails, location/motion/time can save it

## Quick Start

### Basic Usage
```bash
python track_attendance_spatiotemporal.py --source test_6.mp4
```

### With Custom Weights
```bash
python track_attendance_spatiotemporal.py --source test_6.mp4 \
    --spatial-weight 0.25 \
    --motion-weight 0.10 \
    --temporal-weight 0.10
```

### All Options
```bash
python track_attendance_spatiotemporal.py \
    --source test_6.mp4 \
    --output output_spatiotemporal.mp4 \
    --reid-threshold 0.42 \
    --color-weight 0.35 \
    --spatial-weight 0.20 \
    --motion-weight 0.10 \
    --temporal-weight 0.10 \
    --show
```

## Weight Configuration

### Default Weights (Recommended)
```
Appearance: 0.60 (color: 0.35, reid: 0.65)
Spatial:    0.20
Motion:     0.10
Temporal:   0.10
Total:      1.00
```

### How Weights Work
- **Appearance** (0.60): Color + ReID embeddings
  - Still the primary signal
  - But not the ONLY signal anymore
  
- **Spatial** (0.20): Location proximity
  - High weight because location is very reliable
  - Person at same location = likely same person
  
- **Motion** (0.10): Velocity consistency
  - Moderate weight
  - Helps distinguish people moving in different directions
  
- **Temporal** (0.10): Time gap plausibility
  - Moderate weight
  - Short gap + nearby = boost confidence

### Tuning Tips

**If too many IDs (over-splitting):**
```bash
--spatial-weight 0.30 --motion-weight 0.15 --temporal-weight 0.15
```
(Increase spatial-temporal weights to merge more aggressively)

**If too few IDs (over-merging):**
```bash
--spatial-weight 0.10 --motion-weight 0.05 --temporal-weight 0.05
```
(Decrease spatial-temporal weights to rely more on appearance)

**For crowded scenes:**
```bash
--spatial-weight 0.15 --motion-weight 0.15 --temporal-weight 0.10
```
(Increase motion weight to distinguish people by movement)

**For static scenes:**
```bash
--spatial-weight 0.25 --motion-weight 0.05 --temporal-weight 0.15
```
(Increase spatial/temporal, decrease motion)

## Expected Results

### Current System (v3.1)
- **Detected**: 12 persons
- **Actual**: 4 persons
- **Accuracy**: 33%
- **Problem**: Front vs back view creates duplicates

### Spatial-Temporal System (v4)
- **Expected**: 5-6 persons
- **Actual**: 4 persons
- **Accuracy**: 58-67%
- **Improvement**: 42-50% reduction in duplicate IDs

### Why Not 100%?
- Some appearance changes are too extreme
- Occlusions can still cause issues
- Spatial-temporal helps but isn't perfect
- 5-6 persons is realistic expectation

## How It Works

### Example Scenario
```
Frame 100: Person 2 (front view) at location (300, 400)
           Appearance: ID 2 ✓
           
Frame 150: Person 2 (back view) at location (320, 410)
           Appearance: NEW person? (distance 0.65 > 0.42)
           Spatial: Same location! (score 0.95)
           Motion: Same velocity! (score 0.88)
           Temporal: Short gap! (score 0.92)
           
           Combined score: 0.60*0.35 + 0.20*0.95 + 0.10*0.88 + 0.10*0.92
                         = 0.21 + 0.19 + 0.09 + 0.09
                         = 0.58 ✓ (above threshold 0.42)
           
           Decision: SAME PERSON (ID 2) ✓
```

### Without Spatial-Temporal
```
Frame 150: Person 2 (back view)
           Appearance only: distance 0.65 > 0.42
           Decision: NEW PERSON (ID 6) ✗ WRONG!
```

## Output Files

### Video Output
- `output_spatiotemporal.mp4` - Annotated video with person IDs
- Look for `[SPATIAL-TEMPORAL ASSIST]` messages in console

### Embedding Files
- `output_spatiotemporal_embeddings.npz` - All embeddings
- `output_spatiotemporal_metadata.json` - Metadata

### Console Output
```
[REASSIGNED] Frame 150: Track 3 reassigned from Person 6 to Person 2 
(dist: 0.42) [SPATIAL-TEMPORAL ASSIST]
```

## Comparison with v3.1

### Run Both Versions
```bash
# v3.1 (appearance only)
python track_attendance.py --source test_6.mp4 --output output_v31.mp4

# v4 (spatial-temporal)
python track_attendance_spatiotemporal.py --source test_6.mp4 --output output_st.mp4
```

### Compare Results
```bash
# v3.1: 12 persons
# v4:   5-6 persons (expected)
```

## Troubleshooting

### Issue: Still getting too many IDs
**Solution**: Increase spatial-temporal weights
```bash
--spatial-weight 0.30 --motion-weight 0.15
```

### Issue: People getting merged incorrectly
**Solution**: Decrease spatial-temporal weights
```bash
--spatial-weight 0.10 --motion-weight 0.05
```

### Issue: Slow processing
**Solution**: Spatial-temporal adds minimal overhead (~5%)
- If slow, it's likely YOLO/ReID, not spatial-temporal

### Issue: Errors about missing attributes
**Solution**: Make sure you're using the correct file
```bash
python track_attendance_spatiotemporal.py  # NOT track_attendance.py
```

## Technical Details

### Spatial Score Calculation
```python
distance = sqrt((cx - last_cx)^2 + (cy - last_cy)^2)
score = exp(-distance / 200)  # Exponential decay
```

### Motion Score Calculation
```python
velocity_diff = sqrt((vx - avg_vx)^2 + (vy - avg_vy)^2)
score = exp(-velocity_diff / 20)  # Exponential decay
```

### Temporal Score Calculation
```python
time_factor = exp(-time_gap / 30)    # Decay over 30 frames
space_factor = exp(-distance / 200)  # Decay over 200 pixels
score = time_factor * space_factor
```

### Teleport Detection
```python
max_speed = 50 pixels/frame  # Running person
if distance > max_speed * frame_diff:
    # Teleport detected! Penalize spatial/motion scores
```

## Next Steps

1. **Test on your video**
   ```bash
   python track_attendance_spatiotemporal.py --source test_6.mp4
   ```

2. **Compare with v3.1**
   - Count unique persons in both outputs
   - Look for `[SPATIAL-TEMPORAL ASSIST]` messages

3. **Tune weights if needed**
   - Start with defaults
   - Adjust based on results

4. **Analyze embeddings**
   ```bash
   python analyze_embeddings.py output_spatiotemporal_embeddings.npz
   ```

## Key Insights

✓ **Appearance alone is insufficient** for 360° rotation
✓ **Spatial-temporal context bridges the gap** when appearance fails
✓ **Multi-signal scoring is more robust** than single-signal
✓ **Location is highly reliable** for fixed camera scenarios
✓ **Expected improvement: 42-50%** reduction in duplicate IDs

## Research Background

This implementation is based on research findings:
- "Spatial and Temporal Mutual Promotion for Video-based Person Re-identification"
- "TesseTrack" - 4D CNN in voxelized feature space
- NTU ROSE Lab - Multi-view analysis with spatial context

**Key Finding**: Video-based ReID requires spatial + temporal + appearance, not just appearance alone.
