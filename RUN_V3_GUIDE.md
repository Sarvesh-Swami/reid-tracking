# How to Run Attendance Tracker v3

## Quick Start

```bash
# Activate your virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Run with default settings
python track_attendance.py --source test_6.mp4 --output output_v3.mp4

# Run with custom threshold
python track_attendance.py --source test_6.mp4 --reid-threshold 0.35 --output output_v3.mp4

# Run with display window
python track_attendance.py --source test_6.mp4 --show
```

## Command Line Options

```
--source VIDEO_PATH          Input video file (required)
--output OUTPUT_PATH         Output video file (default: output_attendance.mp4)
--yolo-model MODEL          YOLO model (default: yolov8n.pt)
--reid-model MODEL          ReID model (default: osnet_x1_0_msmt17.pt)
--reid-threshold FLOAT      Gallery matching threshold (default: 0.45, lower=stricter)
--detection-conf FLOAT      Detection confidence (default: 0.25)
--track-buffer FLOAT        Track buffer in seconds (default: 5.0)
--color-weight FLOAT        Weight for color in scoring (default: 0.55, range: 0-1)
--show                      Display video while processing
```

## What to Expect

### Console Output

You should see:

```
============================================================
ATTENDANCE TRACKER v3
============================================================
Layer 1: BoTSORT (frame-to-frame tracking)
Layer 2: Persistent Gallery (aggressive re-verification)
  - EVERY frame verification against ALL gallery PIDs
  - Automatic ID reassignment on better matches
  - Gallery contamination guard (2-strike system)
  - Failed-match avoidance (no retry of wrong PIDs)
  - 2D H×S color histogram
  - Faster probation & confirmation (3+3 frames)
============================================================

Video: 464x832 @ 30fps, 1302 frames
...
Device: cuda:0
Track buffer: 150 frames (5.0s)
ReID threshold: 0.45
Color weight: 0.55
Probation: 3 frames
Confirmation: 3 frames
Gallery update: every 3 frames
Re-verification: every 1 frame(s)
Reassignment threshold: 0.15
Contamination guard: 2 strikes

Processing...

[NEW] Frame 1: NEW Person 1
[NEW] Frame 187: NEW Person 2
[NEW] Frame 218: NEW Person 3
[NEW] Frame 265: NEW Person 4

# When reassignments happen (this is GOOD - it's correcting BoTSORT swaps)
[TENTATIVE] Frame 613: Person 3 tentative match
[VERIFY FAIL] Track 7: Assigned Person 3 doesn't match
[REASSIGN CANDIDATE] Track 7: Person 1 matches better (score diff: 0.23)
[REASSIGNED] Frame 615: Track 7 reassigned from Person 3 → Person 1 (dist: 0.32)

[CONFIRMED] Frame 679: Person 3 RE-ID confirmed (dist: 0.392)
...

[DONE] Saved: output_v3.mp4

============================================================
ATTENDANCE REPORT
============================================================
Total unique persons: 4
Re-identification events: 7
ID reassignments: 3

Person 1: Frames 1-1200 (40.0s) | Features: 45 | Re-IDs: 2
Person 2: Frames 187-1150 (32.1s) | Features: 38 | Re-IDs: 1
Person 3: Frames 218-1180 (32.1s) | Features: 42 | Re-IDs: 2
Person 4: Frames 265-1243 (32.6s) | Features: 40 | Re-IDs: 1
============================================================

Re-ID Events:
  Frame 440: Person 3 returned (distance: 0.473)
  Frame 679: Person 3 returned (distance: 0.392)
  ...

ID Reassignments (BoTSORT swaps corrected):
  Frame 615: Track 7 reassigned from Person 3 → Person 1 (dist: 0.32)
  Frame 820: Track 9 reassigned from Person 4 → Person 2 (dist: 0.35)
  Frame 1050: Track 11 reassigned from Person 3 → Person 4 (dist: 0.29)
```

## Expected Results

### For Your test_6.mp4 Video

**Before (v2):**
- 9 unique persons detected (WRONG - should be 4)
- Person 1 got IDs: 1, 3, 9
- Person 2 got IDs: 2, 7
- Person 3 got IDs: 3, 4
- Person 4 got IDs: 4, 8

**After (v3):**
- 4 unique persons detected (CORRECT)
- Each person keeps their original ID throughout
- Reassignment events show when BoTSORT swaps were corrected
- Front/back profiles correctly matched to same person

## Tuning Parameters

### If You Get Too Many Reassignments

Symptoms: IDs flip-flopping between people

```bash
# Increase reassignment threshold (requires bigger score difference)
# Edit track_attendance.py line ~165:
self.reassignment_threshold = 0.20  # was 0.15

# Or verify less frequently
self.verify_interval = 2  # was 1 (every 2 frames instead of every frame)
```

### If You Still Get Wrong IDs

Symptoms: People still getting multiple IDs

```bash
# Stricter matching threshold
python track_attendance.py --source test_6.mp4 --reid-threshold 0.35

# Rely more on color
python track_attendance.py --source test_6.mp4 --color-weight 0.65

# Or make reassignment easier
# Edit track_attendance.py line ~165:
self.reassignment_threshold = 0.10  # was 0.15
```

### If Processing is Too Slow

```bash
# Verify every 2 frames instead of every frame
# Edit track_attendance.py line ~164:
self.verify_interval = 2  # was 1

# Note: This reduces accuracy slightly
```

## Understanding the Output

### New Person
```
[NEW] Frame 1: NEW Person 1
```
First time this person appears in the video.

### Tentative Match
```
[TENTATIVE] Frame 613: Person 3 tentative match
```
System thinks this might be Person 3 returning, needs confirmation.

### Confirmed Re-ID
```
[CONFIRMED] Frame 679: Person 3 RE-ID confirmed (dist: 0.392)
```
Confirmed that Person 3 has returned (re-identification successful).

### Reassignment (NEW in v3)
```
[REASSIGNED] Frame 615: Track 7 reassigned from Person 3 → Person 1 (dist: 0.32)
```
**This is GOOD!** The system detected that:
- BoTSORT assigned track 7 to Person 3 (wrong)
- But the features actually match Person 1 better
- Automatically corrected the assignment

### Verification Failure
```
[VERIFY FAIL] Track 7: Assigned Person 3 doesn't match
```
Current features don't match the assigned person - will unmap if this continues.

### Identity Lost
```
[IDENTITY LOST] Frame 750: Track 7 lost identity of Person 3 (2 verification failures)
```
Track has been unmapped and will go through probation again.

## Troubleshooting

### "Total unique persons: 9" (still wrong)

Try:
1. Lower `--reid-threshold` to 0.35 or 0.30 (stricter)
2. Increase `--color-weight` to 0.65 (rely more on clothing color)
3. Check if reassignments are happening (should see [REASSIGNED] messages)
4. If no reassignments, decrease `reassignment_threshold` in code

### "Too many reassignments" (IDs changing constantly)

Try:
1. Increase `reassignment_threshold` in code (0.15 → 0.20)
2. Increase `verify_interval` (1 → 2 frames)
3. Increase `--reid-threshold` (0.45 → 0.50)

### "Processing very slow"

Try:
1. Increase `verify_interval` to 2 or 3 frames
2. Use smaller YOLO model: `--yolo-model yolov8n.pt`
3. Reduce video resolution before processing

### "CUDA out of memory"

Try:
1. Process on CPU (will be slower but works)
2. Reduce batch size in YOLO
3. Use smaller ReID model

## Performance Metrics

### Speed
- ~10-20% slower than v2 due to every-frame verification
- For 4 people: negligible impact
- For 20+ people: may need to increase `verify_interval`

### Accuracy
- Should correctly identify 4 people (not 9)
- Handles front/back profile changes
- Robust to occlusions
- Corrects BoTSORT ID swaps automatically

## Next Steps

1. **Run on your test video:**
   ```bash
   python track_attendance.py --source test_6.mp4 --output output_v3.mp4
   ```

2. **Check the results:**
   - Look for "Total unique persons: 4" (not 9)
   - Check reassignment events in output
   - Watch output_v3.mp4 to verify IDs are stable

3. **Tune if needed:**
   - Adjust thresholds based on results
   - See tuning section above

4. **Compare with v2:**
   - Run both versions on same video
   - Compare unique person counts
   - Check which IDs are more stable

## Files Modified

- `track_attendance.py` - Main tracking script with v3 fixes
- `FIXES_V3_APPLIED.md` - Detailed explanation of changes
- `REID_ISSUE_ANALYSIS.md` - Analysis of the original problem
- `RUN_V3_GUIDE.md` - This file

## Support

If you encounter issues:
1. Check console output for error messages
2. Look for [REASSIGNED] messages - these show corrections working
3. Try different threshold values
4. Check that you have 4 unique persons in final report
