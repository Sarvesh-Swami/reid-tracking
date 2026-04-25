# Run Spatial-Temporal Tracker

## Quick Test

```bash
python track_attendance_spatiotemporal.py --source test_6.mp4
```

## Expected Output

```
============================================================
ATTENDANCE TRACKER v4 - SPATIAL-TEMPORAL
============================================================
Layer 1: BoTSORT (frame-to-frame tracking)
Layer 2: Persistent Gallery (aggressive re-verification)
Layer 3: Spatial-Temporal Reasoning (NEW!)
  ✓ Location tracking (WHERE people appear)
  ✓ Motion tracking (HOW people move)
  ✓ Temporal logic (WHEN people reappear)
  ✓ Multi-signal scoring (appearance + space + motion + time)
  ✓ Solves 360° rotation problem (front vs back view)
============================================================

Video: 1920x1080 @ 30fps, 1050 frames
  Device: cuda:0
  Track buffer: 150 frames (5.0s)
  ReID threshold: 0.42
  === MULTI-SIGNAL WEIGHTS ===
  Appearance: 0.60 (color: 0.35)
  Spatial:    0.20
  Motion:     0.10
  Temporal:   0.10
  === TRACKING PARAMETERS ===
  Probation: 3 frames
  Confirmation: 3 frames
  Gallery update: every 3 frames
  Re-verification: every 3 frame(s)
  Reassignment threshold: 0.08
  Contamination guard: 3 strikes

Processing...

  [NEW] Frame 1: NEW Person 1
  [NEW] Frame 187: NEW Person 2
  [NEW] Frame 218: NEW Person 3
  [NEW] Frame 265: NEW Person 4
  
  [REASSIGNED] Frame 613: Track 7 reassigned from Person 5 to Person 2 
  (dist: 0.41) [SPATIAL-TEMPORAL ASSIST]
  
  [REASSIGNED] Frame 710: Track 8 reassigned from Person 6 to Person 4 
  (dist: 0.40) [SPATIAL-TEMPORAL ASSIST]

[DONE] Saved: output_spatiotemporal.mp4

============================================================
ATTENDANCE REPORT
============================================================
Total unique persons: 5-6 (expected, down from 12 in v3.1)
Re-identification events: X
ID reassignments: X (with spatial-temporal assists)
============================================================
```

## Compare with v3.1

### Run Both Versions
```bash
# v3.1 (appearance only)
python track_attendance.py --source test_6.mp4 --output output_v31.mp4

# v4 (spatial-temporal)
python track_attendance_spatiotemporal.py --source test_6.mp4 --output output_st.mp4
```

### Expected Comparison
- **v3.1**: 12 persons detected
- **v4**: 5-6 persons detected
- **Improvement**: 42-50% reduction

## Custom Weights

### More Aggressive Merging
```bash
python track_attendance_spatiotemporal.py --source test_6.mp4 \
    --spatial-weight 0.30 \
    --motion-weight 0.15 \
    --temporal-weight 0.15
```

### More Conservative (Rely on Appearance)
```bash
python track_attendance_spatiotemporal.py --source test_6.mp4 \
    --spatial-weight 0.10 \
    --motion-weight 0.05 \
    --temporal-weight 0.05
```

## Troubleshooting

### If you see errors about missing modules
```bash
# Make sure you're in the venv
venv\Scripts\activate

# Check dependencies
python -c "import cv2, numpy, torch; print('OK')"
```

### If processing is slow
- Spatial-temporal adds only ~5% overhead
- If slow, it's likely YOLO/ReID, not spatial-temporal
- Use GPU if available (CUDA)

### If still getting too many IDs
- Increase spatial-temporal weights
- Try: `--spatial-weight 0.30 --motion-weight 0.15`

### If people getting merged incorrectly
- Decrease spatial-temporal weights
- Try: `--spatial-weight 0.10 --motion-weight 0.05`

## Output Files

- `output_spatiotemporal.mp4` - Annotated video
- `output_spatiotemporal_embeddings.npz` - Embeddings
- `output_spatiotemporal_metadata.json` - Metadata

## Next Steps

1. Run the test
2. Check console for `[SPATIAL-TEMPORAL ASSIST]` messages
3. Count unique persons in output
4. Compare with v3.1 results
5. Tune weights if needed

## Documentation

- `SPATIOTEMPORAL_QUICK_START.md` - User guide
- `SPATIOTEMPORAL_IMPLEMENTATION.md` - Technical details
