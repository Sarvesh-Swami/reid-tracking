# Testing Checklist - Attendance Tracker v3

## Pre-Test Setup

- [ ] Virtual environment activated: `venv\Scripts\activate`
- [ ] All dependencies installed
- [ ] test_6.mp4 video file available
- [ ] Sufficient disk space for output video

## Run Test

```bash
python track_attendance.py --source test_6.mp4 --output output_v3.mp4
```

## During Processing - Watch Console

### Expected Messages (Good Signs)

- [ ] See "ATTENDANCE TRACKER v3" header
- [ ] See "Re-verification: every 1 frame(s)"
- [ ] See "Reassignment threshold: 0.15"
- [ ] See "[NEW] Frame X: NEW Person 1, 2, 3, 4" (only 4 new persons)
- [ ] See "[REASSIGNED]" messages (shows corrections working)
- [ ] See "[CONFIRMED]" messages (re-identification working)

### Warning Messages (May Appear)

- [ ] "[VERIFY FAIL]" - Normal, leads to reassignment
- [ ] "[IDENTITY LOST]" - Track unmapped, will be remapped
- [ ] "[GALLERY REJECT]" - Contamination guard working

### Bad Signs (Should NOT See)

- [ ] "[NEW] Frame X: NEW Person 5, 6, 7, 8, 9" - Too many persons
- [ ] No "[REASSIGNED]" messages - Reassignment not working
- [ ] Many "[GALLERY REJECT]" - Threshold may be too strict

## After Processing - Check Console Output

### Final Report

- [ ] "Total unique persons: 4" (NOT 9)
- [ ] "Re-identification events: X" (should be >0)
- [ ] "ID reassignments: X" (should be >0, shows corrections)

### Person Details

- [ ] Person 1: Single continuous or multiple appearances
- [ ] Person 2: Single continuous or multiple appearances
- [ ] Person 3: Single continuous or multiple appearances
- [ ] Person 4: Single continuous or multiple appearances
- [ ] NO Person 5, 6, 7, 8, 9 in the list

### Reassignment Events

- [ ] See "ID Reassignments (BoTSORT swaps corrected):"
- [ ] See reassignment details with frame numbers
- [ ] Reassignments make sense (e.g., Track 7 from Person 3 → Person 1)

## Watch Output Video

### Load output_v3.mp4

- [ ] Video plays correctly
- [ ] Bounding boxes visible
- [ ] Person IDs displayed on boxes

### Check ID Stability

- [ ] Person 1 (woman in pink): Same ID throughout
- [ ] Person 2: Same ID throughout
- [ ] Person 3 (man in black): Same ID throughout
- [ ] Person 4: Same ID throughout

### Check Profile Changes

- [ ] Person 1 back profile: Same ID
- [ ] Person 1 front profile: Same ID (not different)
- [ ] Person 2 back profile: Same ID
- [ ] Person 2 front profile: Same ID
- [ ] Person 3 back profile: Same ID
- [ ] Person 3 front profile: Same ID
- [ ] Person 4 back profile: Same ID
- [ ] Person 4 front profile: Same ID

### Check Occlusions

- [ ] When people overlap: IDs stay correct
- [ ] After overlap: IDs don't swap
- [ ] Person exits and returns: Same ID

### Check "RETURNED!" Labels

- [ ] See "RETURNED!" label when person re-enters
- [ ] Label appears on correct person
- [ ] Label disappears after a few seconds

## Compare with v2 (Optional)

### Run v2 for comparison

```bash
# If you have v2 saved
python track_attendance_v2.py --source test_6.mp4 --output output_v2.mp4
```

### Compare Results

- [ ] v2: 9 unique persons vs v3: 4 unique persons
- [ ] v2: Multiple IDs per person vs v3: Single ID per person
- [ ] v3: Has reassignment events, v2: Doesn't

## Performance Check

### Processing Speed

- [ ] Processing completed successfully
- [ ] Time taken: _____ seconds
- [ ] FPS: _____ (should be reasonable)

### Resource Usage

- [ ] CPU usage: Acceptable
- [ ] GPU usage: Acceptable (if using CUDA)
- [ ] Memory usage: No crashes

## Validation Tests

### Test 1: Unique Person Count

```
Expected: 4
Actual: _____
Status: [ ] PASS [ ] FAIL
```

### Test 2: ID Consistency

```
Person 1 IDs seen: _____
Expected: 1 only
Status: [ ] PASS [ ] FAIL

Person 2 IDs seen: _____
Expected: 2 only
Status: [ ] PASS [ ] FAIL

Person 3 IDs seen: _____
Expected: 3 only
Status: [ ] PASS [ ] FAIL

Person 4 IDs seen: _____
Expected: 4 only
Status: [ ] PASS [ ] FAIL
```

### Test 3: Reassignments

```
Number of reassignments: _____
Expected: >0 (shows corrections working)
Status: [ ] PASS [ ] FAIL
```

### Test 4: Re-identifications

```
Number of re-ID events: _____
Expected: >0 (shows people returning)
Status: [ ] PASS [ ] FAIL
```

## Troubleshooting

### If "Total unique persons: 9" (FAIL)

Try:
```bash
python track_attendance.py --source test_6.mp4 --reid-threshold 0.35 --color-weight 0.65
```

Then recheck:
- [ ] Unique persons now 4?
- [ ] If still failing, decrease reid-threshold to 0.30

### If Too Many Reassignments

Edit `track_attendance.py` line ~184:
```python
self.reassignment_threshold = 0.20  # Increase from 0.15
```

Then rerun:
- [ ] Reassignments reduced?
- [ ] IDs still correct?

### If Processing Too Slow

Edit `track_attendance.py` line ~183:
```python
self.verify_interval = 2  # Check every 2 frames
```

Then rerun:
- [ ] Processing faster?
- [ ] Accuracy still good?

### If IDs Still Wrong

Try stricter settings:
```bash
python track_attendance.py --source test_6.mp4 --reid-threshold 0.30 --color-weight 0.70
```

And edit line ~184:
```python
self.reassignment_threshold = 0.10  # Easier reassignment
```

## Final Validation

### Overall Assessment

- [ ] Total unique persons: 4 ✓
- [ ] Each person has consistent ID ✓
- [ ] Front/back profiles matched correctly ✓
- [ ] Reassignments show corrections ✓
- [ ] No gallery contamination ✓
- [ ] Output video looks correct ✓

### Test Result

```
[ ] PASS - All checks passed, system working correctly
[ ] PARTIAL - Some issues, needs tuning
[ ] FAIL - Major issues, needs investigation
```

## Next Steps

### If PASS

- [ ] Deploy to production videos
- [ ] Monitor performance on different videos
- [ ] Document any needed tuning per video type

### If PARTIAL

- [ ] Apply tuning recommendations above
- [ ] Retest with adjusted parameters
- [ ] Document optimal settings

### If FAIL

- [ ] Check console output for errors
- [ ] Verify video format is supported
- [ ] Check dependencies are installed correctly
- [ ] Review REID_ISSUE_ANALYSIS.md for understanding
- [ ] Try with different video first

## Documentation Review

Before deploying, review:

- [ ] V3_SUMMARY.md - Understand changes
- [ ] RUN_V3_GUIDE.md - Usage instructions
- [ ] FIXES_V3_APPLIED.md - Technical details
- [ ] QUICK_REFERENCE.md - Quick help
- [ ] BEFORE_AFTER_COMPARISON.md - Expected improvements

## Sign-off

```
Tester: _________________
Date: ___________________
Result: [ ] PASS [ ] PARTIAL [ ] FAIL
Notes: _________________________________________________
_______________________________________________________
_______________________________________________________
```

## Support

If you encounter issues:
1. Check console output for specific errors
2. Review troubleshooting section above
3. Check RUN_V3_GUIDE.md for detailed help
4. Verify all dependencies installed correctly
5. Try with default parameters first

## Success Criteria Summary

✅ 4 unique persons (not 9)
✅ Consistent IDs per person
✅ Reassignments logged
✅ Re-identifications working
✅ Output video correct
✅ No major errors

If all criteria met: **v3 is working correctly!**
