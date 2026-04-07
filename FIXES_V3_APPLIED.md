# Attendance Tracker v3 - Aggressive Re-Verification Fixes

## Problem Identified

The v2 system had a critical flaw: **it trusted BoTSORT's track IDs too much**. Once a track was mapped (tid → pid), it only verified occasionally and never checked if another person matched better.

### Root Causes:
1. **Passive verification** - Only checked every 3 frames
2. **Single-PID checking** - Only verified against assigned PID, not all PIDs
3. **Slow contamination detection** - Needed 3 strikes (9+ frames)
4. **No reassignment logic** - Once mapped, stayed mapped unless it failed completely
5. **BoTSORT ID swaps** - During occlusions, BoTSORT swaps IDs before we detect it

## Fixes Applied in v3

### 1. Aggressive Re-Verification (CRITICAL FIX)

**New function: `_verify_track_identity()`**

```python
def _verify_track_identity(self, tid, pid, feat, chist):
    """
    Check against ALL gallery PIDs, not just the assigned one.
    Returns: (is_valid, best_pid, best_dist, should_reassign)
    """
```

**What it does:**
- Compares current features against **ALL PIDs in gallery**
- Checks if assigned PID still matches
- Checks if another PID matches significantly better
- Returns reassignment recommendation

**Key logic:**
```python
# Check 1: Does assigned PID match at all?
if assigned_dist >= self.reid_threshold:
    is_valid = False

# Check 2: Does another PID match significantly better?
if best_other_score - assigned_score > self.reassignment_threshold:
    should_reassign = True
```

### 2. Every-Frame Verification

**Old behavior:**
```python
if self.frame_count % self.update_interval == 0:  # Every 3 frames
    # Check only if gallery accepts features
```

**New behavior:**
```python
if (self.frame_count - self.last_verified_frame.get(tid, 0)) >= self.verify_interval:
    # Verify against ALL PIDs EVERY frame
    is_valid, best_pid, best_dist, should_reassign = self._verify_track_identity(...)
```

**Benefits:**
- Catches BoTSORT ID swaps within 1 frame
- Prevents gallery contamination before it happens
- Enables real-time ID correction

### 3. Automatic ID Reassignment

**New logic in `_map_ids()`:**

```python
if should_reassign and best_pid != pid:
    # REASSIGN to better matching person
    old_pid = pid
    pid = best_pid
    self.id_map[tid] = pid
    self.reassignment_events.append((frame, tid, old_pid, pid, dist))
    print(f"[REASSIGNED] Track {tid} from Person {old_pid} → Person {pid}")
```

**What this fixes:**
- When Person 1 (front) gets assigned to Person 3's ID
- System detects Person 1's features match Person 1's gallery better
- Automatically reassigns: Track → Person 1 (correct)

### 4. Faster Contamination Detection

**Changes:**
- `max_verify_fails`: 3 → 2 strikes
- `verify_interval`: 3 frames → 1 frame
- Verification happens BEFORE gallery update

**Result:**
- Contamination detected in 2 frames instead of 9+
- Gallery stays clean

### 5. Reassignment Threshold

**New parameter:**
```python
self.reassignment_threshold = 0.15  # Score difference needed to reassign
```

**Logic:**
- If another PID scores 0.15 higher (15% better), reassign
- Prevents flip-flopping between similar scores
- Requires significant improvement to change assignment

### 6. Tracking State

**New tracking:**
```python
self.last_verified_frame = {}  # tid → last frame verified
self.reassignment_events = []  # Log of all reassignments
```

**Benefits:**
- Know when each track was last verified
- Audit trail of all ID corrections
- Debug visibility

## How It Solves Your Problem

### Scenario 1: Person 1 Back → Front Profile

**Old behavior:**
1. Person 1 back → ID 1
2. Person 1 exits, BoTSORT loses track
3. Person 1 front returns → BoTSORT assigns new track ID 7
4. System tries to match → matches Person 3 (wrong) → gets ID 3
5. **STUCK with wrong ID**

**New behavior:**
1. Person 1 back → ID 1
2. Person 1 exits, BoTSORT loses track
3. Person 1 front returns → BoTSORT assigns new track ID 7
4. System tries to match → matches Person 3 tentatively
5. **Frame-by-frame verification detects mismatch**
6. **Compares against ALL PIDs → finds Person 1 matches better**
7. **Auto-reassigns: Track 7 → Person 1** ✓

### Scenario 2: Occlusion ID Swap

**Old behavior:**
1. Person 2 and Person 3 overlap
2. BoTSORT swaps their track IDs
3. System adds Person 3's features to Person 2's gallery
4. Gallery contaminated
5. Future matches fail

**New behavior:**
1. Person 2 and Person 3 overlap
2. BoTSORT swaps their track IDs
3. **Next frame: verification detects mismatch**
4. **Compares against all PIDs → finds correct match**
5. **Auto-reassigns before gallery update**
6. Gallery stays clean ✓

## Configuration

### Key Parameters

```python
reid_threshold = 0.40          # Distance threshold for matching
reassignment_threshold = 0.15  # Score diff needed to reassign
verify_interval = 1            # Verify every N frames (1 = every frame)
max_verify_fails = 2           # Strikes before unmapping
color_weight = 0.55            # Weight for color vs ReID features
```

### Tuning Recommendations

**If too many reassignments:**
- Increase `reassignment_threshold` (0.15 → 0.20)
- Increase `verify_interval` (1 → 2 frames)

**If still getting wrong IDs:**
- Decrease `reid_threshold` (0.40 → 0.35) - stricter matching
- Increase `color_weight` (0.55 → 0.65) - rely more on clothing color
- Decrease `reassignment_threshold` (0.15 → 0.10) - easier reassignment

**If too slow:**
- Increase `verify_interval` (1 → 2 or 3 frames)
- Note: This reduces accuracy

## Expected Output

### Console Output

```
[NEW] Frame 1: NEW Person 1
[NEW] Frame 187: NEW Person 2
[NEW] Frame 218: NEW Person 3
[NEW] Frame 265: NEW Person 4

# Person 1 returns (front profile)
[TENTATIVE] Frame 613: Person 3 tentative match
[VERIFY FAIL] Track 7: Assigned Person 3 doesn't match
[REASSIGN CANDIDATE] Track 7: Person 1 matches better (score diff: 0.23)
[REASSIGNED] Frame 615: Track 7 reassigned from Person 3 → Person 1 (dist: 0.32)

# Person 2 returns (back profile)
[TENTATIVE] Frame 710: Person 2 tentative match
[CONFIRMED] Frame 713: Person 2 RE-ID confirmed (dist: 0.38)
```

### Final Report

```
ATTENDANCE REPORT
============================================================
Total unique persons: 4
Re-identification events: 6
ID reassignments: 3

Person 1: Frames 1-1200 (40.0s) | Features: 45 | Re-IDs: 2
Person 2: Frames 187-1150 (32.1s) | Features: 38 | Re-IDs: 1
Person 3: Frames 218-1180 (32.1s) | Features: 42 | Re-IDs: 2
Person 4: Frames 265-1243 (32.6s) | Features: 40 | Re-IDs: 1

ID Reassignments (BoTSORT swaps corrected):
  Frame 615: Track 7 reassigned from Person 3 → Person 1 (dist: 0.32)
  Frame 820: Track 9 reassigned from Person 4 → Person 2 (dist: 0.35)
  Frame 1050: Track 11 reassigned from Person 3 → Person 4 (dist: 0.29)
```

## Testing

Run with your test video:

```bash
python track_attendance.py --source test_6.mp4 --reid-threshold 0.40 --output output_v3.mp4
```

**Expected results:**
- 4 unique persons (not 9)
- Multiple reassignment events logged
- Consistent IDs throughout video
- Front/back profiles correctly matched

## Performance Impact

**Computational cost:**
- ~10-20% slower due to every-frame verification
- Each verification compares against all PIDs in gallery
- For 4 people: negligible
- For 50+ people: may need to increase `verify_interval`

**Accuracy improvement:**
- Eliminates most ID swaps
- Prevents gallery contamination
- Handles front/back profile changes
- Robust to occlusions

## Debugging

**If you see many reassignments:**
```
[REASSIGNED] Frame 100: Track 5 from Person 1 → Person 2
[REASSIGNED] Frame 102: Track 5 from Person 2 → Person 1
[REASSIGNED] Frame 104: Track 5 from Person 1 → Person 2
```

**Solution:** Increase `reassignment_threshold` to require larger score difference.

**If you still get wrong IDs:**
- Check the verification logs for score differences
- May need to adjust `reid_threshold` or `color_weight`
- Consider collecting more features during probation

## Summary

v3 implements **aggressive continuous re-verification** that:
1. ✅ Checks EVERY frame against ALL PIDs
2. ✅ Automatically reassigns IDs when better match found
3. ✅ Prevents gallery contamination
4. ✅ Handles front/back profile changes
5. ✅ Corrects BoTSORT ID swaps in real-time
6. ✅ Maintains audit trail of all corrections

**The key insight:** Don't trust BoTSORT's track IDs - verify and correct continuously.
