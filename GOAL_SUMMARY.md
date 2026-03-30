# Project Goal: Person Re-Identification System

## 🎯 What We're Trying to Achieve

Build a **person tracking system** that maintains **consistent, persistent IDs** for individuals throughout a video, similar to attendance/counting systems used by companies.

## 📋 Requirements

### Core Functionality
1. **Assign unique ID** to each person when first detected
2. **Maintain same ID** when person:
   - Goes behind another person (brief occlusion of 1-2 seconds)
   - Leaves the frame completely
   - Returns after 10+ seconds or even minutes
   - Turns around or changes pose/viewing angle
   - Moves to different locations in frame

### Use Case
**Attendance/Counting System** - Like what companies use where:
- People walk through a camera view naturally
- System automatically counts unique individuals
- Same person returning later gets the same ID (not counted as new person)
- Works in real-time without requiring people to stop and pose for the camera
- No manual intervention needed

## 🎬 Specific Scenarios to Handle

### Scenario 1: Brief Occlusion
```
Frame 1:   [Person A] [Person B]     → IDs: 1, 2
Frame 2:   [Person A walks in front of Person B]
Frame 3:   [Person A] [Person B]     → IDs: 1, 2 (SAME IDs maintained)
```

### Scenario 2: Person Leaves and Returns
```
Frame 1:     [Person A]              → ID: 1
Frame 100:   [Person A leaves]
Frame 200:   [Person B appears]      → ID: 2
Frame 300:   [Person A returns]      → ID: 1 (SAME ID reassigned)
```

### Scenario 3: Pose/Appearance Changes
```
Frame 1:   [Person A - front view]   → ID: 1
Frame 50:  [Person A - turns around]
Frame 51:  [Person A - back view]    → ID: 1 (SAME ID maintained)
```

## ✅ Success Criteria

The system is successful if:
1. **Each unique person** gets exactly **one ID** throughout the entire video
2. **IDs persist** across brief occlusions (1-2 seconds)
3. **IDs are reassigned** when person returns after leaving (10+ seconds)
4. **IDs remain stable** despite pose/appearance changes
5. Works in **real-time** or near real-time
6. Requires **no manual intervention** or person cooperation

## 📹 Video Context

- **Video file**: `test_3.mp4`
- **Content**: Multiple people moving around
- **Challenges**: 
  - People walking in front of each other
  - People entering and exiting frame
  - Various poses and viewing angles
  - Natural movement patterns

## 🎯 Expected Output

For a video with 5 unique people:
- **Total IDs created**: 5 (one per person)
- **ID consistency**: Same person always has same ID
- **No ID switches**: Person doesn't get new ID during occlusions
- **Re-identification**: Person returning gets original ID back

## 🔧 Technical Approach

Using **BoxMOT** tracking library with:
- **Object Detection**: YOLOv8 to detect people in each frame
- **Tracking**: Multi-object tracker to maintain IDs across frames
- **Re-Identification (ReID)**: Deep learning features to recognize same person by appearance
- **Persistent Database**: Store features of all persons ever seen for long-term re-ID

## 📊 Current Status

**Repository**: BoxMOT (yolo_tracking) - https://github.com/mikel-brostrom/yolo_tracking
- Contains 5 state-of-the-art tracking methods
- Supports ReID-based appearance matching
- Integrates with YOLOv8 detection

**Challenge**: Standard trackers are designed for **continuous short-term tracking**, not **long-term re-identification** with persistent IDs across gaps.

## 🎯 What Needs to Work

1. **Detection**: YOLO must detect people even during partial occlusions
2. **Matching**: Tracker must correctly match detections to existing tracks
3. **Persistence**: IDs must survive gaps when person isn't detected
4. **Re-identification**: System must recognize returning persons and reassign original IDs
5. **Robustness**: Handle appearance changes, pose variations, lighting changes

## 💡 Key Insight

This is essentially an **attendance system** problem, not just a tracking problem:
- **Tracking**: Maintain IDs frame-to-frame (short-term)
- **Attendance**: Recognize unique individuals across entire session (long-term)

We need both capabilities combined.
