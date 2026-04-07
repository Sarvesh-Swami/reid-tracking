# Fixes Applied to BoxMOT

## Issue 1: AttributeError in reid_model_factory.py

**Error**: `AttributeError: 'str' object has no attribute 'name'`

**Location**: `boxmot/appearance/reid_model_factory.py` line 96

**Root Cause**: The `get_model_name()` function expected a Path object with `.name` attribute, but sometimes received a string.

**Fix Applied**:
```python
# Before:
def get_model_name(model):
    for x in __model_types:
        if x in model.name:
            return x
    return None

# After:
def get_model_name(model):
    # Handle both Path objects and strings
    model_str = str(model.name) if hasattr(model, 'name') else str(model)
    for x in __model_types:
        if x in model_str:
            return x
    return None
```

## Issue 2: AttributeError in reid_multibackend.py

**Error**: `AttributeError: 'str' object has no attribute 'suffix'`

**Location**: `boxmot/appearance/reid_multibackend.py` line 74

**Root Cause**: The code expected `w` to be a Path object with `.suffix` attribute, but it was a string.

**Fix Applied**:
```python
# In ReIDDetectMultiBackend.__init__():
# Added after line 46:
w = weights[0] if isinstance(weights, list) else weights
# Ensure w is a Path object
w = Path(w) if not isinstance(w, Path) else w
```

## Testing

Both fixes ensure that the ReID backend can handle model weights passed as either:
- String: `"osnet_x1_0_msmt17.pt"`
- Path object: `Path("osnet_x1_0_msmt17.pt")`

## How to Run

Now you can run the 3-pass pipeline:

```bash
# Make sure venv is activated
(venv) C:\Users\...\> python run_persistent_reid.py --video test_6.mp4
```

Or run steps individually:

```bash
(venv) C:\Users\...\> python step1_extract_embeddings.py --video test_6.mp4
(venv) C:\Users\...\> python step2_cluster_tracks.py --input test_6_embeddings.pkl
(venv) C:\Users\...\> python step3_gallery_tracking.py --video test_6.mp4 --gallery test_6_gallery.pkl
```

## Files Modified

1. `boxmot/appearance/reid_model_factory.py` - Fixed `get_model_name()` to handle strings
2. `boxmot/appearance/reid_multibackend.py` - Added Path conversion in `__init__()`

## Files Created

1. `step1_extract_embeddings.py` - Extract embeddings from video
2. `step2_cluster_tracks.py` - Cluster tracks to identify unique persons
3. `step3_gallery_tracking.py` - Re-track with gallery for persistent IDs
4. `run_persistent_reid.py` - Master script to run all 3 steps
5. `PERSISTENT_REID_README.md` - Complete documentation
