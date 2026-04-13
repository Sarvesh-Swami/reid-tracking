# Person Re-Identification & Tracking System

A research and experimentation project for building a **persistent person tracking system** that maintains consistent IDs for individuals throughout a video — designed for attendance and counting use cases.

---

## What This Project Does

The system detects people in video frames using YOLOv8, tracks them frame-to-frame using multi-object trackers, and attempts to re-identify the same person when they return after leaving the frame. The goal is that each unique person gets exactly one ID for the entire session.

**Core challenge being solved:** Standard trackers assign a new ID every time a person re-enters the frame. This project explores multiple approaches to make IDs persistent.

---

## Project Structure

```
reid-tracking/
├── boxmot/                        # Core tracking library (BoxMOT)
│   ├── __init__.py                # Exports all trackers + version
│   ├── tracker_zoo.py             # Factory: creates any tracker from config
│   ├── trackers/
│   │   ├── deepocsort/            # DeepOCSORT tracker (motion + ReID)
│   │   │   ├── deep_ocsort.py     # Main tracker logic + persistent embedding storage
│   │   │   └── embedding.py       # Embedding extraction helper
│   │   ├── deepocsort_persistent/ # Experimental persistent variant of DeepOCSORT
│   │   │   └── persistent_deep_ocsort.py
│   │   ├── strongsort/            # StrongSORT tracker (motion + ReID)
│   │   │   └── strong_sort.py
│   │   ├── strongsort_persistent.py  # StrongSORT extended with persistent gallery
│   │   ├── botsort/               # BoTSORT tracker (motion + ReID + camera compensation)
│   │   │   └── bot_sort.py
│   │   ├── bytetrack/             # ByteTrack tracker (motion only, fast)
│   │   │   └── byte_tracker.py
│   │   └── ocsort/                # OCSORT tracker (motion only, lightweight)
│   │       └── ocsort.py
│   ├── appearance/
│   │   ├── reid_multibackend.py   # Multi-backend ReID inference (PT/ONNX/TRT/OV/TFLite)
│   │   ├── reid_model_factory.py  # Model URL registry and loader
│   │   └── backbones/             # OSNet, MobileNet, ResNet, LightMBN architectures
│   ├── motion/
│   │   └── kalman_filter.py       # Kalman filter for motion prediction
│   ├── utils/
│   │   ├── matching.py            # IoU/embedding distance, linear assignment, gating
│   │   ├── association.py         # Batch IoU (IoU/GIoU/DIoU/CIoU), speed consistency
│   │   ├── persistent_reid_matching.py  # Persistent gallery for deleted track re-ID
│   │   ├── cmc.py                 # Camera motion compensation
│   │   ├── gmc.py                 # Global motion compensation
│   │   ├── preprocessing.py       # Image preprocessing utilities
│   │   └── torch_utils.py         # Device selection helper
│   └── configs/
│       ├── deepocsort.yaml        # DeepOCSORT hyperparameters
│       ├── strongsort.yaml        # StrongSORT hyperparameters
│       ├── botsort.yaml           # BoTSORT hyperparameters
│       ├── bytetrack.yaml         # ByteTrack hyperparameters
│       └── ocsort.yaml            # OCSORT hyperparameters
│
├── examples/
│   ├── track.py                   # Official BoxMOT tracking pipeline (CLI entry point)
│   ├── val.py                     # Benchmark evaluation on MOT16/17/20
│   ├── evolve.py                  # Hyperparameter evolution via Optuna
│   └── utils.py                   # MOT result writer utility
│
├── assets/
│   ├── images/                    # Demo GIFs showing tracking results
│   └── MOT17-mini/                # Mini MOT17 dataset for testing (7 sequences)
│       └── train/
│           └── MOT17-XX-FRCNN/
│               ├── MOT17-XX-FRCNN/  # Video frames (JPG)
│               ├── det/det.txt      # Detection annotations
│               ├── gt/gt.txt        # Ground truth annotations
│               └── seqinfo.ini      # Sequence metadata
│
├── ultimate_person_tracker.py     # Most complete system: detection + tracking + ReID + counting
├── robust_person_tracker.py       # Production-style system with modular classes
├── track_attendance.py            # Attendance-focused tracker (BoTSORT + persistent gallery)
├── track_embedding_persistent.py  # Embedding-only tracker (no BoxMOT tracker dependency)
├── track_persistent_reid.py       # StrongSORT + persistent gallery via StrongSORTPersistent
├── track_stable_ids.py            # DeepOCSORT with tuned params for stable IDs
├── track_simple_stable.py         # ByteTrack with lenient matching (simplest approach)
├── track_ultimate.py              # BoTSORT with extreme settings (all optimizations)
├── run_tracking.py                # Wrapper for PyTorch 2.6+ compatibility
├── patch_torch_load.py            # One-time patch for torch.load safe globals
├── debug_reid.py                  # Debug script to inspect ReID matching behavior
├── demo_embedding_persistence.py  # Demo showing embedding extraction and matching
├── test_persistent_simple.py      # Unit test for cosine distance / embedding logic
│
├── person_gallery.json            # Saved person embedding gallery (runtime artifact)
├── bus_person_gallery.json        # Gallery from bus video experiment
├── embeddings.json                # Embedding database from track_embedding_persistent.py
├── test_gallery.pkl               # Pickled gallery from track_persistent_reid.py
│
├── osnet_x1_0_msmt17.pt           # OSNet x1.0 ReID model weights
├── yolov8n.pt                     # YOLOv8 nano detection model
├── yolov8m.pt                     # YOLOv8 medium detection model
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── Dockerfile                     # Docker environment
├── .flake8                        # Linting config
├── .pre-commit-config.yaml        # Pre-commit hooks
│
└── docs/
    ├── PROJECT_SUMMARY.md         # Full problem description + all attempted solutions
    ├── GOAL_SUMMARY.md            # Requirements and success criteria
    ├── FINAL_SOLUTION.md          # Root cause analysis and recommendations
    ├── DIAGNOSIS.md               # Why persistent ReID fails + fixes
    ├── IMPLEMENTATION_COMPLETE.md # Technical details of persistent ReID implementation
    ├── PERSISTENT_REID_GUIDE.md   # Usage guide for persistent ReID system
    ├── EMBEDDING_PERSISTENCE_README.md  # Embedding-based tracking documentation
    ├── compare_tracking.md        # Standard vs persistent ReID comparison
    └── PROJECT_GUIDE.md           # Architecture overview and how to run
```

---

## File Descriptions

### Main Tracking Scripts

| File | What it does |
|------|-------------|
| `ultimate_person_tracker.py` | The most complete script. Combines YOLOv8 detection, DeepOCSORT tracking, OSNet ReID gallery with Hungarian assignment, directional line-crossing counting, and a HUD overlay. Best starting point for production use. |
| `robust_person_tracker.py` | Modular production-style system split into `Detector`, `Tracker`, `IdentityManager`, `PeopleCounter`, and `Visualizer` classes. Supports IN/OUT counting with deduplication. |
| `track_attendance.py` | Attendance-focused script using BoTSORT for short-term tracking and a persistent gallery for long-term re-identification. Fixed-camera use case. |
| `track_embedding_persistent.py` | Bypasses BoxMOT trackers entirely. Uses only YOLOv8 + OSNet embeddings stored in a JSON database. Assigns IDs purely by cosine similarity. |
| `track_persistent_reid.py` | Uses `StrongSORTPersistent` (custom class). Saves a gallery to disk (`.pkl`) so IDs persist across video sessions. |
| `track_stable_ids.py` | DeepOCSORT with tuned parameters (low IoU threshold, high max_age) for stable IDs during brief occlusions. |
| `track_simple_stable.py` | Simplest approach: ByteTrack with very lenient matching thresholds. No ReID, motion only. |
| `track_ultimate.py` | BoTSORT with extreme settings: YOLOv8x, OSNet x1.0, 300-frame track buffer, conf=0.15. Throws every optimization at the problem. |

### Utility / Support Scripts

| File | What it does |
|------|-------------|
| `run_tracking.py` | Wrapper around `examples/track.py` that applies a PyTorch 2.6+ compatibility patch before running. |
| `patch_torch_load.py` | One-time script to register ultralytics model classes as safe globals for `torch.load`. Run once if you get serialization errors. |
| `debug_reid.py` | Runs StrongSORTPersistent and prints detailed ReID matching logs to diagnose why IDs are or aren't being reused. |
| `demo_embedding_persistence.py` | Standalone demo showing how OSNet embeddings are extracted and compared. No video required. |
| `test_persistent_simple.py` | Minimal test that verifies cosine distance logic works correctly for same-person vs different-person embeddings. |

### BoxMOT Library (`boxmot/`)

| File | What it does |
|------|-------------|
| `__init__.py` | Exports all 5 trackers + `ReIDDetectMultiBackend` + version string. |
| `tracker_zoo.py` | `create_tracker()` factory — reads a YAML config and instantiates the correct tracker class. |
| `trackers/deepocsort/deep_ocsort.py` | DeepOCSORT implementation. Kalman filter + IoU + ReID embedding matching. Modified to store persistent embeddings for deleted tracks. |
| `trackers/deepocsort/embedding.py` | Extracts ReID embeddings from person crops for DeepOCSORT. |
| `trackers/deepocsort_persistent/persistent_deep_ocsort.py` | Experimental variant of DeepOCSORT with built-in persistent gallery. |
| `trackers/strongsort/strong_sort.py` | StrongSORT implementation. Uses `NearestNeighborDistanceMetric` for appearance matching. |
| `trackers/strongsort_persistent.py` | Extended StrongSORT that checks deleted track gallery before creating new IDs. Core of `track_persistent_reid.py`. |
| `trackers/botsort/bot_sort.py` | BoTSORT implementation. Adds camera motion compensation (CMC) on top of DeepOCSORT. |
| `trackers/bytetrack/byte_tracker.py` | ByteTrack implementation. Two-stage matching using high/low confidence detections. Motion only. |
| `trackers/ocsort/ocsort.py` | OCSORT implementation. Observation-centric re-update for better occlusion handling. Motion only. |
| `appearance/reid_multibackend.py` | Loads and runs ReID models across multiple backends (PyTorch, TorchScript, ONNX, OpenVINO, TensorRT, TFLite). Preprocesses crops to 256×128 and returns normalized feature vectors. |
| `appearance/reid_model_factory.py` | Registry of downloadable ReID model URLs. Used by `reid_multibackend.py` to auto-download weights. |
| `utils/matching.py` | Core matching math: IoU distance, embedding (cosine) distance, Mahalanobis gating, linear assignment (Hungarian via LAP). Also contains `NearestNeighborDistanceMetric` with persistent gallery extension. |
| `utils/association.py` | Batch IoU computation (IoU, GIoU, DIoU, CIoU variants). Speed and direction consistency checks for OCSORT. |
| `utils/persistent_reid_matching.py` | `PersistentNearestNeighborDistanceMetric` — extends the standard metric to keep features for deleted tracks and match returning persons. |
| `utils/cmc.py` | Camera motion compensation using sparse optical flow or ECC. |
| `utils/gmc.py` | Global motion compensation for BoTSORT. |
| `motion/kalman_filter.py` | Kalman filter with 8-dimensional state `[x, y, scale, aspect, vx, vy, vs, va]`. Used by all trackers for motion prediction. |

### Examples (`examples/`)

| File | What it does |
|------|-------------|
| `track.py` | Official BoxMOT CLI entry point. Accepts `--tracking-method`, `--source`, `--reid-model`, `--conf`, `--save`, etc. Integrates with YOLOv8 predictor callbacks. |
| `val.py` | Evaluates tracker performance on MOT16/17/20 benchmarks. Downloads datasets automatically. |
| `evolve.py` | Runs hyperparameter search using Optuna to find optimal tracker config for a given benchmark. |
| `utils.py` | Writes tracking results to MOT-format `.txt` files for benchmark evaluation. |

### Config Files (`boxmot/configs/`)

Each YAML file controls the hyperparameters for its tracker:

| File | Key parameters |
|------|---------------|
| `deepocsort.yaml` | `max_age`, `min_hits`, `iou_thresh`, `w_association_emb`, `delta_t`, `asso_func` |
| `strongsort.yaml` | `max_dist`, `max_iou_dist`, `max_age`, `n_init`, `nn_budget`, `ema_alpha` |
| `botsort.yaml` | `track_high_thresh`, `new_track_thresh`, `track_buffer`, `match_thresh`, `cmc_method`, `lambda_` |
| `bytetrack.yaml` | `track_thresh`, `match_thresh`, `track_buffer`, `frame_rate` |
| `ocsort.yaml` | `det_thresh`, `max_age`, `min_hits`, `iou_thresh`, `delta_t`, `asso_func`, `inertia` |

---

## How Data Flows

```
Video Frame
    │
    ▼
YOLOv8 Detection
    → [x1, y1, x2, y2, conf, cls=0 (person)]
    │
    ▼
Tracker.update(detections, frame)
    ├─ Kalman filter predicts next position
    ├─ IoU matching (spatial overlap)
    ├─ ReID embedding extraction from crops
    ├─ Embedding distance matching (cosine)
    └─ Hungarian algorithm assignment
    → [x1, y1, x2, y2, track_id, conf, cls]
    │
    ▼
IdentityManager (optional, in ultimate/robust scripts)
    ├─ Extracts OSNet embedding for each track
    ├─ Searches persistent gallery for match
    ├─ If match found → reuse existing global_id
    └─ If no match → mint new global_id
    → stable global_id per person
    │
    ▼
Visualization + Output Video
```

---

## Trackers at a Glance

| Tracker | Motion Model | Uses ReID | Camera Compensation | Best For |
|---------|-------------|-----------|--------------------|---------| 
| DeepOCSORT | Kalman | Yes | No | General purpose |
| StrongSORT | Kalman | Yes | No | Appearance-heavy scenes |
| BoTSORT | Kalman | Yes | Yes (optical flow) | Moving camera |
| ByteTrack | Kalman | No | No | Speed, crowded scenes |
| OCSORT | Kalman | No | No | Lightweight, fast |

---

## ReID Models Supported

| Model | Size | Accuracy | Use When |
|-------|------|----------|---------|
| `osnet_x1_0_msmt17.pt` | Large | Best | Accuracy matters most |
| `osnet_x0_75_msmt17.pt` | Medium | Good | Balanced |
| `osnet_x0_5_msmt17.pt` | Small | OK | Speed matters |
| `osnet_x0_25_msmt17.pt` | Tiny | Baseline | Very fast / low memory |
| `lmbn_n_cuhk03_d.pt` | Small | Good | Efficient alternative |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Basic tracking (DeepOCSORT)
python examples/track.py --source test_3.mp4 --tracking-method deepocsort --save

# Best single-script solution
python ultimate_person_tracker.py --source test_3.mp4 --similarity 0.65

# Persistent ReID (IDs survive re-entry)
python track_persistent_reid.py --source test_3.mp4 --reid-threshold 0.4 --show

# Embedding-only (no tracker dependency)
python track_embedding_persistent.py --source test_3.mp4 --similarity-threshold 0.7

# If you get torch.load errors (PyTorch 2.6+)
python run_tracking.py --source test_3.mp4 --tracking-method deepocsort --save
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch >= 2.2` | Deep learning inference |
| `torchvision >= 0.17` | Image transforms |
| `numpy < 2` | Array operations (NumPy 2.x breaks compiled modules) |
| `ultralytics` | YOLOv8 detection |
| `opencv-python` | Video I/O and visualization |
| `scipy` | Cosine distance, sparse matrix ops |
| `lapx` | Linear assignment (Hungarian algorithm) |
| `filterpy` | Kalman filter (OCSORT, DeepOCSORT) |
| `gdown` | Auto-download ReID model weights |
| `loguru` | Logging |
| `PyYAML` | Read tracker config files |
| `pandas` | Export results |

---

## Known Limitations

The core problem this project investigates is an **open research challenge**:

- When a person is fully occluded, YOLO stops detecting them. No tracker can track what the detector doesn't see.
- ReID features change significantly with pose (front vs back view), lighting, and partial visibility.
- There is no threshold that perfectly separates "same person different pose" from "different person similar clothing."

**What works well:** 2–5 people, brief occlusions under 1 second, distinct appearances, good lighting.

**What struggles:** Dense crowds, occlusions over 2 seconds, similar clothing, poor camera angle.

For production attendance systems, consider pairing this with face recognition or a commercial solution (NVIDIA DeepStream, AWS Rekognition) for more reliable long-term re-identification.
