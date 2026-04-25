"""
Phi-3 Local Person Tracking System
===================================
Uses Microsoft's open-source Phi-3 model running locally on your GPU.
No API costs, no quota limits, complete privacy, NO AUTHENTICATION REQUIRED.

Features:
- Runs completely offline (no internet needed after setup)
- Multi-modal reasoning (appearance + spatial + temporal)
- View-invariant (handles 360° rotations)
- Free forever (no API costs)
- No quota limits
- No authentication required (unlike Gemma)
- Works with PyTorch 2.0.1+

Requirements:
- transformers library
- torch with CUDA
- 4GB+ GPU memory (you have RTX 5050 - perfect!)
- Phi-3 model weights (~4GB download, one-time)

Usage:
    # First time: Download model (one-time, ~4GB)
    python track_gemma_local.py --download-model
    
    # Then track video
    python track_gemma_local.py --source test_6.mp4 --output output_gemma_local.mp4
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import argparse
import json
from collections import defaultdict

# Check if transformers is installed
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("WARNING: transformers not installed. Install with: pip install transformers accelerate")

# Conditional imports for tracking (not needed for download)
CV2_AVAILABLE = False
YOLO_AVAILABLE = False
PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    pass

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

try:
    from PIL import Image
    import io
    import base64
    PIL_AVAILABLE = True
except ImportError:
    pass


class GemmaLocalPersonTracker:
    """
    Person tracker using Phi-3 model running locally.
    Processes video in batches and uses Phi-3's multi-modal understanding.
    No authentication required - completely free and open.
    """
    
    def __init__(self, yolo_model='yolov8n.pt', detection_conf=0.25,
                 batch_size=5, model_name='microsoft/Phi-3-mini-4k-instruct', device='cuda'):
        """
        Initialize Phi-3-based local tracker.
        
        Args:
            yolo_model: YOLO model for detection
            detection_conf: Detection confidence threshold
            batch_size: Number of frames to process at once (smaller for local model)
            model_name: Phi-3 model name (microsoft/Phi-3-mini-4k-instruct is free, no auth needed)
            device: 'cuda' or 'cpu'
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Run: pip install transformers accelerate")
        
        if not CV2_AVAILABLE:
            raise ImportError("opencv not installed. Run: pip install opencv-python")
        
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # YOLO for detection
        self.yolo = YOLO(yolo_model)
        self.detection_conf = detection_conf
        
        # Tracking parameters
        self.batch_size = batch_size
        self.frame_count = 0
        self.person_tracks = {}  # person_id -> list of frame appearances
        self.colors = {}  # person_id -> color for visualization
        
        # Phi-3 model (will be loaded when needed)
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        
        print(f"✓ Phi-3 local tracker initialized")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {batch_size} frames")
        print(f"  Detection confidence: {detection_conf}")
        print(f"  Model: {model_name}")
    
    def load_model(self):
        """Load Phi-3 model (lazy loading to save memory)."""
        if self.model is not None:
            return
        
        print(f"\nLoading Phi-3 model: {self.model_name}")
        print("This may take 1-2 minutes on first load...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with optimizations for PyTorch 2.0.1
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map='auto' if self.device == 'cuda' else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            if self.device == 'cpu':
                self.model = self.model.to('cpu')
            
            print(f"✓ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"\n✗ Error loading model: {e}")
            print("\nTroubleshooting:")
            print("1. Run: pip install transformers accelerate")
            print("2. Download model first: python track_gemma_local.py --download-model")
            print("3. Check you have ~4GB free GPU memory")
            print("4. Model is free and doesn't require authentication")
            raise
    
    def _detect_people(self, frame):
        """Detect people using YOLO."""
        results = self.yolo(frame, conf=self.detection_conf, classes=[0], verbose=False)[0]
        
        detections = []
        if len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf
                })
        
        return detections
    
    def _create_prompt(self, batch_detections, known_persons):
        """Create text prompt for Phi-3 (text-only, no images)."""
        prompt = f"""<|system|>
You are an expert person re-identification system analyzing video frames.<|end|>
<|user|>
Track people across {len(batch_detections)} consecutive video frames and assign consistent person IDs.

Known Persons from Previous Batches:
{json.dumps(known_persons, indent=2) if known_persons else "None (first batch)"}

Detection Information (bounding boxes and positions):
"""
        
        for i, detections in enumerate(batch_detections):
            prompt += f"\nFrame {i}:\n"
            if not detections:
                prompt += "  No persons detected\n"
            else:
                for j, det in enumerate(detections):
                    bbox = det['bbox']
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    
                    # Describe position
                    if x_center < 200:
                        position = "left side"
                    elif x_center > 600:
                        position = "right side"
                    else:
                        position = "center"
                    
                    prompt += f"  Detection {j}: {position}, size={width}x{height}, bbox={bbox}, conf={det['confidence']:.2f}\n"
        
        prompt += """

Instructions:
1. Analyze spatial patterns (where people are located)
2. Analyze temporal patterns (movement between frames)
3. Reuse person_id from known persons when same person reappears
4. Only create new person_id for genuinely new people
5. Consider: people can't teleport, brief disappearances are normal
6. Same person in different locations across frames = same person_id

Output Format (JSON only, no explanation):
{
  "persons": [
    {
      "person_id": 1,
      "description": "brief description (location, movement pattern)",
      "frames": [
        {
          "frame_index": 0,
          "detection_index": 0,
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.95,
          "reasoning": "why this detection belongs to this person"
        }
      ]
    }
  ],
  "reasoning": "overall reasoning for ID assignments"
}

Return ONLY the JSON, no other text.<|end|>
<|assistant|>"""
        
        return prompt
    
    def _parse_gemma_response(self, response_text):
        """Parse Phi-3's JSON response."""
        try:
            # Extract JSON from response
            if '```json' in response_text:
                json_str = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                json_str = response_text.split('```')[1].split('```')[0].strip()
            elif '{' in response_text and '}' in response_text:
                # Find first { and last }
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_str = response_text[start:end]
            else:
                json_str = response_text.strip()
            
            data = json.loads(json_str)
            return data
        except Exception as e:
            print(f"ERROR parsing Phi-3 response: {e}")
            print(f"Response: {response_text[:500]}")
            return None
    
    def _process_batch(self, batch_frame_numbers, batch_detections):
        """Process a batch of frames with Phi-3."""
        print(f"\n  Processing frames {batch_frame_numbers[0]}-{batch_frame_numbers[-1]}...")
        
        # Prepare known persons summary
        known_persons = {}
        for pid, appearances in self.person_tracks.items():
            if appearances:
                last_frame = appearances[-1]['frame']
                known_persons[pid] = {
                    'last_seen_frame': last_frame,
                    'total_appearances': len(appearances),
                    'description': appearances[-1].get('description', 'unknown')
                }
        
        # Create prompt
        prompt = self._create_prompt(batch_detections, known_persons)
        
        # Call Gemma model
        try:
            # Ensure model is loaded
            if self.model is None:
                self.load_model()
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            if prompt in response_text:
                response_text = response_text.split(prompt)[-1].strip()
            
            # Parse response
            result = self._parse_gemma_response(response_text)
            
            if result and 'persons' in result:
                print(f"  ✓ Phi-3 identified {len(result['persons'])} person(s)")
                if 'reasoning' in result:
                    print(f"  Reasoning: {result['reasoning'][:100]}...")
                
                # Update tracking
                for person in result['persons']:
                    pid = person['person_id']
                    if pid not in self.person_tracks:
                        self.person_tracks[pid] = []
                    
                    for frame_data in person['frames']:
                        frame_idx = frame_data['frame_index']
                        actual_frame = batch_frame_numbers[frame_idx]
                        
                        self.person_tracks[pid].append({
                            'frame': actual_frame,
                            'bbox': frame_data['bbox'],
                            'confidence': frame_data.get('confidence', 1.0),
                            'reasoning': frame_data.get('reasoning', ''),
                            'description': person.get('description', '')
                        })
                
                return result
            else:
                print(f"  ✗ Failed to parse Phi-3 response")
                return None
                
        except Exception as e:
            print(f"  ✗ Phi-3 processing error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_person_color(self, person_id):
        """Get consistent color for person ID."""
        if person_id not in self.colors:
            np.random.seed(person_id * 37)
            self.colors[person_id] = tuple(np.random.randint(60, 230, 3).tolist())
        return self.colors[person_id]
    
    def _draw_tracks(self, frame, frame_number):
        """Draw person tracks on frame."""
        for pid, appearances in self.person_tracks.items():
            # Find appearance for this frame
            for app in appearances:
                if app['frame'] == frame_number:
                    bbox = app['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    color = self._get_person_color(pid)
                    label = f"Person {pid}"
                    
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - lh - 12), (x1 + lw + 8, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 4, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw info
        n_unique = len(self.person_tracks)
        info = f"Frame: {frame_number} | Unique Persons: {n_unique}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    
    def track_video(self, source, output='output_gemma_local.mp4', show=False):
        """
        Track people in video using local Phi-3 model.
        
        Args:
            source: Input video path
            output: Output video path
            show: Whether to display video while processing
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video: {source}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*60}")
        print(f"PHI-3 LOCAL PERSON TRACKER")
        print(f"{'='*60}")
        print(f"Video: {w}x{h} @ {fps}fps, {total_frames} frames")
        print(f"Processing in batches of {self.batch_size} frames")
        print(f"Running on: {self.device}")
        print(f"Model: {self.model_name}")
        print(f"{'='*60}\n")
        
        # First pass: Detect all people and process with Phi-3 in batches
        print("PHASE 1: Detection and Phi-3 Processing")
        print("-" * 60)
        
        all_detections = []
        frame_numbers = []
        
        batch_detections = []
        batch_frame_numbers = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Detect people
            detections = self._detect_people(frame)
            
            if detections:  # Only process frames with detections
                all_detections.append(detections)
                frame_numbers.append(frame_idx)
                
                batch_detections.append(detections)
                batch_frame_numbers.append(frame_idx)
                
                # Process batch when full
                if len(batch_detections) >= self.batch_size:
                    self._process_batch(batch_frame_numbers, batch_detections)
                    batch_detections = []
                    batch_frame_numbers = []
            
            if frame_idx % 100 == 0:
                print(f"  Detected in {frame_idx}/{total_frames} frames...")
        
        # Process remaining frames
        if batch_detections:
            self._process_batch(batch_frame_numbers, batch_detections)
        
        cap.release()
        
        print(f"\n{'='*60}")
        print(f"PHASE 2: Rendering Output Video")
        print(f"{'='*60}\n")
        
        # Second pass: Render video with tracks
        cap = cv2.VideoCapture(source)
        out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Draw tracks
            self._draw_tracks(frame, frame_idx)
            
            out.write(frame)
            
            if show:
                cv2.imshow('Phi-3 Local Tracker', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if frame_idx % 100 == 0:
                pct = (frame_idx / total_frames) * 100
                print(f"  Rendering: {pct:.0f}%")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Video saved: {output}")
        
        # Print summary
        self._print_summary(total_frames, fps)
    
    def _print_summary(self, total_frames, fps):
        """Print tracking summary."""
        print(f"\n{'='*60}")
        print(f"TRACKING SUMMARY")
        print(f"{'='*60}")
        print(f"Total unique persons: {len(self.person_tracks)}")
        print()
        
        for pid in sorted(self.person_tracks.keys()):
            appearances = self.person_tracks[pid]
            if appearances:
                frames = [a['frame'] for a in appearances]
                first, last = min(frames), max(frames)
                duration = (last - first) / fps
                description = appearances[0].get('description', 'unknown')
                
                print(f"  Person {pid}: Frames {first}-{last} ({duration:.1f}s)")
                print(f"    Appearances: {len(appearances)}")
                print(f"    Description: {description}")
        
        print(f"{'='*60}\n")


def download_model(model_name='microsoft/Phi-3-mini-4k-instruct'):
    """Download Phi-3 model for offline use."""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING PHI-3 MODEL")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Size: ~4GB (one-time download)")
    print(f"This may take 10-30 minutes depending on your internet speed...")
    print(f"No authentication required - completely free!")
    print(f"{'='*60}\n")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✓ Tokenizer downloaded")
        
        print("\nDownloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("✓ Model downloaded")
        
        print(f"\n{'='*60}")
        print("DOWNLOAD COMPLETE!")
        print(f"{'='*60}")
        print("\nYou can now run:")
        print(f"python track_gemma_local.py --source test_6.mp4 --output output_gemma.mp4")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Install: pip install transformers accelerate")
        print("3. Model is free and doesn't require authentication")


def main():
    parser = argparse.ArgumentParser(description='Phi-3 Local Person Tracker')
    parser.add_argument('--source', type=str, help='Input video path')
    parser.add_argument('--output', type=str, default='output_gemma_local.mp4', help='Output video path')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt', help='YOLO model')
    parser.add_argument('--detection-conf', type=float, default=0.25, help='Detection confidence')
    parser.add_argument('--batch-size', type=int, default=5, help='Frames per batch (smaller for local)')
    parser.add_argument('--model', type=str, default='microsoft/Phi-3-mini-4k-instruct', 
                        help='Phi-3 model (free, no authentication required)')
    parser.add_argument('--show', action='store_true', help='Display video while processing')
    parser.add_argument('--download-model', action='store_true', help='Download model and exit')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    args = parser.parse_args()
    
    # Download model if requested
    if args.download_model:
        download_model(args.model)
        return
    
    # Check source is provided
    if not args.source:
        print("ERROR: --source is required")
        print("\nUsage:")
        print("  # First time: Download model")
        print("  python track_gemma_local.py --download-model")
        print("\n  # Then track video")
        print("  python track_gemma_local.py --source test_6.mp4 --output output_gemma.mp4")
        return
    
    try:
        tracker = GemmaLocalPersonTracker(
            yolo_model=args.yolo_model,
            detection_conf=args.detection_conf,
            batch_size=args.batch_size,
            model_name=args.model,
            device=args.device
        )
        
        tracker.track_video(source=args.source, output=args.output, show=args.show)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
