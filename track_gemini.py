"""
Gemini-Based Person Tracking System
====================================
Uses Google Gemini API for intelligent person re-identification.
Handles front/back profiles, similar clothing, and occlusions better than embedding-based approaches.

Features:
- Multi-modal reasoning (appearance + spatial + temporal)
- View-invariant (handles 360° rotations)
- Context-aware person matching
- Structured JSON output for tracking

Usage:
    python track_gemini.py --source test_6.mp4 --api-key YOUR_API_KEY --output output_gemini.mp4
    
    # Or set API key as environment variable:
    export GEMINI_API_KEY=your_api_key_here
    python track_gemini.py --source test_6.mp4 --output output_gemini.mp4
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import argparse
import json
from collections import defaultdict
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image

# Check if google-generativeai is installed
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("WARNING: google-generativeai not installed. Install with: pip install google-generativeai")


class GeminiPersonTracker:
    """
    Person tracker using Gemini API for re-identification.
    Processes video in batches and uses Gemini's multi-modal understanding.
    """
    
    def __init__(self, api_key=None, yolo_model='yolov8n.pt', detection_conf=0.25,
                 batch_size=10, model_name='gemini-2.5-flash'):
        """
        Initialize Gemini-based tracker.
        
        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env variable)
            yolo_model: YOLO model for detection
            detection_conf: Detection confidence threshold
            batch_size: Number of frames to send to Gemini at once
            model_name: Gemini model to use
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        # Get API key
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key required. Set --api-key or GEMINI_API_KEY environment variable")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Try to create model with proper name format
        try:
            self.model = genai.GenerativeModel(model_name)
        except Exception as e:
            print(f"Warning: Could not load {model_name}, trying gemini-pro...")
            try:
                self.model = genai.GenerativeModel('gemini-pro')
                model_name = 'gemini-pro'
            except Exception as e2:
                print(f"Error: Could not load any Gemini model: {e2}")
                raise
        
        # YOLO for detection
        self.yolo = YOLO(yolo_model)
        self.detection_conf = detection_conf
        
        # Tracking parameters
        self.batch_size = batch_size
        self.frame_count = 0
        self.person_tracks = {}  # person_id -> list of frame appearances
        self.colors = {}  # person_id -> color for visualization
        
        print(f"✓ Gemini tracker initialized")
        print(f"  Model: {model_name}")
        print(f"  Batch size: {batch_size} frames")
        print(f"  Detection confidence: {detection_conf}")
    
    def _encode_image(self, image):
        """Encode image to base64 for Gemini API."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Resize if too large (Gemini has size limits)
        max_size = 1024
        if max(pil_image.size) > max_size:
            ratio = max_size / max(pil_image.size)
            new_size = tuple(int(dim * ratio) for dim in pil_image.size)
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        return pil_image
    
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
    
    def _create_prompt(self, batch_frames, batch_detections, known_persons):
        """Create prompt for Gemini API."""
        prompt = f"""You are an expert person re-identification system. Analyze these {len(batch_frames)} video frames and track people across them.

**Your Task:**
1. Identify each unique person across all frames
2. Assign consistent person IDs (use existing IDs when person reappears)
3. Handle front/back views, similar clothing, and occlusions
4. Use spatial and temporal context for matching

**Known Persons from Previous Batches:**
{json.dumps(known_persons, indent=2) if known_persons else "None (first batch)"}

**Detection Information:**
"""
        
        for i, detections in enumerate(batch_detections):
            prompt += f"\nFrame {i}: {len(detections)} person(s) detected\n"
            for j, det in enumerate(detections):
                bbox = det['bbox']
                prompt += f"  Person {j}: bbox=[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}], conf={det['confidence']:.2f}\n"
        
        prompt += """
**Output Format (JSON only, no explanation):**
```json
{
  "persons": [
    {
      "person_id": 1,
      "description": "brief appearance description",
      "frames": [
        {
          "frame_index": 0,
          "detection_index": 0,
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.95,
          "notes": "front view" or "back view" or "occluded" etc
        }
      ]
    }
  ],
  "reasoning": "brief explanation of ID assignments"
}
```

**Important Rules:**
1. Reuse person_id from known persons when same person reappears
2. Only create new person_id for genuinely new people
3. Same person from different angles = same person_id
4. Consider spatial continuity (person can't teleport)
5. Consider temporal continuity (brief disappearances are normal)
6. Similar clothing is common - use body shape, location, movement
7. Front and back views of same person = same person_id

Return ONLY the JSON, no other text."""
        
        return prompt
    
    def _parse_gemini_response(self, response_text):
        """Parse Gemini's JSON response."""
        try:
            # Extract JSON from response (may have markdown code blocks)
            if '```json' in response_text:
                json_str = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                json_str = response_text.split('```')[1].split('```')[0].strip()
            else:
                json_str = response_text.strip()
            
            data = json.loads(json_str)
            return data
        except Exception as e:
            print(f"ERROR parsing Gemini response: {e}")
            print(f"Response: {response_text[:500]}")
            return None
    
    def _process_batch(self, batch_frames, batch_frame_numbers, batch_detections):
        """Process a batch of frames with Gemini."""
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
        prompt = self._create_prompt(batch_frames, batch_detections, known_persons)
        
        # Prepare images for Gemini
        images = [self._encode_image(frame) for frame in batch_frames]
        
        # Call Gemini API
        try:
            content = [prompt] + images
            response = self.model.generate_content(content)
            
            # Parse response
            result = self._parse_gemini_response(response.text)
            
            if result and 'persons' in result:
                print(f"  ✓ Gemini identified {len(result['persons'])} person(s)")
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
                            'notes': frame_data.get('notes', ''),
                            'description': person.get('description', '')
                        })
                
                return result
            else:
                print(f"  ✗ Failed to parse Gemini response")
                return None
                
        except Exception as e:
            print(f"  ✗ Gemini API error: {e}")
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
                    
                    # Draw notes if any
                    if app.get('notes'):
                        cv2.putText(frame, app['notes'], (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw info
        n_unique = len(self.person_tracks)
        info = f"Frame: {frame_number} | Unique Persons: {n_unique}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    
    def track_video(self, source, output='output_gemini.mp4', show=False):
        """
        Track people in video using Gemini API.
        
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
        print(f"GEMINI PERSON TRACKER")
        print(f"{'='*60}")
        print(f"Video: {w}x{h} @ {fps}fps, {total_frames} frames")
        print(f"Processing in batches of {self.batch_size} frames")
        print(f"{'='*60}\n")
        
        # First pass: Detect all people and process with Gemini in batches
        print("PHASE 1: Detection and Gemini Processing")
        print("-" * 60)
        
        all_frames = []
        all_detections = []
        frame_numbers = []
        
        batch_frames = []
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
                all_frames.append(frame.copy())
                all_detections.append(detections)
                frame_numbers.append(frame_idx)
                
                batch_frames.append(frame.copy())
                batch_detections.append(detections)
                batch_frame_numbers.append(frame_idx)
                
                # Process batch when full
                if len(batch_frames) >= self.batch_size:
                    self._process_batch(batch_frames, batch_frame_numbers, batch_detections)
                    batch_frames = []
                    batch_detections = []
                    batch_frame_numbers = []
            
            if frame_idx % 100 == 0:
                print(f"  Detected in {frame_idx}/{total_frames} frames...")
        
        # Process remaining frames
        if batch_frames:
            self._process_batch(batch_frames, batch_frame_numbers, batch_detections)
        
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
                cv2.imshow('Gemini Tracker', frame)
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


def main():
    parser = argparse.ArgumentParser(description='Gemini-Based Person Tracker')
    parser.add_argument('--source', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default='output_gemini.mp4', help='Output video path')
    parser.add_argument('--api-key', type=str, help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt', help='YOLO model')
    parser.add_argument('--detection-conf', type=float, default=0.25, help='Detection confidence')
    parser.add_argument('--batch-size', type=int, default=10, help='Frames per Gemini API call')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash', 
                        help='Gemini model (gemini-2.5-flash, gemini-flash-latest, gemini-pro-latest, etc)')
    parser.add_argument('--show', action='store_true', help='Display video while processing')
    args = parser.parse_args()
    
    try:
        tracker = GeminiPersonTracker(
            api_key=args.api_key,
            yolo_model=args.yolo_model,
            detection_conf=args.detection_conf,
            batch_size=args.batch_size,
            model_name=args.model
        )
        
        tracker.track_video(source=args.source, output=args.output, show=args.show)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
