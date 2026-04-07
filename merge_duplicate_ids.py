"""
Post-processing script to merge duplicate person IDs based on embedding analysis
"""
import numpy as np
import json
import cv2
from pathlib import Path
from collections import defaultdict
import sys

def cosine_distance(feat1, feat2):
    """Calculate cosine distance between two feature vectors"""
    return 1.0 - np.dot(feat1, feat2)

def check_temporal_overlap(frames1, frames2, max_gap=5):
    """Check if two persons appear at the same time (with small gap tolerance)"""
    for f1 in frames1:
        for f2 in frames2:
            if abs(f1 - f2) <= max_gap:
                return True
    return False

def find_merge_candidates(embeddings_file, metadata_file, 
                          distance_threshold=0.30, 
                          temporal_gap=5):
    """
    Find pairs of person IDs that should be merged based on:
    1. Low inter-person distance
    2. No temporal overlap (never appear together)
    """
    print("="*70)
    print("FINDING MERGE CANDIDATES")
    print("="*70)
    
    # Load data
    data = np.load(embeddings_file)
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Extract embeddings and frames by PID
    embeddings_by_pid = {}
    frames_by_pid = defaultdict(list)
    
    for key in data.keys():
        if key.endswith('_features'):
            pid = int(key.split('_')[1])
            embeddings_by_pid[pid] = data[key]
    
    for entry in metadata['frame_info']:
        frames_by_pid[entry['pid']].append(entry['frame'])
    
    pids = sorted(embeddings_by_pid.keys())
    
    # Calculate inter-person distances
    merge_candidates = []
    
    for i, pid1 in enumerate(pids):
        for pid2 in pids[i+1:]:
            features1 = embeddings_by_pid[pid1]
            features2 = embeddings_by_pid[pid2]
            
            # Calculate mean distance
            distances = []
            for f1 in features1:
                for f2 in features2:
                    distances.append(cosine_distance(f1, f2))
            
            mean_dist = np.mean(distances)
            min_dist = np.min(distances)
            
            # Check temporal overlap
            frames1 = frames_by_pid[pid1]
            frames2 = frames_by_pid[pid2]
            has_overlap = check_temporal_overlap(frames1, frames2, temporal_gap)
            
            print(f"\nPerson {pid1} vs Person {pid2}:")
            print(f"  Mean distance: {mean_dist:.4f}")
            print(f"  Min distance:  {min_dist:.4f}")
            print(f"  Temporal overlap: {has_overlap}")
            print(f"  Person {pid1} frames: {min(frames1)}-{max(frames1)}")
            print(f"  Person {pid2} frames: {min(frames2)}-{max(frames2)}")
            
            # Merge criteria
            if mean_dist < distance_threshold and not has_overlap:
                merge_candidates.append({
                    'pid1': pid1,
                    'pid2': pid2,
                    'distance': mean_dist,
                    'min_distance': min_dist,
                    'reason': f'Low distance ({mean_dist:.3f}) + No overlap'
                })
                print(f"  ✓ MERGE CANDIDATE!")
            elif mean_dist < distance_threshold and has_overlap:
                print(f"  ⚠ Low distance but temporal overlap - NOT merging")
            
    return merge_candidates, frames_by_pid

def create_merge_map(merge_candidates):
    """
    Create a mapping from old PIDs to new merged PIDs
    Handles transitive merging (if A→B and B→C, then A→C)
    """
    print("\n" + "="*70)
    print("CREATING MERGE MAP")
    print("="*70)
    
    # Build merge groups using union-find
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            # Always merge to lower ID
            if px < py:
                parent[py] = px
            else:
                parent[px] = py
    
    # Process merge candidates
    for candidate in merge_candidates:
        union(candidate['pid1'], candidate['pid2'])
    
    # Create final mapping
    merge_map = {}
    for pid in parent.keys():
        root = find(pid)
        if root != pid:
            merge_map[pid] = root
    
    # Print merge groups
    groups = defaultdict(list)
    for pid in parent.keys():
        root = find(pid)
        groups[root].append(pid)
    
    print("\nMerge Groups:")
    for root, members in sorted(groups.items()):
        if len(members) > 1:
            print(f"  Person {root} ← {members}")
            print(f"    (Merging {', '.join(map(str, members))} into Person {root})")
    
    return merge_map

def apply_merges_to_video(input_video, output_video, metadata_file, merge_map):
    """
    Re-process video and apply ID merges
    """
    print("\n" + "="*70)
    print("APPLYING MERGES TO VIDEO")
    print("="*70)
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Create frame→pid mapping with merges applied
    frame_pid_map = defaultdict(dict)  # frame → {tid: merged_pid}
    
    for entry in metadata['frame_info']:
        frame = entry['frame']
        tid = entry['tid']
        old_pid = entry['pid']
        
        # Apply merge
        new_pid = merge_map.get(old_pid, old_pid)
        frame_pid_map[frame][tid] = new_pid
    
    # Open video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Cannot open {input_video}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nInput: {input_video}")
    print(f"Output: {output_video}")
    print(f"Resolution: {w}x{h} @ {fps}fps")
    print(f"Frames: {total}")
    
    # Create output video
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    # Process video
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Note: We can't re-draw boxes without detection data
        # This just copies the video
        # To actually update IDs on screen, we'd need to re-run detection+tracking
        
        out.write(frame)
        
        if frame_num % (fps * 2) == 0:
            pct = (frame_num / total) * 100
            print(f"  {pct:.0f}% processed...")
    
    cap.release()
    out.release()
    
    print(f"\n✓ Video saved: {output_video}")
    print("\nNote: This video has the same visual IDs as the original.")
    print("To see merged IDs, you need to re-run tracking with the merge map.")
    
    return True

def save_merge_report(merge_candidates, merge_map, frames_by_pid, output_file):
    """Save detailed merge report"""
    print("\n" + "="*70)
    print("SAVING MERGE REPORT")
    print("="*70)
    
    report = {
        'merge_candidates': [
            {
                'from_pid': c['pid2'],
                'to_pid': c['pid1'],
                'distance': float(c['distance']),
                'min_distance': float(c['min_distance']),
                'reason': c['reason']
            }
            for c in merge_candidates
        ],
        'merge_map': {str(k): int(v) for k, v in merge_map.items()},
        'summary': {
            'total_merges': len(merge_map),
            'original_pids': len(set(list(merge_map.keys()) + list(merge_map.values()))),
            'final_pids': len(set(merge_map.values()))
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Merge report saved: {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("MERGE SUMMARY")
    print("="*70)
    
    original_count = len(frames_by_pid)
    merged_count = original_count - len(merge_map)
    
    print(f"\nOriginal person count: {original_count}")
    print(f"Persons merged: {len(merge_map)}")
    print(f"Final person count: {merged_count}")
    
    print("\nMerge Details:")
    for old_pid, new_pid in sorted(merge_map.items()):
        print(f"  Person {old_pid} → Person {new_pid}")
    
    # Show final person list
    final_pids = set(frames_by_pid.keys()) - set(merge_map.keys()) | set(merge_map.values())
    print(f"\nFinal Person IDs: {sorted(final_pids)}")
    
    return report

def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_duplicate_ids.py <base_name> [distance_threshold]")
        print("\nExample:")
        print("  python merge_duplicate_ids.py output_v3")
        print("  python merge_duplicate_ids.py output_v3 0.30")
        print("\nThis will:")
        print("  1. Load output_v3_embeddings.npz and output_v3_metadata.json")
        print("  2. Find duplicate IDs based on low distance + no temporal overlap")
        print("  3. Create merge map and save report")
        print("  4. Save merged_output_v3.mp4 (optional)")
        sys.exit(1)
    
    base_name = sys.argv[1]
    distance_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
    
    embeddings_file = f"{base_name}_embeddings.npz"
    metadata_file = f"{base_name}_metadata.json"
    input_video = f"{base_name}.mp4"
    output_video = f"merged_{base_name}.mp4"
    report_file = f"{base_name}_merge_report.json"
    
    # Check files exist
    if not Path(embeddings_file).exists():
        print(f"Error: {embeddings_file} not found")
        sys.exit(1)
    
    if not Path(metadata_file).exists():
        print(f"Error: {metadata_file} not found")
        sys.exit(1)
    
    print(f"Distance threshold: {distance_threshold}")
    print(f"Temporal gap tolerance: 5 frames")
    print()
    
    # Find merge candidates
    merge_candidates, frames_by_pid = find_merge_candidates(
        embeddings_file, metadata_file, distance_threshold
    )
    
    if not merge_candidates:
        print("\n" + "="*70)
        print("NO MERGE CANDIDATES FOUND")
        print("="*70)
        print("\nAll persons are either:")
        print("  - Too different (distance >= threshold)")
        print("  - Appear at the same time (temporal overlap)")
        print("\nTry:")
        print(f"  - Increase threshold: python merge_duplicate_ids.py {base_name} 0.35")
        print(f"  - Check analysis: python analyze_embeddings.py {base_name}")
        return
    
    # Create merge map
    merge_map = create_merge_map(merge_candidates)
    
    # Save report
    save_merge_report(merge_candidates, merge_map, frames_by_pid, report_file)
    
    # Optionally process video
    if Path(input_video).exists():
        print("\n" + "="*70)
        print("VIDEO PROCESSING")
        print("="*70)
        response = input(f"\nProcess video {input_video}? (y/n): ")
        if response.lower() == 'y':
            apply_merges_to_video(input_video, output_video, metadata_file, merge_map)
        else:
            print("Skipping video processing")
    else:
        print(f"\nNote: {input_video} not found, skipping video processing")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print(f"\nMerge report saved: {report_file}")
    print("\nTo apply merges in future tracking, you can:")
    print("1. Use the merge_map to post-process results")
    print("2. Manually adjust the reid_threshold")
    print("3. Re-run tracking with adjusted parameters")

if __name__ == '__main__':
    main()
