"""
Analyze saved embeddings and calculate distances between persons
"""
import numpy as np
import json
from pathlib import Path
import sys

def cosine_distance(feat1, feat2):
    """Calculate cosine distance between two feature vectors"""
    return 1.0 - np.dot(feat1, feat2)

def euclidean_distance(feat1, feat2):
    """Calculate Euclidean distance between two feature vectors"""
    return np.linalg.norm(feat1 - feat2)

def analyze_embeddings(embeddings_file, metadata_file):
    """Analyze embeddings and calculate intra/inter person distances"""
    
    print("="*70)
    print("EMBEDDING DISTANCE ANALYSIS")
    print("="*70)
    
    # Load data
    print(f"\nLoading: {embeddings_file}")
    data = np.load(embeddings_file)
    
    print(f"Loading: {metadata_file}")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nTotal embeddings: {metadata['total_embeddings']}")
    print(f"Unique persons: {metadata['unique_pids']}")
    print()
    
    # Extract embeddings by PID
    embeddings_by_pid = {}
    for key in data.keys():
        if key.endswith('_features'):
            pid = int(key.split('_')[1])
            embeddings_by_pid[pid] = data[key]
    
    pids = sorted(embeddings_by_pid.keys())
    
    # Calculate intra-person distances (same person)
    print("="*70)
    print("INTRA-PERSON DISTANCES (Same Person)")
    print("="*70)
    
    intra_distances = {}
    for pid in pids:
        features = embeddings_by_pid[pid]
        if len(features) < 2:
            print(f"\nPerson {pid}: Only {len(features)} embedding(s), skipping")
            continue
        
        # Calculate all pairwise distances within this person
        distances = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                dist = cosine_distance(features[i], features[j])
                distances.append(dist)
        
        intra_distances[pid] = distances
        
        print(f"\nPerson {pid}: {len(features)} embeddings, {len(distances)} pairs")
        print(f"  Min distance:  {np.min(distances):.4f}")
        print(f"  Max distance:  {np.max(distances):.4f}")
        print(f"  Mean distance: {np.mean(distances):.4f}")
        print(f"  Std distance:  {np.std(distances):.4f}")
        print(f"  Median:        {np.median(distances):.4f}")
        print(f"  25th percentile: {np.percentile(distances, 25):.4f}")
        print(f"  75th percentile: {np.percentile(distances, 75):.4f}")
    
    # Calculate inter-person distances (different persons)
    print("\n" + "="*70)
    print("INTER-PERSON DISTANCES (Different Persons)")
    print("="*70)
    
    inter_distances = {}
    for i, pid1 in enumerate(pids):
        for pid2 in pids[i+1:]:
            features1 = embeddings_by_pid[pid1]
            features2 = embeddings_by_pid[pid2]
            
            # Calculate distances between all pairs
            distances = []
            for f1 in features1:
                for f2 in features2:
                    dist = cosine_distance(f1, f2)
                    distances.append(dist)
            
            inter_distances[(pid1, pid2)] = distances
            
            print(f"\nPerson {pid1} vs Person {pid2}: {len(distances)} pairs")
            print(f"  Min distance:  {np.min(distances):.4f}")
            print(f"  Max distance:  {np.max(distances):.4f}")
            print(f"  Mean distance: {np.mean(distances):.4f}")
            print(f"  Std distance:  {np.std(distances):.4f}")
            print(f"  Median:        {np.median(distances):.4f}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    all_intra = []
    for distances in intra_distances.values():
        all_intra.extend(distances)
    
    all_inter = []
    for distances in inter_distances.values():
        all_inter.extend(distances)
    
    if all_intra:
        print(f"\nAll INTRA-person distances (same person):")
        print(f"  Count: {len(all_intra)}")
        print(f"  Min:    {np.min(all_intra):.4f}")
        print(f"  Max:    {np.max(all_intra):.4f}")
        print(f"  Mean:   {np.mean(all_intra):.4f}")
        print(f"  Median: {np.median(all_intra):.4f}")
        print(f"  Std:    {np.std(all_intra):.4f}")
    
    if all_inter:
        print(f"\nAll INTER-person distances (different persons):")
        print(f"  Count: {len(all_inter)}")
        print(f"  Min:    {np.min(all_inter):.4f}")
        print(f"  Max:    {np.max(all_inter):.4f}")
        print(f"  Mean:   {np.mean(all_inter):.4f}")
        print(f"  Median: {np.median(all_inter):.4f}")
        print(f"  Std:    {np.std(all_inter):.4f}")
    
    # Threshold recommendations
    print("\n" + "="*70)
    print("THRESHOLD RECOMMENDATIONS")
    print("="*70)
    
    if all_intra and all_inter:
        # Find optimal threshold
        intra_95 = np.percentile(all_intra, 95)  # 95% of same-person pairs below this
        inter_5 = np.percentile(all_inter, 5)    # 5% of different-person pairs below this
        
        print(f"\nIntra-person 95th percentile: {intra_95:.4f}")
        print(f"  (95% of same-person pairs have distance < {intra_95:.4f})")
        
        print(f"\nInter-person 5th percentile: {inter_5:.4f}")
        print(f"  (5% of different-person pairs have distance < {inter_5:.4f})")
        
        if intra_95 < inter_5:
            optimal = (intra_95 + inter_5) / 2
            print(f"\n✓ Good separation! Recommended threshold: {optimal:.4f}")
            print(f"  (Midpoint between intra-95% and inter-5%)")
        else:
            print(f"\n⚠ Poor separation! Intra-95% ({intra_95:.4f}) >= Inter-5% ({inter_5:.4f})")
            print(f"  Same-person and different-person distances overlap significantly")
            print(f"  This explains why the system creates multiple IDs")
            
            # Find best compromise
            overlap_start = inter_5
            overlap_end = intra_95
            compromise = (overlap_start + overlap_end) / 2
            print(f"\n  Compromise threshold: {compromise:.4f}")
            print(f"  Warning: This will have false positives and/or false negatives")
        
        # Current threshold
        current_threshold = 0.35  # From your code
        print(f"\nCurrent reid_threshold: {current_threshold:.4f}")
        
        intra_above = np.sum(np.array(all_intra) >= current_threshold)
        inter_below = np.sum(np.array(all_inter) < current_threshold)
        
        print(f"  Same-person pairs rejected: {intra_above}/{len(all_intra)} ({100*intra_above/len(all_intra):.1f}%)")
        print(f"  Different-person pairs accepted: {inter_below}/{len(all_inter)} ({100*inter_below/len(all_inter):.1f}%)")
        
        if intra_above > 0:
            print(f"\n  ⚠ {intra_above} same-person pairs are being rejected!")
            print(f"     This causes the system to create new IDs for the same person")
        
        if inter_below > 0:
            print(f"\n  ⚠ {inter_below} different-person pairs would be accepted!")
            print(f"     This could cause wrong ID assignments")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return {
        'intra_distances': intra_distances,
        'inter_distances': inter_distances,
        'all_intra': all_intra,
        'all_inter': all_inter
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_embeddings.py <base_name>")
        print("Example: python analyze_embeddings.py output_v3")
        print("\nThis will load:")
        print("  - output_v3_embeddings.npz")
        print("  - output_v3_metadata.json")
        sys.exit(1)
    
    base_name = sys.argv[1]
    embeddings_file = f"{base_name}_embeddings.npz"
    metadata_file = f"{base_name}_metadata.json"
    
    if not Path(embeddings_file).exists():
        print(f"Error: {embeddings_file} not found")
        sys.exit(1)
    
    if not Path(metadata_file).exists():
        print(f"Error: {metadata_file} not found")
        sys.exit(1)
    
    analyze_embeddings(embeddings_file, metadata_file)

if __name__ == '__main__':
    main()
