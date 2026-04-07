"""
STEP 2: Cluster tracks to merge same person
Analyzes embeddings and merges tracks that belong to the same person
"""
import pickle
import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

def cluster_tracks(embeddings_file='embeddings.pkl', output_file='gallery.pkl', distance_threshold=0.4):
    """
    Cluster tracks based on embedding similarity
    
    Parameters:
        embeddings_file: Input pickle from step1
        output_file: Output gallery file
        distance_threshold: Maximum distance to merge tracks (lower = stricter)
    
    Returns:
        gallery: {person_id: [embeddings]}
    """
    print("=" * 80)
    print("STEP 2: CLUSTERING TRACKS")
    print("=" * 80)
    print(f"Input: {embeddings_file}")
    print(f"Output: {output_file}")
    print(f"Distance Threshold: {distance_threshold}")
    print("=" * 80)
    
    # Load embeddings
    print("\n📂 Loading embeddings...")
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
    
    embeddings_dict = data['embeddings']
    track_info = data['info']
    
    print(f"   Loaded {len(embeddings_dict)} tracks")
    print(f"   Total embeddings: {sum(len(embs) for embs in embeddings_dict.values())}")
    
    # Show frame ranges for each track
    print("\n📅 Track Timeline (Frame Ranges):")
    print(f"   {'Track ID':<10} {'First Frame':<12} {'Last Frame':<12} {'Duration':<10} {'# Embeddings':<15}")
    print("   " + "-" * 70)
    
    for track_id in sorted(embeddings_dict.keys()):
        info = track_info[track_id]
        num_embs = len(embeddings_dict[track_id])
        first_frame = info['first_frame']
        last_frame = info['last_frame']
        duration = last_frame - first_frame + 1
        
        print(f"   Track {track_id:<5} {first_frame:<12} {last_frame:<12} {duration:<10} {num_embs:<15}")
    
    # Show timeline visualization
    print("\n📊 Visual Timeline:")
    total_frames = data['total_frames']
    chunk_size = 50  # Each character represents 50 frames
    num_chunks = (total_frames + chunk_size - 1) // chunk_size
    
    print(f"   Frame range: 1 to {total_frames} (each '█' = {chunk_size} frames)")
    print("   " + "-" * 80)
    
    for track_id in sorted(embeddings_dict.keys()):
        info = track_info[track_id]
        first_frame = info['first_frame']
        last_frame = info['last_frame']
        
        timeline = ""
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size + 1
            chunk_end = min((chunk_idx + 1) * chunk_size, total_frames)
            
            # Check if track is active in this chunk
            if first_frame <= chunk_end and last_frame >= chunk_start:
                timeline += "█"
            else:
                timeline += "·"
        
        print(f"   Track {track_id:<3}: {timeline} (frames {first_frame}-{last_frame})")
    
    # Compute average embedding per track
    print("\n🧮 Computing average embeddings per track...")
    print("   ⚠️  Using MEDIAN instead of MEAN to reduce outlier impact")
    track_ids = []
    track_avg_embeddings = []
    
    for track_id in sorted(embeddings_dict.keys()):
        embs = np.array(embeddings_dict[track_id])
        
        # Use MEDIAN instead of MEAN - more robust to outliers/contamination
        median_emb = np.median(embs, axis=0)
        median_emb = median_emb / (np.linalg.norm(median_emb) + 1e-8)  # Normalize
        
        track_ids.append(track_id)
        track_avg_embeddings.append(median_emb)
        
        # Show embedding diversity (std dev)
        emb_std = np.std(embs, axis=0).mean()
        print(f"   Track {track_id}: {len(embs)} embeddings, diversity (std): {emb_std:.3f}")
    
    track_avg_embeddings = np.array(track_avg_embeddings)
    print(f"   Computed {len(track_avg_embeddings)} median embeddings")
    
    # Compute pairwise distances
    print("\n📏 Computing pairwise distances...")
    from scipy.spatial.distance import cdist
    distances = cdist(track_avg_embeddings, track_avg_embeddings, metric='cosine')
    
    print(f"   Distance matrix shape: {distances.shape}")
    print(f"   Min distance: {distances[distances > 0].min():.3f}")
    print(f"   Max distance: {distances.max():.3f}")
    print(f"   Mean distance: {distances[distances > 0].mean():.3f}")
    
    # Show pairwise distances
    print("\n🔍 Pairwise Track Distances:")
    print("   " + "-" * 80)
    for i, track_i in enumerate(track_ids):
        for j, track_j in enumerate(track_ids):
            if i < j:  # Only show upper triangle
                dist = distances[i, j]
                if dist < distance_threshold:
                    status = "✅ SIMILAR (will merge)"
                elif dist < distance_threshold + 0.1:
                    status = "⚠️  BORDERLINE"
                else:
                    status = "❌ DIFFERENT"
                print(f"   Track {track_i:2d} ↔ Track {track_j:2d}: {dist:.3f} {status}")
    
    # Visual distance matrix (for small number of tracks)
    if len(track_ids) <= 10:
        print("\n📊 Distance Matrix Heatmap:")
        print("   " + " " * 10 + "".join([f"Trk{tid:2d} " for tid in track_ids]))
        for i, track_i in enumerate(track_ids):
            row_str = f"   Track {track_i:2d}  "
            for j, track_j in enumerate(track_ids):
                dist = distances[i, j]
                if i == j:
                    row_str += " ---- "
                elif dist < 0.3:
                    row_str += f" {dist:.2f}✓"
                elif dist < distance_threshold:
                    row_str += f" {dist:.2f}~"
                else:
                    row_str += f" {dist:.2f} "
            print(row_str)
        print("   Legend: ✓=very similar, ~=similar, (no mark)=different")
    
    # Cluster using hierarchical clustering
    print(f"\n🔗 Clustering with threshold {distance_threshold}...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distances)
    
    n_clusters = len(set(labels))
    print(f"   Found {n_clusters} unique persons (clusters)")
    
    # Show clustering results
    print("\n📊 Clustering Results:")
    cluster_to_tracks = defaultdict(list)
    for track_id, label in zip(track_ids, labels):
        cluster_to_tracks[label].append(track_id)
    
    for cluster_id in sorted(cluster_to_tracks.keys()):
        tracks = cluster_to_tracks[cluster_id]
        print(f"\n   🧑 Person {cluster_id + 1}: Tracks {tracks} ({len(tracks)} tracks)")
        
        # Show frame ranges for this person
        all_first_frames = [track_info[tid]['first_frame'] for tid in tracks]
        all_last_frames = [track_info[tid]['last_frame'] for tid in tracks]
        person_first_frame = min(all_first_frames)
        person_last_frame = max(all_last_frames)
        
        print(f"      Frame range: {person_first_frame} to {person_last_frame}")
        print(f"      Track details:")
        for tid in tracks:
            info = track_info[tid]
            print(f"        - Track {tid}: frames {info['first_frame']}-{info['last_frame']} ({len(embeddings_dict[tid])} embeddings)")
        
        # Show why these tracks were merged
        if len(tracks) > 1:
            print(f"      Merge reasoning:")
            for i, track_i in enumerate(tracks):
                for j, track_j in enumerate(tracks):
                    if i < j:
                        idx_i = track_ids.index(track_i)
                        idx_j = track_ids.index(track_j)
                        dist = distances[idx_i, idx_j]
                        if dist < distance_threshold - 0.1:
                            confidence = "✅ confident"
                        elif dist < distance_threshold:
                            confidence = "⚠️  borderline"
                        else:
                            confidence = "❓ questionable"
                        print(f"      - Track {track_i} ↔ Track {track_j}: {dist:.3f} {confidence}")
            
            # Calculate average intra-cluster distance
            intra_distances = []
            for i, track_i in enumerate(tracks):
                for j, track_j in enumerate(tracks):
                    if i < j:
                        idx_i = track_ids.index(track_i)
                        idx_j = track_ids.index(track_j)
                        intra_distances.append(distances[idx_i, idx_j])
            avg_intra = np.mean(intra_distances)
            print(f"      Average intra-cluster distance: {avg_intra:.3f}")
        else:
            print(f"      Single track - no merging needed")
    
    # Build gallery: merge embeddings from same person
    print("\n🏛️ Building gallery...")
    gallery = {}  # {person_id: [all_embeddings]}
    person_info = {}  # {person_id: {'tracks': [], 'total_embeddings': X}}
    
    for cluster_id in sorted(cluster_to_tracks.keys()):
        person_id = cluster_id + 1
        tracks = cluster_to_tracks[cluster_id]
        
        # Collect ALL embeddings from all tracks in this cluster
        all_embeddings = []
        for track_id in tracks:
            all_embeddings.extend(embeddings_dict[track_id])
        
        gallery[person_id] = all_embeddings
        person_info[person_id] = {
            'tracks': tracks,
            'total_embeddings': len(all_embeddings)
        }
        
        print(f"   Person {person_id}: {len(all_embeddings)} embeddings from {len(tracks)} tracks")
        if len(tracks) > 1:
            print(f"      ✅ Merged tracks: {tracks}")
    
    # Save gallery
    print(f"\n💾 Saving gallery to {output_file}...")
    gallery_data = {
        'gallery': gallery,
        'person_info': person_info,
        'cluster_labels': labels.tolist(),
        'track_ids': track_ids,
        'distance_threshold': distance_threshold,
        'video': data['video'],
        'reid_model': data['reid_model']
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(gallery_data, f)
    
    print(f"✅ Gallery saved!")
    print(f"\n📈 Summary:")
    print(f"   Original tracks: {len(track_ids)}")
    print(f"   Unique persons: {n_clusters}")
    print(f"   Tracks merged: {len(track_ids) - n_clusters}")
    
    # Show merge summary
    if len(track_ids) - n_clusters > 0:
        print(f"\n🔗 Merge Summary:")
        for cluster_id in sorted(cluster_to_tracks.keys()):
            tracks = cluster_to_tracks[cluster_id]
            if len(tracks) > 1:
                person_id = cluster_id + 1
                print(f"   Person {person_id}: {tracks[0]} + {tracks[1:]} → merged into single identity")
    
    print("=" * 80)
    
    return gallery_data

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Cluster tracks to identify unique persons')
    parser.add_argument('--input', type=str, default='embeddings.pkl', help='Input embeddings file from step1')
    parser.add_argument('--output', type=str, default='gallery.pkl', help='Output gallery file')
    parser.add_argument('--threshold', type=float, default=0.4, help='Distance threshold for clustering (lower = stricter)')
    
    args = parser.parse_args()
    
    cluster_tracks(args.input, args.output, args.threshold)
