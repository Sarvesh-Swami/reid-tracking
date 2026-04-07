"""
Inspect the embeddings.pkl file to verify step1 output
"""
import pickle
import numpy as np

print("=" * 80)
print("INSPECTING EMBEDDINGS.PKL")
print("=" * 80)

# Load the file
with open('embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

print("\n📦 File Contents:")
print(f"   Keys: {list(data.keys())}")

print("\n🎬 Video Info:")
print(f"   Video: {data['video']}")
print(f"   ReID Model: {data['reid_model']}")
print(f"   Total Frames: {data['total_frames']}")
print(f"   FPS: {data['fps']:.2f}")

print("\n👥 Embeddings Data:")
embeddings = data['embeddings']
track_info = data['info']

print(f"   Total unique track IDs: {len(embeddings)}")
print(f"   Total embeddings stored: {sum(len(embs) for embs in embeddings.values())}")

print("\n📊 Per-Track Details:")
print(f"   {'Track ID':<10} {'# Embeddings':<15} {'First Frame':<12} {'Last Frame':<12} {'Duration':<10}")
print("   " + "-" * 70)

for track_id in sorted(embeddings.keys()):
    embs = embeddings[track_id]
    info = track_info[track_id]
    
    num_embs = len(embs)
    first_frame = info['first_frame']
    last_frame = info['last_frame']
    duration = last_frame - first_frame
    
    print(f"   {track_id:<10} {num_embs:<15} {first_frame:<12} {last_frame:<12} {duration:<10}")

print("\n🔍 Embedding Quality Check:")
# Check first track's embeddings
first_track_id = sorted(embeddings.keys())[0]
first_track_embs = embeddings[first_track_id]

print(f"   Checking Track {first_track_id}:")
print(f"   - Number of embeddings: {len(first_track_embs)}")
print(f"   - Embedding shape: {first_track_embs[0].shape}")
print(f"   - Embedding dtype: {first_track_embs[0].dtype}")
print(f"   - Embedding norm: {np.linalg.norm(first_track_embs[0]):.3f} (should be ~1.0)")
print(f"   - Min value: {first_track_embs[0].min():.3f}")
print(f"   - Max value: {first_track_embs[0].max():.3f}")

# Check if embeddings are different
if len(first_track_embs) >= 2:
    emb1 = first_track_embs[0]
    emb2 = first_track_embs[1]
    similarity = np.dot(emb1, emb2)
    distance = 1.0 - similarity
    print(f"   - Distance between first 2 embeddings: {distance:.3f}")
    print(f"     (Should be small ~0.0-0.2 for same person)")

# Check distance between different tracks
if len(embeddings) >= 2:
    track_ids = sorted(embeddings.keys())
    track1_id = track_ids[0]
    track2_id = track_ids[1]
    
    emb1 = embeddings[track1_id][0]
    emb2 = embeddings[track2_id][0]
    
    similarity = np.dot(emb1, emb2)
    distance = 1.0 - similarity
    
    print(f"\n   Distance between Track {track1_id} and Track {track2_id}: {distance:.3f}")
    print(f"   (Should be large ~0.5-1.0 if different people)")

print("\n" + "=" * 80)
print("✅ INSPECTION COMPLETE")
print("=" * 80)

print("\n💡 What to look for:")
print("   - Each track should have multiple embeddings (not just 1)")
print("   - Embeddings should be normalized (norm ~1.0)")
print("   - Same track embeddings should be similar (distance < 0.3)")
print("   - Different track embeddings should be different (distance > 0.5)")
print("   - Track IDs should span across frames (not all in same frame)")
