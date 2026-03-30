"""
Simple test to verify persistent ReID is working
"""
import numpy as np

# Test the persistent storage directly
print("Testing Persistent ReID Implementation...")
print("=" * 60)

# Simulate embeddings
emb1 = np.random.randn(512)
emb2 = emb1 + np.random.randn(512) * 0.1  # Similar to emb1
emb3 = np.random.randn(512)  # Different person

# Normalize
emb1 = emb1 / np.linalg.norm(emb1)
emb2 = emb2 / np.linalg.norm(emb2)
emb3 = emb3 / np.linalg.norm(emb3)

# Test distances
dist_same = 1.0 - np.dot(emb1, emb2)
dist_diff = 1.0 - np.dot(emb1, emb3)

print(f"\n1. Embedding Distance Test:")
print(f"   Same person (different pose): {dist_same:.3f}")
print(f"   Different person: {dist_diff:.3f}")
print(f"   ✓ Same person should be < 0.5")
print(f"   ✓ Different person should be > 0.5")

# Test the DeepOCSORT persistent storage
print(f"\n2. Testing DeepOCSORT Persistent Storage:")

from boxmot.trackers.deepocsort.deep_ocsort import DeepOCSort

tracker = DeepOCSort(
    model_weights='osnet_x0_25_msmt17.pt',
    device='cpu',
    fp16=False,
    max_age=30
)

print(f"   Initial state:")
print(f"   - Persistent embeddings: {len(tracker.persistent_embeddings)}")
print(f"   - Deleted IDs: {len(tracker.deleted_track_ids)}")

# Store some embeddings
tracker._store_persistent_embedding(1, emb1)
tracker._store_persistent_embedding(2, emb3)

print(f"\n   After storing 2 embeddings:")
print(f"   - Persistent embeddings: {len(tracker.persistent_embeddings)}")
print(f"   - IDs stored: {list(tracker.persistent_embeddings.keys())}")

# Mark ID 1 as deleted
tracker._mark_track_deleted(1)

print(f"\n   After marking ID 1 as deleted:")
print(f"   - Deleted IDs: {list(tracker.deleted_track_ids)}")

# Try to find match for similar embedding
matched_id, distance = tracker._find_matching_deleted_id(emb2, threshold=0.5)

print(f"\n   Searching for match with similar embedding:")
print(f"   - Matched ID: {matched_id}")
print(f"   - Distance: {distance:.3f if distance else 'N/A'}")

if matched_id == 1:
    print(f"   ✅ SUCCESS: Correctly matched to ID 1!")
else:
    print(f"   ❌ FAILED: Should have matched to ID 1")

# Try with different embedding
matched_id2, distance2 = tracker._find_matching_deleted_id(emb3, threshold=0.5)

print(f"\n   Searching for match with different embedding:")
print(f"   - Matched ID: {matched_id2}")
print(f"   - Distance: {distance2:.3f if distance2 else 'N/A'}")

if matched_id2 is None:
    print(f"   ✅ SUCCESS: Correctly rejected (ID 2 not deleted)")
else:
    print(f"   ❌ FAILED: Should not have matched")

print(f"\n" + "=" * 60)
print(f"Test Complete!")
