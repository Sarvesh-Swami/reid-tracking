"""
Demo script showing persistent person identity using embeddings
"""
import sys
from pathlib import Path
import numpy as np

# Add boxmot to path
sys.path.insert(0, str(Path(__file__).parent))

from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend
from boxmot.utils import WEIGHTS


def demo_embedding_matching():
    """Demonstrate embedding extraction and matching"""

    print("🔍 Demo: Persistent Person Identity using Embeddings")
    print("=" * 60)

    # Initialize ReID model
    reid_model = ReIDDetectMultiBackend(
        weights=WEIGHTS / 'osnet_x1_0_msmt17.pt',
        device='cpu',
        fp16=False
    )

    # Simulate embedding database
    embedding_db = {}
    next_person_id = 1
    similarity_threshold = 0.7

    def extract_embedding(image_crop):
        """Extract normalized embedding from image crop"""
        try:
            embedding = reid_model([image_crop])
            if hasattr(embedding, 'detach'):
                embedding = embedding.detach().cpu().numpy()
            embedding = embedding.flatten()
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            print(f"Error: {e}")
            return None

    def cosine_similarity(emb1, emb2):
        """Compute cosine similarity"""
        return np.dot(emb1, emb2)

    def match_person(embedding):
        """Match embedding against database"""
        best_id = None
        best_similarity = 0.0

        for person_id, stored_emb in embedding_db.items():
            similarity = cosine_similarity(embedding, stored_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = person_id

        if best_similarity >= similarity_threshold:
            return best_id, best_similarity
        return None, best_similarity

    def update_embedding(person_id, new_embedding):
        """Update stored embedding with weighted average"""
        if person_id in embedding_db:
            old_emb = embedding_db[person_id]
            updated_emb = 0.8 * old_emb + 0.2 * new_embedding
            # Re-normalize
            norm = np.linalg.norm(updated_emb)
            if norm > 0:
                updated_emb = updated_emb / norm
            embedding_db[person_id] = updated_emb
        else:
            embedding_db[person_id] = new_embedding

    # Create dummy image crops (in real usage, these would be person detections)
    print("\n📸 Simulating person detections...")

    # Create some dummy crops (random images for demo)
    dummy_crops = []
    for i in range(5):
        # Create random RGB image (64x128 for person-like aspect ratio)
        crop = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
        dummy_crops.append(crop)

    print(f"Created {len(dummy_crops)} dummy person detections")

    # Process detections
    print("\n🔄 Processing detections with persistent identity...")

    for i, crop in enumerate(dummy_crops):
        print(f"\nDetection {i+1}:")

        # Extract embedding
        embedding = extract_embedding(crop)
        if embedding is None:
            print("  ❌ Failed to extract embedding")
            continue

        print(f"  ✅ Extracted embedding (shape: {embedding.shape})")

        # Match against database
        matched_id, similarity = match_person(embedding)

        if matched_id is not None:
            print(f"  🔍 Matched existing person ID {matched_id} (similarity: {similarity:.3f})")
            person_id = matched_id
        else:
            print(f"  🆕 Created new person ID {next_person_id} (best similarity: {similarity:.3f})")
            person_id = next_person_id
            next_person_id += 1

        # Update stored embedding
        update_embedding(person_id, embedding)
        print(f"  💾 Updated embedding for person ID {person_id}")

    print("\n📊 Final Results:")
    print(f"  - Total persons in database: {len(embedding_db)}")
    print(f"  - Next person ID to assign: {next_person_id}")
    print(f"  - Similarity threshold: {similarity_threshold}")

    # Demonstrate persistence
    print("\n🔄 Testing persistence (re-using first detection):")
    first_crop = dummy_crops[0]
    embedding = extract_embedding(first_crop)
    matched_id, similarity = match_person(embedding)

    if matched_id is not None:
        print(f"  ✅ Same person detected again! ID {matched_id} (similarity: {similarity:.3f})")
        print("  🎉 Persistent identity working correctly!")
    else:
        print("  ❌ Persistence failed - this shouldn't happen")

    print("\n✨ Demo completed successfully!")


if __name__ == '__main__':
    import torch
    demo_embedding_matching()