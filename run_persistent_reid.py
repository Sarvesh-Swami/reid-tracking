"""
Master script to run all 3 steps of persistent ReID
"""
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from step1_extract_embeddings import extract_embeddings
from step2_cluster_tracks import cluster_tracks
from step3_gallery_tracking import gallery_tracking

def run_all_steps(video_path, reid_model='osnet_x1_0_msmt17.pt', 
                  cluster_threshold=0.4, match_threshold=0.45,
                  output_video=None):
    """
    Run all 3 steps of persistent ReID pipeline
    
    Parameters:
        video_path: Input video file
        reid_model: ReID model to use
        cluster_threshold: Distance threshold for clustering (step 2)
        match_threshold: Distance threshold for matching (step 3)
        output_video: Output video file (default: video_name_persistent.mp4)
    """
    video_name = Path(video_path).stem
    
    if output_video is None:
        output_video = f"{video_name}_persistent.mp4"
    
    embeddings_file = f"{video_name}_embeddings.pkl"
    gallery_file = f"{video_name}_gallery.pkl"
    
    print("\n" + "=" * 80)
    print("PERSISTENT REID PIPELINE - 3 PASS APPROACH")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"ReID Model: {reid_model}")
    print(f"Cluster Threshold: {cluster_threshold}")
    print(f"Match Threshold: {match_threshold}")
    print(f"Output: {output_video}")
    print("=" * 80)
    print()
    
    # Step 1: Extract embeddings
    print("\n🔹 STEP 1/3: Extracting embeddings...")
    extract_embeddings(video_path, embeddings_file, reid_model)
    
    # Step 2: Cluster tracks
    print("\n🔹 STEP 2/3: Clustering tracks...")
    cluster_tracks(embeddings_file, gallery_file, cluster_threshold)
    
    # Step 3: Gallery tracking
    print("\n🔹 STEP 3/3: Gallery-based tracking...")
    gallery_tracking(video_path, gallery_file, output_video, reid_model, match_threshold)
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Generated files:")
    print(f"   - {embeddings_file} (step 1 output)")
    print(f"   - {gallery_file} (step 2 output)")
    print(f"   - {output_video} (final output)")
    print("\n💡 To adjust results:")
    print(f"   - Lower cluster_threshold ({cluster_threshold}) = fewer persons (stricter merging)")
    print(f"   - Higher cluster_threshold = more persons (looser merging)")
    print(f"   - Lower match_threshold ({match_threshold}) = stricter matching in step 3")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run persistent ReID pipeline (3-pass approach)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_persistent_reid.py --video test_6.mp4
  
  # Adjust thresholds
  python run_persistent_reid.py --video test_6.mp4 --cluster-threshold 0.35 --match-threshold 0.40
  
  # Use different ReID model
  python run_persistent_reid.py --video test_6.mp4 --reid-model mobilenetv2_x1_4_dukemtmcreid.pt
        """
    )
    
    parser.add_argument('--video', type=str, required=True, 
                        help='Path to video file')
    parser.add_argument('--reid-model', type=str, default='osnet_x1_0_msmt17.pt',
                        help='ReID model (default: osnet_x1_0_msmt17.pt)')
    parser.add_argument('--cluster-threshold', type=float, default=0.4,
                        help='Clustering threshold - lower = stricter (default: 0.4)')
    parser.add_argument('--match-threshold', type=float, default=0.45,
                        help='Matching threshold - lower = stricter (default: 0.45)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video file (default: <video_name>_persistent.mp4)')
    
    args = parser.parse_args()
    
    run_all_steps(
        args.video,
        args.reid_model,
        args.cluster_threshold,
        args.match_threshold,
        args.output
    )
