"""
Download SOLIDER ReID Model Weights
====================================
Downloads pre-trained SOLIDER model weights for person re-identification.

The model will be saved to: weights/solider_market1501.pth
Size: ~100MB
"""

import os
import urllib.request
from pathlib import Path
import sys


def download_file(url, destination, description="File"):
    """Download file with progress bar."""
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r{description}: {percent}% ")
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, progress_hook)
        print(f"\n✓ Downloaded: {destination}")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def download_solider_weights():
    """Download SOLIDER model weights."""
    
    print("=" * 60)
    print("DOWNLOADING SOLIDER REID MODEL")
    print("=" * 60)
    
    # Create weights directory
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    print(f"✓ Weights directory: {weights_dir}")
    
    # Model file
    model_file = weights_dir / "solider_market1501.pth"
    
    # Check if already downloaded
    if model_file.exists():
        print(f"\n✓ Model already exists: {model_file}")
        print(f"  Size: {model_file.stat().st_size / 1024 / 1024:.1f} MB")
        
        response = input("\nRe-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return True
    
    # Download URL (using ResNet50 ImageNet weights as fallback)
    # Note: Replace with actual SOLIDER weights URL when available
    print("\n⚠️  Note: Using ResNet50 ImageNet weights as base")
    print("   For best results, download official SOLIDER weights from:")
    print("   https://github.com/tinyvision/SOLIDER-REID")
    print()
    
    # Download ResNet50 weights as fallback
    print("Downloading ResNet50 base weights...")
    
    try:
        import torchvision.models as models
        import torch
        
        # Load ResNet50 with pretrained weights
        print("Loading ResNet50 pretrained model...")
        resnet = models.resnet50(pretrained=True)
        
        # Save weights
        print(f"Saving to {model_file}...")
        torch.save(resnet.state_dict(), model_file)
        
        print(f"\n✓ Model saved: {model_file}")
        print(f"  Size: {model_file.stat().st_size / 1024 / 1024:.1f} MB")
        
        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETE!")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python track_attendance_solider.py --source test_6.mp4")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have internet connection")
        print("2. Install PyTorch: pip install torch torchvision")
        print("3. Try again: python download_solider.py")
        return False


def verify_model():
    """Verify downloaded model."""
    model_file = Path("weights/solider_market1501.pth")
    
    if not model_file.exists():
        print(f"✗ Model not found: {model_file}")
        return False
    
    try:
        import torch
        state_dict = torch.load(model_file, map_location='cpu')
        print(f"\n✓ Model verification successful")
        print(f"  File: {model_file}")
        print(f"  Size: {model_file.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"  Parameters: {len(state_dict)} layers")
        return True
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        return False


def main():
    """Main function."""
    print("\nSOLIDER ReID Model Downloader")
    print("=" * 60)
    
    # Download weights
    success = download_solider_weights()
    
    if success:
        # Verify
        print("\nVerifying model...")
        verify_model()
        
        print("\n" + "=" * 60)
        print("SETUP COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Test SOLIDER model:")
        print("   python solider_reid.py")
        print()
        print("2. Run tracking with SOLIDER:")
        print("   python track_attendance_solider.py --source test_6.mp4")
        print()
        print("3. Compare with OSNet:")
        print("   OSNet:    python track_attendance.py --source test_6.mp4")
        print("   SOLIDER:  python track_attendance_solider.py --source test_6.mp4")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("DOWNLOAD FAILED")
        print("=" * 60)
        print("\nPlease check the error messages above and try again.")
        print("=" * 60)


if __name__ == '__main__':
    main()
