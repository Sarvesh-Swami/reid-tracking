"""
Final fix for transformers compatibility
Installs the exact version that works with both PyTorch 2.0.1 and Phi-3
"""

import sys
import subprocess

print("=" * 60)
print("FINAL FIX - INSTALLING CORRECT TRANSFORMERS VERSION")
print("=" * 60)

# Check PyTorch
print("\n[1/3] Checking PyTorch...")
try:
    import torch
    print(f"      ✓ PyTorch {torch.__version__}")
except ImportError:
    print("      ✗ PyTorch not found!")
    exit(1)

# Install correct transformers version
print("\n[2/3] Installing transformers 4.37.2...")
print("      (This version has cache_utils AND works with PyTorch 2.0.1)")
print("      (This may take 1-2 minutes...)")

# Uninstall first
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "transformers", "-y"],
    capture_output=True
)

# Install specific version that works
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "transformers==4.37.2", "accelerate"],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print("      ✗ Failed to install")
    print(result.stderr)
    exit(1)

print("      ✓ transformers 4.37.2 installed")

# Verify
print("\n[3/3] Verifying...")
try:
    # Force reimport
    if 'transformers' in sys.modules:
        del sys.modules['transformers']
    
    import transformers
    print(f"      ✓ transformers {transformers.__version__}")
    
    # Check for cache_utils
    from transformers import cache_utils
    print("      ✓ cache_utils available")
    
    # Check model classes
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("      ✓ Model classes available")
    
    print("\n" + "=" * 60)
    print("SUCCESS! NOW RUN THE DOWNLOAD:")
    print("=" * 60)
    print("\n  python download_phi3_only.py")
    print("\n" + "=" * 60)
    
except Exception as e:
    print(f"      ✗ Verification failed: {e}")
    print("\nTrying alternative version...")
    
    # Try 4.38.0
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "transformers==4.38.0", "--force-reinstall"],
        capture_output=True
    )
    
    print("\nPlease run this script again:")
    print("  python final_fix.py")
    exit(1)
