"""
One-command fix and download script
Automatically fixes transformers compatibility and downloads Phi-3
"""

import sys
import subprocess

print("=" * 60)
print("PHI-3 AUTO-FIX AND DOWNLOAD")
print("=" * 60)

# Step 1: Check PyTorch
print("\n[1/4] Checking PyTorch...")
try:
    import torch
    print(f"      ✓ PyTorch {torch.__version__}")
    print(f"      ✓ CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("      ✗ PyTorch not found!")
    exit(1)

# Step 2: Fix transformers version
print("\n[2/4] Fixing transformers version...")
print("      (This may take 1-2 minutes...)")

result = subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "transformers", "-y"],
    capture_output=True,
    text=True
)

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "transformers<4.36.0", "accelerate"],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print("      ✗ Failed to install transformers")
    print(result.stderr)
    exit(1)

print("      ✓ transformers fixed")

# Step 3: Verify
print("\n[3/4] Verifying installation...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    print(f"      ✓ transformers {transformers.__version__}")
    print("      ✓ Model classes available")
except Exception as e:
    print(f"      ✗ Verification failed: {e}")
    exit(1)

# Step 4: Download Phi-3 model
print("\n[4/4] Downloading Phi-3 model...")
print("      Size: ~4GB (one-time download)")
print("      This will take 10-30 minutes...")
print("      NO AUTHENTICATION REQUIRED!")
print()

model_name = 'microsoft/Phi-3-mini-4k-instruct'

try:
    print("      Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print("      ✓ Tokenizer downloaded")
    
    print("\n      Downloading model weights (~4GB)...")
    print("      (Progress will be shown below)")
    print()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("\n      ✓ Model downloaded")
    
    print("\n" + "=" * 60)
    print("SUCCESS! EVERYTHING IS READY!")
    print("=" * 60)
    print("\nYou can now run:")
    print("  python track_gemma_local.py --source test_6.mp4 --output output_phi3.mp4")
    print("=" * 60)
    
except Exception as e:
    print(f"\n      ✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Try again: python fix_and_download.py")
    print("3. See FIX_TRANSFORMERS.md for manual steps")
    exit(1)
