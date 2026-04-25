"""
Simple test script to download Phi-3 model
Run this from your venv: python test_phi3_download.py
"""

import sys
import os

print("=" * 60)
print("PHI-3 MODEL DOWNLOAD TEST")
print("=" * 60)

# Check torch FIRST (before transformers)
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("✗ PyTorch NOT installed")
    print("\nThis shouldn't happen - you have PyTorch!")
    exit(1)

# Check transformers version compatibility
try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
    
    # Check if version is too new for PyTorch 2.0.1
    from packaging import version
    tf_version = version.parse(transformers.__version__)
    
    if tf_version >= version.parse("4.36.0"):
        print("\n⚠️  WARNING: transformers version too new for PyTorch 2.0.1")
        print("   Current: transformers", transformers.__version__)
        print("   Required: transformers<4.36.0")
        print("\n⚠️  FIXING: Downgrading transformers...")
        print("   (This will take 1-2 minutes)")
        
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "transformers<4.36.0", "--force-reinstall"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ Fixed! Please run this script again:")
            print("  python test_phi3_download.py")
        else:
            print("✗ Auto-fix failed. Please run manually:")
            print("  python -m pip uninstall transformers -y")
            print("  python -m pip install 'transformers<4.36.0' accelerate")
        exit(0)
        
except ImportError:
    print("✗ transformers NOT installed")
    print("\nPlease run:")
    print("  python -m pip install 'transformers<4.36.0' accelerate")
    exit(1)

# Now try to import model classes
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✓ Model classes available")
except Exception as e:
    print(f"✗ Error importing model classes: {e}")
    print("\n⚠️  FIXING: Installing compatible transformers version...")
    print("   (This will take 1-2 minutes)")
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "transformers<4.36.0", "--force-reinstall", "--no-deps"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Fixed! Please run this script again:")
        print("  python test_phi3_download.py")
    else:
        print("✗ Auto-fix failed. Please run manually:")
        print("  python -m pip uninstall transformers -y")
        print("  python -m pip install 'transformers<4.36.0' accelerate")
    exit(1)

# Download model
model_name = 'microsoft/Phi-3-mini-4k-instruct'

print(f"\nDownloading model: {model_name}")
print("Size: ~4GB (one-time download)")
print("This may take 10-30 minutes...")
print("NO AUTHENTICATION REQUIRED!")
print("=" * 60)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("\n1. Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print("   ✓ Tokenizer downloaded")
    
    print("\n2. Downloading model weights...")
    print("   (This will take 10-30 minutes...)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("   ✓ Model downloaded")
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print("\nYou can now run:")
    print("  python track_gemma_local.py --source test_6.mp4 --output output_phi3.mp4")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Make sure you're in venv: venv\\Scripts\\activate")
    print("3. Install dependencies: python -m pip install transformers accelerate")
