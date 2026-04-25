"""
Simple Phi-3 download script (after transformers is fixed)
"""

import torch

print("=" * 60)
print("PHI-3 MODEL DOWNLOAD")
print("=" * 60)

# Verify setup
print("\nVerifying setup...")
try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import cache_utils
    print(f"✓ transformers {transformers.__version__}")
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print("✓ All dependencies ready")
except Exception as e:
    print(f"✗ Setup incomplete: {e}")
    print("\nPlease run first:")
    print("  python final_fix.py")
    exit(1)

# Download model
model_name = 'microsoft/Phi-3-mini-4k-instruct'

print(f"\nDownloading model: {model_name}")
print("Size: ~4GB (one-time download)")
print("This will take 10-30 minutes...")
print("=" * 60)

try:
    print("\n[1/2] Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print("      ✓ Tokenizer downloaded")
    
    print("\n[2/2] Downloading model weights (~4GB)...")
    print("      Progress will be shown below:")
    print()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    print("\n      ✓ Model downloaded")
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print("\nYou can now track videos:")
    print("  python track_gemma_local.py --source test_6.mp4 --output output_phi3.mp4")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Make sure transformers is correct version:")
    print("   python final_fix.py")
    print("3. Try again:")
    print("   python download_phi3_only.py")
    exit(1)
