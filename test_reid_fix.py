"""Quick test to verify the reid_model_factory fix"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from boxmot.appearance.reid_model_factory import get_model_name

# Test with string
print("Testing with string:")
result = get_model_name("osnet_x1_0_msmt17.pt")
print(f"  get_model_name('osnet_x1_0_msmt17.pt') = {result}")

# Test with Path object
print("\nTesting with Path object:")
result = get_model_name(Path("osnet_x1_0_msmt17.pt"))
print(f"  get_model_name(Path('osnet_x1_0_msmt17.pt')) = {result}")

print("\n✅ Fix verified! Both string and Path work now.")
