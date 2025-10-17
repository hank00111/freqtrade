import torch

print("=" * 50)
print("PyTorch GPU Configuration Check")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("No CUDA support detected")
    print("Current PyTorch is CPU-only version")
    print("\nTo enable GPU training, you need to install PyTorch with CUDA:")
    print("  pip uninstall torch")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")

print("=" * 50)
