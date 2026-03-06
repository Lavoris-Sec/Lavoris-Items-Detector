import torch
import sys

def main():
    print("--- GPU Diagnostic Tool ---")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {'✅ YES' if cuda_available else '❌ NO'}")
    
    if cuda_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {round(torch.cuda.memory_allocated(0)/1024**3, 1)} GB")
        print(f"Memory Cached:    {round(torch.cuda.memory_reserved(0)/1024**3, 1)} GB")
    else:
        print("Suggestion: Install NVIDIA Drivers and CUDA Toolkit to speed up YOLO.")

if __name__ == "__main__":
    main()