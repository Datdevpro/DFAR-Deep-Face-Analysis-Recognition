import torch

# Check if CUDA (NVIDIA GPU) is available
if torch.cuda.is_available():
    print(f"CUDA is available! Found {torch.cuda.device_count()} GPU(s).")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    # Check for Apple Silicon (M1/M2/M3)
    if torch.backends.mps.is_available():
        print("Apple MPS (Metal Performance Shaders) is available.")
    else:
        print("GPU not detected. Using CPU.")