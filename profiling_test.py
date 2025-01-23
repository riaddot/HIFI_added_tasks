import time
import torch

# Simulate CPU-based computations with time module
def cpu_computation():
    start_time = time.time()
    result = sum([i ** 2 for i in range(10**6)])  # Simulated computation
    end_time = time.time()
    cpu_time = (end_time - start_time) * 1000  # Time in milliseconds
    print(f"CPU computation time: {cpu_time:.3f} ms")

# Simulate GPU-based computations with torch.cuda.Event
def gpu_computation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create random tensor on the GPU
    tensor_a = torch.randn((1000, 1000), device=device)
    tensor_b = torch.randn((1000, 1000), device=device)

    # Events for profiling
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Record time before computation
    start_event.record()
    
    # Simulated computation (matrix multiplication on GPU)
    result = torch.mm(tensor_a, tensor_b)
    
    # Record time after computation
    end_event.record()

    # Wait for all operations to finish
    torch.cuda.synchronize()
    
    # Calculate elapsed time
    gpu_time = start_event.elapsed_time(end_event)  # Time in milliseconds
    print(f"GPU computation time: {gpu_time:.3f} ms")

# Run simulations
cpu_computation()
if torch.cuda.is_available():
    gpu_computation()
else:
    print("CUDA is not available on this device.")



import cProfile

# Example of profiling with PyTorch
import torch

# Simulate operations
# Simulate operations
def simulate_operations():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulated operations
    tensor_a = torch.randn((1000, 1000), device=device)
    tensor_b = torch.randn((1000, 1000), device=device)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        with torch.autograd.profiler.record_function("Matrix Multiplication"):
            result = torch.mm(tensor_a, tensor_b)

        with torch.autograd.profiler.record_function("Element-wise Addition"):
            result = result + tensor_a

        with torch.autograd.profiler.record_function("ReLU"):
            result = torch.relu(result)

    # Print out the table sorted by CPU/GPU time
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))  # For CPU time
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))  # For GPU time if applicable

simulate_operations()



