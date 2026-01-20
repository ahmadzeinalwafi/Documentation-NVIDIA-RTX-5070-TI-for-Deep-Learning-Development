import torch
import time
import traceback

def separator(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

separator("BASIC CUDA CHECK")

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (PyTorch):", torch.version.cuda)
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("cuDNN version:", torch.backends.cudnn.version())

if not torch.cuda.is_available():
    print("\n❌ CUDA not available. Exiting tests.")
    exit(0)

separator("DEVICE INFORMATION")

device_count = torch.cuda.device_count()
print("Number of CUDA devices:", device_count)

for i in range(device_count):
    props = torch.cuda.get_device_properties(i)
    print(f"\nGPU {i}: {props.name}")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"  Multi-Processor Count: {props.multi_processor_count}")

device = torch.device("cuda:0")
torch.cuda.set_device(device)

separator("MEMORY STATUS (INITIAL)")

def print_memory():
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    max_alloc = torch.cuda.max_memory_allocated(0) / 1024**2
    print(f"Allocated: {allocated:.2f} MB")
    print(f"Reserved:  {reserved:.2f} MB")
    print(f"Max Alloc: {max_alloc:.2f} MB")

print_memory()

separator("BASIC GPU COMPUTATION TEST")

try:
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)

    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()

    print("Matrix multiplication successful ✅")
    print(f"Execution time: {end - start:.3f} seconds")

except Exception as e:
    print("❌ GPU computation failed")
    traceback.print_exc()

separator("SIMPLE TRAINING TEST")

try:
    x = torch.randn(10_000, 100, device=device)
    y = torch.randn(10_000, 10, device=device)

    model = torch.nn.Sequential(
        torch.nn.Linear(100, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    print("Starting training...")
    print_memory()

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.6f}")
        print_memory()

    print("\n✅ Training completed successfully")

except Exception:
    print("❌ Training failed")
    traceback.print_exc()

separator("MEMORY STRESS TEST")

try:
    blocks = []
    for i in range(5):
        blocks.append(torch.randn(1024, 1024, 512, device=device))
        print(f"Allocated block {i+1}")
        print_memory()

    del blocks
    torch.cuda.empty_cache()

    print("\nAfter cleanup:")
    print_memory()

except Exception:
    print("❌ Memory stress test failed")
    traceback.print_exc()
