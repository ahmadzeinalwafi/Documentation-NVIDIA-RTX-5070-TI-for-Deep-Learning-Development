import tensorflow as tf
import time
import traceback
import os

def separator(title):
    print("\n" + "="*60)
    print(title)
    print("="*60)

separator("BASIC CUDA CHECK")

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
gpu_available = len(gpus) > 0
print("GPU available:", gpu_available)
print("cuDNN enabled:", tf.test.is_built_with_cuda())

if not gpu_available:
    print("❌ No GPU found. Exiting tests.")
    exit(0)

separator("DEVICE INFORMATION")

for i, gpu in enumerate(gpus):
    print(f"\nGPU {i}: {gpu}")
    try:
        details = tf.config.experimental.get_device_details(gpu)
        cc = details.get("compute_capability", None)
        print(f"  Compute Capability: {cc}")
    except Exception:
        print("  Could not fetch detailed info for GPU", i)

separator("BASIC GPU COMPUTATION TEST")

try:
    with tf.device('/GPU:0'):
        a = tf.random.normal([4096, 4096])
        b = tf.random.normal([4096, 4096])

        start = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()  # force evaluation
        end = time.time()

    print("Matrix multiplication successful ✅")
    print(f"Execution time: {end - start:.3f} seconds")

except Exception:
    print("❌ GPU computation failed")
    traceback.print_exc()

separator("SIMPLE TRAINING TEST")

try:
    with tf.device('/GPU:0'):
        x = tf.random.normal([10_000, 100])
        y = tf.random.normal([10_000, 10])

        model = tf.keras.Sequential([
            tf.keras.Input(shape=(100,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        model.compile(optimizer='adam', loss='mse')

        print("Starting training...")
        model.fit(x, y, epochs=3, batch_size=128)
    print("\n✅ Training completed successfully")

except Exception:
    print("❌ Training failed")
    traceback.print_exc()

separator("MEMORY STRESS TEST")

try:
    blocks = []
    with tf.device('/GPU:0'):
        for i in range(5):
            blocks.append(tf.random.normal([1024, 1024, 512]))
            print(f"Allocated block {i+1}")

    del blocks
    print("\nAfter cleanup: memory should be freed (TensorFlow manages GPU memory automatically)")

except Exception:
    print("❌ Memory stress test failed")
    traceback.print_exc()
