# NVIDIA RTX 5070 Ti Deep Learning Development Guide
## Description
This repository serves as a technical blueprint for configuring the NVIDIA RTX 5070 Ti for AI development. It addresses the unique requirements of the Blackwell architecture (Compute Capability 12.0), providing stable configurations for CUDA 12.9, tf-nightly, and PyTorch 2.8. It includes diagnostic scripts and hybrid CNN-LSTM examples to verify hardware acceleration and memory management across both Windows and WSL2 environments.

## Instalation
- [NVCC - CUDA](https://developer.nvidia.com/cuda-12-9-0-download-archive)
- [Tensorflow](https://www.tensorflow.org/install/pip#nightly)
- [Pytorch](https://pytorch.org/get-started/locally/)
- [cuDNN](https://developer.nvidia.com/cudnn-downloads)

## Version
### WSL
#### Python
```
Python 3.10.12
```
#### NVCC
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Apr__9_19:24:57_PDT_2025
Cuda compilation tools, release 12.9, V12.9.41
Build cuda_12.9.r12.9/compiler.35813241_0
```
#### TF Nightly
```
Name: tf_nightly
Version: 2.21.0.dev20260113
Summary: TensorFlow is an open source machine learning framework for everyone.
Home-page: https://www.tensorflow.org/
Author: Google Inc.
Author-email: packages@tensorflow.org
License: Apache 2.0
Location: /home/ahmad/.local/lib/python3.10/site-packages
Requires: absl-py, astunparse, flatbuffers, gast, google_pasta, grpcio, h5py, keras-nightly, libclang, ml_dtypes, numpy, opt_einsum, packaging, protobuf, requests, setuptools, six, tb-nightly, termcolor, typing_extensions, wrapt
Required-by:
```
#### CUDA NVIDIA
```
NVIDIA-SMI 575.57.04              Driver Version: 576.52         CUDA Version: 12.9 
```

### Windows
#### Python
```
Python 3.10.12
```
#### NVCC
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Apr__9_19:24:57_PDT_2025
Cuda compilation tools, release 12.9, V12.9.41
Build cuda_12.9.r12.9/compiler.35813241_0
```
#### Pytorch
```
Name: torch
Version: 2.8.0+cu129
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3-Clause
Location: C:\Users\dwork\AppData\Local\Programs\Python\Python311\Lib\site-packages
Requires: filelock, fsspec, jinja2, networkx, sympy, typing-extensions
Required-by: torchvision
```
#### CUDA NVIDIA
```
NVIDIA-SMI 576.52                 Driver Version: 576.52         CUDA Version: 12.9 
```
## Tensorflow
### Compatibility for Tensorflow Version > 2.10
As described in documentation [Install TensorFlow with pip](https://www.tensorflow.org/install/pip#nightly) and discussed in issue [What is preventing TF to use GPU when used in native windows? #69750
](https://github.com/tensorflow/tensorflow/issues/69750), Tensorflow > 2.10 does not support windows anymore to running using available GPU. As described in [Install TensorFlow with pip](https://www.tensorflow.org/install/pip#nightly) that TensorFlow with GPU access is supported for WSL2 on Windows 10 19044 or higher, the example will use WSL Ubuntu-22.04. Please run the WSL using
```
wsl -d Ubuntu-22.04
```
The windows drive will be available in path
```
/mnt
```

### Basic Check
This script performs basic GPU tests using TensorFlow. It checks TensorFlow and CUDA/cuDNN versions, detects available GPUs, and prints device information. It then runs a matrix multiplication on the GPU to verify computation, performs a simple training loop with a small dense neural network, and conducts a memory stress test by allocating large tensors. All steps report success or failure, and GPU memory management is handled automatically by TensorFlow. This is for testing if LSTM, CNN, and NN really works
```
python 3 tensorflow/basic_check.py
```
### Example MNIST
This script trains a hybrid CNN-LSTM model on the MNIST dataset using TensorFlow/Keras. It normalizes the data, reshapes it for CNN input, and builds a model with convolutional and pooling layers for feature extraction, followed by LSTM layers for sequence modeling, and dense layers for classification. The model is trained for a few epochs with Adam optimizer and categorical cross-entropy loss, reporting training time, and final test loss and accuracy.
```
python 3 tensorflow/mnist_example.py
```
## Pytorch
### Basic Check
This script performs a series of basic CUDA and GPU tests using PyTorch. It first checks the PyTorch version, CUDA availability, CUDA/cuDNN versions, and device properties. It then reports GPU memory usage, runs a basic matrix multiplication on the GPU to verify computation, and performs a small training loop with a simple neural network to test model training on the GPU. Finally, it conducts a memory stress test by allocating large tensors, reports memory usage at each step, and cleans up to ensure memory is freed. Errors in any step are caught and printed.
```
python 3 pytorch/basic_check.py
```
### Example MNIST
This script trains a hybrid CNN-LSTM model on the MNIST dataset using PyTorch. It automatically selects the available device (CUDA, MPS, or CPU) and prepares the data with normalization and DataLoaders. The model extracts features with convolutional layers, processes sequences with LSTM layers, and outputs class predictions. Training runs for a few epochs with Adam optimizer and cross-entropy loss, reporting loss, accuracy, and elapsed time for both training and validation. If using a GPU, it also prints memory usage after training. This is for testing if LSTM, CNN, and NN really works
```
python 3 pytorch/mnist_example.py
```

## Issue
### Official Tensorflow Compatibility with Device RTX 50xx
The basic script for reproducible is below
```
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
It will print message like this
```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1768485307.091684     549 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1768485307.636426     549 cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1768485308.759699     549 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
W0000 00:00:1768485309.633310     549 gpu_device.cc:2459] TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0a. CUDA kernels will be jit-compiled from PTX, which could take 30 minutes or longer.
W0000 00:00:1768485309.637092     549 gpu_device.cc:2459] TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0a. CUDA kernels will be jit-compiled from PTX, which could take 30 minutes or longer. 
I0000 00:00:1768485309.812206     549 gpu_device.cc:2043] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13123 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 5070 Ti, pci bus id: 0000:01:00.0, compute capability: 12.0a
tf.Tensor(-2029.3302, shape=(), dtype=float32)
```

Please highly note on the message 
```
W0000 00:00:1768485309.633310     549 gpu_device.cc:2459] TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0a. CUDA kernels will be jit-compiled from PTX, which could take 30 minutes or longer.
```
As discussed in issue [TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0 CUDA_ERROR_INVALID_HANDLE #97387](https://github.com/tensorflow/tensorflow/issues/97387) the issue is on the tensorflow itself. If using official tensorflow per 15/1/2025 on version 2.20 it will produce error matmul like below

```
W0000 00:00:1768485113.698150 475 gpu_device.cc:2431] TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0. CUDA kernels will be jit-compiled from PTX, which could take 30 minutes or longer. I0000 00:00:1768485113.875307 475 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13123 MB memory: -> device: 0, name: NVIDIA GeForce RTX 5070 Ti, pci bus id: 0000:01:00.0, compute capability: 12.0 2026-01-15 20:51:53.962048: W tensorflow/compiler/mlir/tools/kernel_gen/tf_gpu_runtime_wrappers.cc:40] 'cuModuleLoadData(&module, data)' failed with 'CUDA_ERROR_INVALID_PTX' 2026-01-15 20:51:53.962118: W tensorflow/compiler/mlir/tools/kernel_gen/tf_gpu_runtime_wrappers.cc:40] 'cuModuleGetFunction(&function, module, kernel_name)' failed with 'CUDA_ERROR_INVALID_HANDLE' 2026-01-15 20:51:53.962520: W tensorflow/core/framework/op_kernel.cc:1842] INTERNAL: 'cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, 0, reinterpret_cast<CUstream>(stream), params, nullptr)' failed with 'CUDA_ERROR_INVALID_HANDLE' 2026-01-15 20:51:53.962557: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: INTERNAL: 'cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, 0, reinterpret_cast<CUstream>(stream), params, nullptr)' failed with 'CUDA_ERROR_INVALID_HANDLE' Traceback (most recent call last): File "/mnt/e/test-deep-learning-library/check.py", line 13, in <module> a = tf.random.normal((1024, 1024)) File "/home/ahmad/.local/lib/python3.10/site-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler raise e.with_traceback(filtered_tb) from None File "/home/ahmad/.local/lib/python3.10/site-packages/tensorflow/python/framework/ops.py", line 6027, in raise_from_not_ok_status raise core._status_to_exception(e) from None # pylint: disable=protected-access tensorflow.python.framework.errors_impl.InternalError: {{function_node __wrapped__Mul_device_/job:localhost/replica:0/task:0/device:GPU:0}} 'cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, 0, reinterpret_cast<CUstream>(stream), params, nullptr)' failed with 'CUDA_ERROR_INVALID_HANDLE' [Op:Mul] name:
```
However, per 15/1/2025, tf-nightly can solve the issue. The issue most likely due to not support on RTX 50xx with compute capability 12.0

### RESOURCE_EXHAUSTED
TensorFlow’s default allocator (BFCAllocator) is very aggressive with large allocations. Session hit the RESOURCE_EXHAUSTED error when trying to allocate multiple ~2 GiB blocks. Please try:
```
export TF_GPU_ALLOCATOR=cuda_malloc_async
```
### NVCC not found
Please add the CUDA to the PATH
```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```