# Base Image: NVIDIA PyTorch (Optimized for H100/Hopper)
# Includes: PyTorch 2.1+, CUDA 12.2+, cuDNN 8+
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set environment variables for non-interactive install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# H100 Architecture (Hopper is sm_90)
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Mamba-SSM (State Space Model)
# This requires CUDA compilation, handled by the base image's nvcc
RUN pip install mamba-ssm --no-cache-dir

# Install Triton-based CausalLM kernels (Optional but good for MoE/GQA)
RUN pip install causallm-triton --no-cache-dir

# Install other Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --no-cache-dir

# Set workspace
WORKDIR /workspace

# Copy source code
COPY . /workspace

# Default command
CMD ["/bin/bash"]
