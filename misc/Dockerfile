FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install necessary packages
RUN apt update && \
    apt install -y wget make git cmake ninja-build

# Add NVIDIA package key and install full CUDA toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-4 && \
    rm -f cuda-keyring_1.1-1_all.deb && \
    apt-get -y install cudnn9-cuda-12
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install Python and pip
RUN apt update && \
    apt install python3 python3-dev python3-pip -y && \
    pip install 'jax[cuda12]' 'numpy<2' einops flax pybind11 packaging ninja matplotlib graphviz tqdm scikit-learn && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install flash-attn --no-build-isolation
# For B200: pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Clone ThunderKittens and switch branch
RUN git clone https://github.com/HazyResearch/ThunderKittens

# Set working directory
WORKDIR /ThunderKittens

# Default command
CMD ["bash"]
