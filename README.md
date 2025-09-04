# GPU Experiments

A collection of CUDA micro-experiments and benchmarks for my own research.

## Requirements

- ThunderKittens (included as a Git submodule)
- CUDA 12.8+
- Hopper (H100) or Blackwell (B200) GPUs
- Python 3 with PyTorch and pybind11

Build and execution are solely handled by the Makefiles in each subdirectory.

## Organization

- `hopper/`: CUDA experiments targeting H100
- `blackwell/`: CUDA experiments targeting Blackwell

## Topics

- A lot of random micro-experiments 
- Multi-GPU
- MXFP8
- IPC
- Attention
