# GPU Experiments

A collection of tests and benchmarks for my own research.

I originally created this repository to learn CUDA through hands-on coding. Now, I use it to run various micro-experiments that help me quickly develop new ideas.

## Requirements

- Thunderkittens
- NCU in $HOME/ncu
- CUDA in /usr/local/cuda-12
- LD_LIBRARY_PATH configured
- `uftrace` installed (https://uftrace.github.io/slide/#6)

Everything should be done through the Makefile.

Everything targets H100 (sm_90a)

For nccl, custom build was made with `make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90a"`, after adding debug/profiler flags (`-finstrument-functions -pg -g`) to the nccl makefile manually.
