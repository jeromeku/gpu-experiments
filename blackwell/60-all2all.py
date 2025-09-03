"""
torchrun --nproc_per_node=8 60-all2all.py
"""

from datetime import timedelta
import os
import sys
import time

import numpy as np
import torch
import torch.distributed
torch.set_printoptions(sci_mode=False)


def all2all_4D(
    input: torch.tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group: torch.distributed.ProcessGroup | None = None,
    sync: bool = False
) -> torch.tensor:
    """
    all-to-all collective operation on a 4D tensor. Gathers on gather_idx, scatters on scatter_idx.

    Args:
        input: torch.tensor sharded on gather_idx
        scatter_idx: index to scatter
        gather_idx: index to gather
        group: torch process group
        sync: whether to synchronize after all-to-all

    Returns:
        torch.tensor: torch.tensor gathered on gather_idx and sharded on scatter_idx
    """
    assert input.dim() == 4, f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    num_ranks = torch.distributed.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        B, N_per_rank, H, D = input.shape
        N = N_per_rank * num_ranks
        H_per_rank = H // num_ranks

        input_t = input.view(B, N_per_rank, num_ranks, H_per_rank, D).permute(2, 1, 0, 3, 4).contiguous()
        
        if num_ranks > 1:
            output = torch.empty_like(input_t)
            torch.distributed.all_to_all_single(output, input_t, group=group)
            if sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        return output.permute(2, 0, 1, 3, 4).reshape(B, N_per_rank * num_ranks, H_per_rank, D)
    
    elif scatter_idx == 1 and gather_idx == 2:
        B, N, H_per_rank, D = input.shape
        H = H_per_rank * num_ranks
        N_per_rank = N // num_ranks

        input_t = input.view(B, num_ranks, N_per_rank, H_per_rank, D).permute(1, 3, 2, 0, 4).contiguous()
        
        if num_ranks > 1:
            output = torch.empty_like(input_t)
            torch.distributed.all_to_all_single(output, input_t, group=group)
            if sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        return output.permute(3, 2, 0, 1, 4).reshape(B, N_per_rank, H_per_rank * num_ranks, D)
    
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


# Parameters
N = 65536
H = 128
D_qk = 192
D_vo = 128

# Initialize distributed environment
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
world_size = int(os.environ.get("WORLD_SIZE", 1))
assert world_size == local_world_size, "multi-node runs are not supported"
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)
torch.distributed.init_process_group(
    backend="nccl",
    device_id=local_rank,
    rank=rank,
    world_size=world_size,
    timeout=timedelta(seconds=30),
)

# Print global parameters
if rank == 0:
    print(f"N: {N}, H: {H}, D_qk: {D_qk}, D_vo: {D_vo}")

# Generate inputs
torch.random.manual_seed(rank + 42)
A = torch.randn(1, N // world_size, H, D_qk, dtype=torch.bfloat16, device=device)

# Run PyTorch reference
all2all_4D(A, scatter_idx=2, gather_idx=1)
torch.cuda.synchronize()

# Benchmark
NUM_WARMUPS = 5
NUM_ITERS = 10
chunk_size = (N // world_size) * (H // world_size) * D_qk * 2 # chunk sent from one rank to another
per_rank_comm_size = chunk_size * (world_size - 1)
total_comm_size = per_rank_comm_size * world_size # N * (N - 1) * chunk_size
l2_cache = torch.empty(1024 * 1024 * 64, dtype=torch.bfloat16, device=device)

for i in range(NUM_WARMUPS):
    all2all_4D(A, scatter_idx=2, gather_idx=1)
torch.distributed.barrier()
torch.cuda.synchronize()

times = []

for i in range(NUM_ITERS):
    l2_cache.random_()
    torch.distributed.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    all2all_4D(A, scatter_idx=2, gather_idx=1)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)

avg_time = np.mean(times)
std_time = np.std(times)

if rank == 0:
    print("B200 Max Unidirectional NVL Bandwidth: 900 GB/s")
    print(f"Per-rank data sent: {per_rank_comm_size * 1e-9:.2f} GB")

for i in range(world_size):
    if i == rank:
        print(f'Time taken: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us')
        print(f'Unidirectional Bandwidth, USP: {per_rank_comm_size * 1e-9 / avg_time:.2f} GB/s')
    torch.distributed.barrier()

# Clean up
torch.distributed.destroy_process_group()
