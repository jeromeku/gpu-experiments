import time
import sys

import numpy as np
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

from flash_attn.cute.interface import FlashAttnFunc

# Global parameters
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
H = 128
D_qk = 192
D_vo = 128
print(f"N: {N}, H: {H}, D_qk: {D_qk}, D_vo: {D_vo}")

# Config
CHECK_CORRECTNESS = False
BENCHMARK = True
NUM_WARMUPS = 5
NUM_ITERS = 10

# Generate inputs
Q = torch.randn(1, N, H, D_qk, dtype=torch.bfloat16, device="cuda")
K = torch.randn(1, N, H, D_qk, dtype=torch.bfloat16, device="cuda")
V = torch.randn(1, N, H, D_vo, dtype=torch.bfloat16, device="cuda")
softmax_scale = 1 / (D_qk ** 0.5)


def pytorch_attn_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        S = Q.permute(0, 2, 1, 3) @ K.permute(0, 2, 3, 1) * softmax_scale
        mask = torch.triu(torch.ones(N, N, device=S.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
        P = S.softmax(dim=-1)
        O = P @ V.permute(0, 2, 1, 3)
        return O.permute(0, 2, 1, 3)


def flash_attn_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        O, _ = FlashAttnFunc.apply(Q, K, V, softmax_scale, True) # always causal
        return O


# Check correctness
if CHECK_CORRECTNESS:
    # Diff checking utility
    def check_diff(name, A, A_ref):
        print(f"\n{name}")
        print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
        print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
        print(f"Mean:      {A.abs().mean().item():.10f}")
        print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")
        print(f"Max:       {A.abs().max().item():.10f}")
        print(f"Ref max:   {A_ref.abs().max().item():.10f}")

    # Flash MHA
    O_flash = flash_attn_fwd(Q, K, V)

    # Pytorch MHA
    O_ref = pytorch_attn_fwd(Q, K, V)

    # Check results
    check_diff("O (flash)", O_flash, O_ref)

# Benchmark
if BENCHMARK:
    l2_cache = torch.empty(1024 * 1024 * 64, dtype=torch.bfloat16, device="cuda")

    for i in range(NUM_WARMUPS):
        flash_attn_fwd(Q, K, V)
    torch.cuda.synchronize()

    times = []

    for i in range(NUM_ITERS):
        l2_cache.random_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        flash_attn_fwd(Q, K, V)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = np.mean(times)
    std_time = np.std(times)
    flops = 2 * H * (N * (N + 1) / 2) * (D_qk + D_vo)
    tflops = flops * 1e-12

    print(f'Time taken: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us')
    print(f'TFLOPS: {tflops / avg_time:.2f} TFLOP/s')
