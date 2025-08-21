import torch

from _C import kernel


M = 16384
N = 16384

A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

kernel(A)
torch.cuda.synchronize()
