import torch
torch.random.manual_seed(42)

# Import our Python bindings
from _C import kernel

# Matrix dimensions (should not change)
M = 512
K = 128
N = 256

print('Generating inputs...')
A = (torch.randn(M, K, dtype=torch.bfloat16, device="cuda:0") / K ** 0.25).to(torch.float8_e4m3fn)
B = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda:0") / K ** 0.25).to(torch.float8_e4m3fn)
C = torch.zeros(M, N, dtype=torch.float16, device="cuda:0")

print('Launching kernel...')
kernel(A, B, C)
torch.cuda.synchronize()

# Check correctness
C_ref = torch.matmul(A.to(torch.float16), B.T.to(torch.float16))
assert C_ref.dtype == C.dtype
abs_diff = torch.abs(C_ref - C)
print(f"Max absolute difference: {abs_diff.max()}")
print(f"Mean absolute difference: {abs_diff.mean()}")

breakpoint()