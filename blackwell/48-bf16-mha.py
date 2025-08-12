import numpy as np
import torch
torch.manual_seed(42)

# Import our Python bindings
from _C import fwd_attend_ker_128_noncausal, bwd_attend_prep_ker_128, bwd_attend_ker_128_noncausal


def check_diff(name, A, A_ref):
    print(f"\n{name}")
    print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
    print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
    print(f"Mean:      {A.abs().mean().item():.10f}")
    print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")
    print(f"Max:       {A.abs().max().item():.10f}")
    print(f"Ref max:   {A_ref.abs().max().item():.10f}")


# Input dimensions
B = 1
N = 1536
H = 1
D = 128

# Input tensors
print('Generating inputs...')
Q =      torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
K =      torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
V =      torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
L =      torch.zeros((B, H, 1, N), dtype=torch.float,    device='cuda')
O =      torch.zeros((B, H, N, D), dtype=torch.bfloat16, device='cuda')
D_vec =  torch.empty((B, H, 1, N), dtype=torch.float,    device='cuda')
Q_grad = torch.zeros_like(Q,       dtype=torch.float,    device='cuda')
K_grad = torch.zeros_like(K,       dtype=torch.float,    device='cuda')
V_grad = torch.zeros_like(V,       dtype=torch.float,    device='cuda')
O_grad = torch.ones_like(O,        dtype=torch.bfloat16, device='cuda')

# Run forward kernel
print("\nRunning forward kernel...")
fwd_attend_ker_128_noncausal(Q, K, V, L, O)
torch.cuda.synchronize()

# Run forward reference
Q.requires_grad = True
K.requires_grad = True
V.requires_grad = True
scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
attn = torch.softmax(scores, dim=-1)
O_ref = torch.matmul(attn, V)

# Check forward pass
check_diff("O", O, O_ref)

# Run backward reference
O_ref.backward(O_grad)
Q_grad_ref = Q.grad.detach()
K_grad_ref = K.grad.detach()
V_grad_ref = V.grad.detach()
D_vec_ref = (O * O_grad).sum(dim=-1).unsqueeze(2)

# Run backward Pytorch
V_grad_torch = torch.matmul(attn.transpose(-2, -1), O_grad) # dL/dV = attn^T @ dL/dO
attn_grad = torch.matmul(O_grad, V.transpose(-2, -1)) # dL/d(attn) = dL/dO @ V^T
scores_grad = (attn_grad * attn - attn * (attn_grad * attn).sum(dim=-1, keepdim=True)) / (D ** 0.5)
Q_grad_torch = torch.matmul(scores_grad, K) # dL/dQ = dL/d(scores) @ K
K_grad_torch = torch.matmul(scores_grad.transpose(-2, -1), Q) # dL/dK = dL/d(scores)^T @ Q
check_diff("Q_grad_torch", Q_grad_torch, Q_grad_ref)
check_diff("K_grad_torch", K_grad_torch, K_grad_ref)
check_diff("V_grad_torch", V_grad_torch, V_grad_ref)

# Run backward kernel
print("\nRunning backward kernel...")
bwd_attend_prep_ker_128(O_grad, O, D_vec)
bwd_attend_ker_128_noncausal(Q, K, V, O_grad, Q_grad, K_grad, V_grad, L, D_vec, Q.shape[-2], 1)
torch.cuda.synchronize()

# Check backward pass
check_diff("D_vec", D_vec, D_vec_ref)
check_diff("Q_grad", Q_grad, Q_grad_ref)
check_diff("K_grad", K_grad, K_grad_ref)
check_diff("V_grad", V_grad, V_grad_ref)

# Benchmark
NUM_WARMUPS = 5
NUM_ITERS = 10

start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]

for i in range(NUM_WARMUPS):
    fwd_attend_ker_128_noncausal(Q, K, V, L, O)

for i in range(NUM_ITERS):
    start_events[i].record()
    fwd_attend_ker_128_noncausal(Q, K, V, L, O)
    end_events[i].record()
torch.cuda.synchronize()

times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
avg_time = np.mean(times) * 1e-3
std_time = np.std(times) * 1e-3
flops = 4 * B * H * N * N * D
tflops = flops * 1e-12

print("\nForward")
print(f'Time taken: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us')
print(f'TFLOPS: {tflops / avg_time:.2f} TFLOP/s')

for i in range(NUM_WARMUPS):
    bwd_attend_prep_ker_128(O_grad, O, D_vec)
    bwd_attend_ker_128_noncausal(Q, K, V, O_grad, Q_grad, K_grad, V_grad, L, D_vec, Q.shape[-2], 1)
    torch.cuda.synchronize()

for i in range(NUM_ITERS):
    start_events[i].record()
    bwd_attend_prep_ker_128(O_grad, O, D_vec)
    bwd_attend_ker_128_noncausal(Q, K, V, O_grad, Q_grad, K_grad, V_grad, L, D_vec, Q.shape[-2], 1)
    end_events[i].record()
torch.cuda.synchronize()

times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
avg_time = np.mean(times) * 1e-3
std_time = np.std(times) * 1e-3
flops = 2.5 * 4 * B * H * N * N * D
tflops = flops * 1e-12

print("\nBackward")
print(f'Time taken: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us')
print(f'TFLOPS: {tflops / avg_time:.2f} TFLOP/s')
