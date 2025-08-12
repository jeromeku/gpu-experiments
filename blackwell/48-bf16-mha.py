import numpy as np
import torch
torch.manual_seed(42)

# Import our Python bindings
from _C import fwd_attend_ker_128_noncausal, bwd_attend_prep_ker_128, bwd_attend_ker_128_noncausal

# Input dimensions
B = 1
H = 1
N = 1536
D = 128

# Input tensors
print('Generating inputs...')
Q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
K = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
V = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
L = torch.zeros((B, H, 1, N), dtype=torch.float,    device='cuda')
O = torch.zeros((B, H, N, D), dtype=torch.bfloat16, device='cuda')
Q_grad = torch.zeros_like(Q, dtype=torch.float, device='cuda')
K_grad = torch.zeros_like(K, dtype=torch.float, device='cuda')
V_grad = torch.zeros_like(V, dtype=torch.float, device='cuda')
D_vec = torch.empty((B, H, 1, N), dtype=torch.float, device='cuda')

# Run forward kernel
print("Running forward kernel...")
fwd_attend_ker_128_noncausal(Q, K, V, L, O)
torch.cuda.synchronize()

# Run forward reference
Q.requires_grad = True
K.requires_grad = True
V.requires_grad = True
attn = torch.matmul(Q, K.transpose(-2, -1))
attn = attn / (D ** 0.5)
attn = torch.softmax(attn, dim=-1)
O_ref = torch.matmul(attn, V)

# Check forward pass
print("Forward pass")
print("Max diff:", (O - O_ref).abs().max())
print("Mean diff:", (O - O_ref).abs().mean())

# Run backward reference
# O_grad = torch.randn_like(O_ref)
O_grad = torch.ones_like(O_ref)
O_ref.backward(O_grad)
Q_grad_ref = Q.grad.detach()
K_grad_ref = K.grad.detach()
V_grad_ref = V.grad.detach()

# Run backward kernel
print("Running backward kernel...")
bwd_attend_prep_ker_128(O_grad, O, D_vec)
bwd_attend_ker_128_noncausal(Q, K, V, O_grad, Q_grad, K_grad, V_grad, L, D_vec, Q.shape[-2], 1)
torch.cuda.synchronize()

# Check backward pass
print("Backward pass")
print("Max diff:", (Q_grad - Q_grad_ref).abs().max())
print("Mean diff:", (Q_grad - Q_grad_ref).abs().mean())
print("Q grad mean:", Q_grad.abs().mean(), Q_grad_ref.abs().mean())
print("Q grad max:", Q_grad.abs().max(), Q_grad_ref.abs().max())
print("K grad mean:", K_grad.abs().mean(), K_grad_ref.abs().mean())
print("K grad max:", K_grad.abs().max(), K_grad_ref.abs().max())
print("V grad mean:", V_grad.abs().mean(), V_grad_ref.abs().mean())
print("V grad max:", V_grad.abs().max(), V_grad_ref.abs().max())

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

print("Forward pass")
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

print("Backward pass")
print(f'Time taken: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us')
print(f'TFLOPS: {tflops / avg_time:.2f} TFLOP/s')
