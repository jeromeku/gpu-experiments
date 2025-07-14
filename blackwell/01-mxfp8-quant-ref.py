# MXFP8 Quantization
# Following not the OCP MX specs, but the NVIDIA recipe: https://arxiv.org/pdf/2506.08027

import torch
torch.random.manual_seed(42)

N = 1024
Q_BLOCK = 32

V = torch.randn(N, dtype=torch.float32)

# Following the technical details in the Appendix (page 13)
# To be more precise, we have to work with the bit representations
block_amax = torch.amax(torch.abs(V).view(-1, Q_BLOCK), dim=-1)
dest_max = 448.0
decode_scale = block_amax / dest_max
scale = torch.clamp(torch.ceil(torch.log2(decode_scale)), min=-127)
V_quantized_fp32 = V / (2 ** scale.repeat_interleave(Q_BLOCK, dim=-1))
V_quantized_fp8 = V_quantized_fp32.to(torch.float8_e4m3fn)

# Dequantize and check
V_dequantized = V_quantized_fp8.to(torch.float32) * (2 ** scale.repeat_interleave(Q_BLOCK, dim=-1))
abs_diff = torch.abs(V - V_dequantized)
print(V)
print(V_dequantized)
print('Max adiff:', abs_diff.max())
print('Mean adiff:', abs_diff.mean())
