import torch
torch.random.manual_seed(42)


# Function is naive for clarity, should not be like this in production
def quantize_2d(V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(V.shape) == 2
    assert V.dtype == torch.float32

    # Following not the OCP MX specs, but the NVIDIA recipe: https://arxiv.org/pdf/2506.08027
    # This specifically follows the Appendix (page 13)
    # To be more precise, we have to use Blackwell hardware (thus, in the kernel)
    block_amax = torch.amax(torch.abs(V).view(V.shape[0], V.shape[1] // 32, 32), dim=-1)
    dest_max = 448.0
    decode_scale = block_amax / dest_max
    V_sc = torch.clamp(torch.ceil(torch.log2(decode_scale)), min=-127)
    V_fp8 = (V / (2 ** V_sc.repeat_interleave(32, dim=-1))).to(torch.float8_e4m3fn)

    # Torch does not support float8_e8m0, so we need to manually convert to uint8
    fp8e8m0_bias = 127
    V_sc = (V_sc + fp8e8m0_bias).to(torch.uint8)

    return V_fp8, V_sc


# Function is naive for clarity, should not be like this in production
def dequantize_2d(V_fp8: torch.Tensor, V_sc: torch.Tensor) -> torch.Tensor:
    assert len(V_fp8.shape) == 2
    assert len(V_sc.shape) == 2
    assert V_fp8.dtype == torch.float8_e4m3fn
    assert V_sc.dtype == torch.uint8
    assert V_fp8.shape[0] == V_sc.shape[0]
    assert V_fp8.shape[1] == V_sc.shape[1] * 32

    # Torch does not support float8_e8m0, so we need to manually convert
    fp8e8m0_bias = 127
    scale = 2 ** (V_sc.to(torch.float32) - fp8e8m0_bias)

    return V_fp8.to(torch.float32) * scale.repeat_interleave(32, dim=-1)


# Check quantization loss
print("Checking quantization loss...")
V = torch.randn(1024, 1024, dtype=torch.float32)
V_fp8, V_sc = quantize_2d(V)
V_deq = dequantize_2d(V_fp8, V_sc)
abs_diff = torch.abs(V - V_deq)
print('Max adiff:', abs_diff.max())
print('Mean adiff (should be around 1e-2):', abs_diff.mean())
