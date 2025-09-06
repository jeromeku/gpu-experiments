import torch

from _C import create_custom_tensor

A = create_custom_tensor([1024, 1024], torch.bfloat16, 0)
torch.cuda.synchronize()
B = create_custom_tensor([1024, 1024], torch.bfloat16, 0)
torch.cuda.synchronize()

print(A)
torch.cuda.synchronize()
print(B)
torch.cuda.synchronize()

print(A.device)
print(B.device)
print(A.dtype)
print(B.dtype)
print(A.shape)
print(B.shape)

A.random_(0, 10)
B.random_(0, 10)

print(A)
print(B)

C = A + B
C += A @ B

print(C)
