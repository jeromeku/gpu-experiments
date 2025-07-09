import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.parallel as parallel
import torch.cuda.nvtx as nvtx


# Model HP
NUM_HEADS = 64
SEQ_LEN = 256
HEAD_DIM = 128

# Training HP
BATCH_SIZE = 128
LEARNING_RATE = 1e-2
ITERATIONS = 10

# Our "kernel"s
class MyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device):
        """
        Custom Multihead Attention Layer.

        Args:
            embed_dim (int): Total embedding dimension (should be divisible by num_heads).
            num_heads (int): Number of attention heads.
        """
        super(MyMultiheadAttention, self).__init__()

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Each head's dimension

        # Learnable projection matrices
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False, device=device)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False, device=device)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False, device=device)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False, device=device)

    def scaled_dot_product_attention(self, Q, K, V):
        """
        Computes scaled dot-product attention.

        Args:
            Q, K, V: Tensors of shape (batch_size, num_heads, seq_len, head_dim)
        
        Returns:
            Output tensor after attention (same shape as V)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # Apply softmax over last dim
        return torch.matmul(attn_weights, V)

    def forward(self, Q, K, V):
        """
        Forward pass for Multihead Attention.

        Args:
            Q, K, V: Input tensors of shape (batch_size, seq_len, embed_dim)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = Q.shape

        # Linear projections
        Q_proj = self.W_q(Q)  # (batch_size, seq_len, embed_dim)
        K_proj = self.W_k(K)
        V_proj = self.W_v(V)

        # Split into heads: (batch_size, num_heads, seq_len, head_dim)
        Q_heads = Q_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_heads = K_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_heads = V_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply attention
        attn_output = self.scaled_dot_product_attention(Q_heads, K_heads, V_heads)

        # Concatenate heads back: (batch_size, seq_len, embed_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)

        # Final linear projection
        output = self.W_o(attn_output)
        return output


def train_mha(rank, world_size):
    print(f"Running MHA training on rank {rank}.")

    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # This is not strictly necessary, but let's do it anyway
    device = torch.device(f"cuda:{rank}")
    
    # Sync point 1 - construction
    nvtx.range_push(f"Rank {rank} - Construction")
    my_mha = MyMultiheadAttention(embed_dim=NUM_HEADS * HEAD_DIM, num_heads=NUM_HEADS, device=device)
    ddp_model = parallel.DistributedDataParallel(my_mha) # no need to pass in device, it's handled by the model
    nvtx.range_pop()
    print(f"{rank}: Total model size:", sum(p.numel() for p in my_mha.parameters()) * 4 / 1024**2, 'MB')

    optimizer = optim.SGD(ddp_model.parameters(), lr=LEARNING_RATE)

    # Input data
    print(f"{rank}: Initializing Q, K, V tensors...")
    nvtx.range_push(f"Rank {rank} - Input Data")
    Q = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS * HEAD_DIM, device=device)
    K = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS * HEAD_DIM, device=device)
    V = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS * HEAD_DIM, device=device)
    nvtx.range_pop()
    print(f"{rank}: Total input data size:", (Q.numel() + K.numel() + V.numel()) * 4 / 1024**2, 'MB')


    # "Training"
    print(f"{rank}: Starting iterative training...")
    for i in range(ITERATIONS):
        print(f"{rank}: Iteration {i + 1}")

        nvtx.range_push(f"Iter {i} - Zero Grad")
        optimizer.zero_grad()
        nvtx.range_pop()

        nvtx.range_push(f"Iter {i} - Forward Pass")
        output = ddp_model(Q, K, V)
        nvtx.range_pop()

        nvtx.range_push(f"Iter {i} - Compute Loss")
        loss = output.mean()
        nvtx.range_pop()

        nvtx.range_push(f"Iter {i} - Backward Pass")
        loss.backward()
        nvtx.range_pop()

        nvtx.range_push(f"Iter {i} - Optimizer Step")
        optimizer.step()
        nvtx.range_pop()

    # Clean up
    nvtx.range_push(f"Rank {rank} - Cleanup")
    dist.destroy_process_group()
    nvtx.range_pop()
    print(f"{rank}: Done.")

def run_distributed(function, world_size):
    mp.spawn(function,
             args=(world_size,), # note that the rank is injected as first argument
             nprocs=world_size, # number of processes
             join=True)


if __name__ == "__main__":

    assert torch.cuda.is_available(), "CUDA GPU not available, exiting..."
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but only got {n_gpus}"
    world_size = n_gpus

    run_distributed(train_mha, world_size)
