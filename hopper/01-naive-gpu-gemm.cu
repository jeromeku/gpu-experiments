#include "multi-gpu.cuh"

#define BLOCK_SIZE 16  // Change based on your GPU's optimal block size

// CUDA Kernel for NxNxN Matrix Multiplication (GEMM)
__global__ void kernel(float* A, float* B, float* C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

// CUDA Host Code
void runGEMM(int N) {
  // Allocate memory on host (CPU)
  size_t size = N * N * sizeof(float);
  float *h_A = (float*)malloc(size);
  float *h_B = (float*)malloc(size);
  float *h_C = (float*)malloc(size);

  // Initialize matrices with random values
  for (int i = 0; i < N * N; i++) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Allocate memory on device (GPU)
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Define block and grid size
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Launch Kernel
  kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  // Copy result from device to host
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free CPU memory
  free(h_A);
  free(h_B);
  free(h_C);
}

int main() {
  int N = 512; // Adjust as needed
  runGEMM(N);
  std::cout << "Matrix multiplication completed!" << std::endl;
  return 0;
}
