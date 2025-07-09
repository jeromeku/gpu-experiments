#include "multi-gpu.cuh"

// Matrix multiplication kernel
__global__ void matmul_kernel(float* a, float* b, float* c, int n, int n_per_dev, int dev_num) {

    int row = n_per_dev * dev_num + blockIdx.y * blockDim.y + threadIdx.y; // we divide the workload per GPU by matrix rows
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col - n * n_per_dev * dev_num] = sum;
    }
}

int main() {

    // Constants
    constexpr int n_dev = 4;                                              // Number of GPUs
    constexpr int n = 2048;                                               // Matrix dimension
    constexpr int n_per_dev = (n + n_dev - 1) / n_dev;                    // # matrix rows per GPU
    constexpr int block_size = 32;                                        // 32 x 32 => 1024 threads per thread block
    constexpr int grid_xdim = (n + block_size - 1) / block_size;          // 1 element per thread
    constexpr int grid_ydim = (n_per_dev + block_size - 1) / block_size;  // 1 element per thread
    
    // Initialize NCCL communicators
    ncclComm_t comms[n_dev];
    int devs[n_dev];
    for (int i = 0; i < n_dev; ++i) devs[i] = i;
    NCCLCHECK(ncclCommInitAll(comms, n_dev, devs));

    // Allocate host matrices
    float *host_a = (float*)malloc(n * n * sizeof(float));
    float *host_b = (float*)malloc(n * n * sizeof(float));
    float *host_c = (float*)malloc(n * n * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < n * n; ++i) {
        host_a[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        host_b[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    memset(host_c, 0, n * n * sizeof(float));
    
    // Allocate device buffers and streams
    float **dev_a = (float**)malloc(n_dev * sizeof(float*));
    float **dev_b = (float**)malloc(n_dev * sizeof(float*));
    float **dev_c = (float**)malloc(n_dev * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(n_dev * sizeof(cudaStream_t));
    
    // Initialize devices
    for (int i = 0; i < n_dev; ++i) {
        CUDACHECK(cudaSetDevice(i));

        // Allocate device memory (A, B, C)
        CUDACHECK(cudaMalloc(&dev_a[i], n * n * sizeof(float)));
        CUDACHECK(cudaMalloc(&dev_b[i], n * n * sizeof(float)));
        CUDACHECK(cudaMalloc(&dev_c[i], n * n_per_dev * sizeof(float))); // divide workload by rows

        // Create CUDA stream
        CUDACHECK(cudaStreamCreate(&streams[i]));

        // Copy input data (A, B)
        CUDACHECK(cudaMemcpyAsync(dev_a[i], host_a, n * n * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
        CUDACHECK(cudaMemcpyAsync(dev_b[i], host_b, n * n * sizeof(float), cudaMemcpyHostToDevice, streams[i]));

        // Initialize accumulator (C)
        CUDACHECK(cudaMemsetAsync(dev_c[i], 0, n * n_per_dev * sizeof(float), streams[i]));
    }

    // Launch kernels
    dim3 block_dim(block_size, block_size);
    dim3 grid_dim(grid_xdim, grid_ydim);
    for (int i = 0; i < n_dev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        matmul_kernel<<<grid_dim, block_dim, 0, streams[i]>>>(
            dev_a[i], dev_b[i], dev_c[i], n, n_per_dev, i);
        CUDACHECK(cudaMemcpyAsync(host_c + i * n * n_per_dev, dev_c[i], n * n_per_dev * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
    }

    // Wait for all operations to complete 
    for (int i = 0; i < n_dev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    
#ifdef DEBUG
    // Verify result
    bool correct = true;
    constexpr float tol = 1e-3;
    float *expected_c = (float*)malloc(n * n * sizeof(float));
    for (int i = 0; i < n; i++) { // Naive matrix multiplication on CPU
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += host_a[i * n + k] * host_b[k * n + j];
            }
            expected_c [i * n + j] = sum;
            if (abs(host_c[i * n + j] - expected_c[i * n + j]) > tol) {
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }
    printf("Matrix multiplication %s\n", correct ? "PASSED" : "FAILED");
    free(expected_c);
#endif

    // Cleanup GPU
    for (int i = 0; i < n_dev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(dev_a[i]));
        CUDACHECK(cudaFree(dev_b[i]));
        CUDACHECK(cudaFree(dev_c[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }

    // Cleanup host
    free(host_a);
    free(host_b);
    free(host_c);
    free(dev_a);
    free(dev_b);
    free(dev_c);
    free(streams);

    return 0;
}
