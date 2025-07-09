#include "multi-gpu.cuh"

using namespace std;

#define KB(x) ((x) * 1024)
#define MB(x) (KB(x) * 1024)
#define GB(x) (MB(x) * 1024)

#define DEV0 4
#define DEV1 5

static constexpr int SIZE = MB(50);
static constexpr int NUM_BLOCKS = 8; // keep it powers of 2 for convenience
static constexpr int NUM_WARPS = 16;   // keep it powers of 2 for convenience

// Should get 720 GB/s for B200s
void dma_run(unsigned char *host_memory) {
    cout << "Allocating device memory..." << endl;
    unsigned char *d0;
    CUDACHECK(cudaSetDevice(DEV0));
    CUDACHECK(cudaMalloc(&d0, SIZE));
    CUDACHECK(cudaMemcpy(d0, host_memory, SIZE, cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());
    
    unsigned char *d1;
    CUDACHECK(cudaSetDevice(DEV1));
    CUDACHECK(cudaMalloc(&d1, SIZE));
    CUDACHECK(cudaDeviceSynchronize());

    // Check
    cout << "Checking DMA times..." << endl;
    CUDACHECK(cudaSetDevice(DEV1));
    auto begin = chrono::high_resolution_clock::now();
    CUDACHECK(cudaMemcpyAsync(d1, d0, SIZE, cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
    auto end = chrono::high_resolution_clock::now();
    auto time_us = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    cout << "Time: " << time_us << " us" << endl;
    cout << "Bandwidth: " << (SIZE / (time_us / 1e6)) / (1024. * 1024. * 1024.) << " GB/s" << endl;

    cout << "Cleaning up..." << endl;
    CUDACHECK(cudaSetDevice(DEV0));
    CUDACHECK(cudaFree(d0));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaSetDevice(DEV1));
    CUDACHECK(cudaFree(d1));
    CUDACHECK(cudaDeviceSynchronize());
}

__global__ void kernel(unsigned char *dst, const unsigned char *src) {
    constexpr int size_per_block = SIZE / NUM_BLOCKS;
    constexpr int size_per_warp = size_per_block / NUM_WARPS;
    constexpr int num_iters = size_per_warp / 32;

    int start_idx = size_per_block * blockIdx.x + size_per_warp * (threadIdx.x / 32);
    int lane_id = threadIdx.x & 31;

    for (int i = 0; i < num_iters; ++i)
        dst[start_idx + i * 32 + lane_id] = src[start_idx + i * 32 + lane_id];
}

void ker_run(unsigned char *host_memory, bool check = true) {
    cout << "Allocating device memory..." << endl;
    unsigned char *d0;
    CUDACHECK(cudaSetDevice(DEV0));
    CUDACHECK(cudaMalloc(&d0, SIZE));
    CUDACHECK(cudaMemcpy(d0, host_memory, SIZE, cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());
    
    unsigned char *d1;
    CUDACHECK(cudaSetDevice(DEV1));
    CUDACHECK(cudaMalloc(&d1, SIZE));
    CUDACHECK(cudaDeviceSynchronize());

    // Check
    cout << "Checking kernel times..." << endl;
    CUDACHECK(cudaSetDevice(DEV0));
    auto begin = chrono::high_resolution_clock::now();
    kernel<<<NUM_BLOCKS, NUM_WARPS * 32>>>(d1, d0);
    CUDACHECK(cudaDeviceSynchronize());
    auto end = chrono::high_resolution_clock::now();
    auto time_us = chrono::duration_cast<chrono::microseconds>(end - begin).count();
    cout << "Time: " << time_us << " us" << endl;
    cout << "Bandwidth: " << (SIZE / (time_us / 1e6)) / (1024. * 1024. * 1024.) << " GB/s" << endl;

    // Verify
    if (check) {
        cout << "Verifying..." << endl;
        unsigned char *h = new unsigned char[SIZE];
        CUDACHECK(cudaMemcpy(h, d1, SIZE, cudaMemcpyDeviceToHost));
        for (unsigned int i = 0; i < SIZE; i++) {
            if (h[i] != host_memory[i]) {
                cout << "Error at index " << i << ": " << (int)h[i] << " != " << (int)host_memory[i] << endl;
                break;
            }
        }
        delete[] h;
    }

    cout << "Cleaning up..." << endl;
    CUDACHECK(cudaSetDevice(DEV0));
    CUDACHECK(cudaFree(d0));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaSetDevice(DEV1));
    CUDACHECK(cudaFree(d1));
    CUDACHECK(cudaDeviceSynchronize());
}

int main() {
    // Initialize
    cout << "Initializing..." << endl;
    unsigned char *host_memory = new unsigned char[SIZE];
    for (unsigned int i = 0; i < SIZE; i++) host_memory[i] = rand() % 256;
    CUDACHECK(cudaSetDevice(DEV0));
    CUDACHECK(cudaDeviceEnablePeerAccess(DEV1, 0));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaSetDevice(DEV1));
    CUDACHECK(cudaDeviceEnablePeerAccess(DEV0, 0));
    CUDACHECK(cudaDeviceSynchronize());

    // Run
    // dma_run(host_memory);
    // dma_run(host_memory);
    // dma_run(host_memory);
    // dma_run(host_memory);
    ker_run(host_memory);
    ker_run(host_memory);
    // ker_run(host_memory);
    // ker_run(host_memory);

    delete[] host_memory;
    return 0;
}

__global__ void copyKernel1(unsigned char *dst, const unsigned char *src, const unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

__global__ void copyKernel2(unsigned char *dst, const unsigned char *src, const unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        volatile int value;
        asm volatile (
            "{ ld.weak.global.u8 %0, [%1];"
              "st.weak.global.u8 [%2], %0; }"
            : "=r"(value)
            : "l"(src + idx), "l"(dst + idx)
            : "memory"
        );
    }
}

__global__ void copyKernel3(uint4 *dst, const uint4 *src, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx * sizeof(uint4) < size) {
        dst[idx] = src[idx];
    }
}

__global__ void copyKernel4(uint4 *dst, const uint4 *src, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx * sizeof(uint4) < size) {
        volatile uint4 value;
        asm volatile (
            "{ ld.weak.global.v4.u32 {%0, %1, %2, %3}, [%4];"
              "st.weak.global.v4.u32 [%5], {%0, %1, %2, %3}; }"
            : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
            : "l"(src + idx), "l"(dst + idx)
            : "memory"
        );
    }
}

__global__ void copyKernel5(uint4 *dst, const uint4 *src, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx * sizeof(uint4) < size) {
        volatile uint4 value;
        asm volatile (
            "{ ld.weak.global.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
              "st.weak.global.v4.u32 [%5], {%0, %1, %2, %3}; }"
            : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
            : "l"(src + idx), "l"(dst + idx)
            : "memory"
        );
    }
}
