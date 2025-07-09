#include "multi-gpu.cuh"

using namespace std;

__global__ void copyKernel(int* src, int* dst) {
    // Copy value from src to dst
    // *dst = *src;
    volatile int value;
    asm volatile (
        "{ ld.global.u32 %0, [%1];"
        "st.global.u32 [%2], %0; }"
        : "=r"(value)
        : "l"(src), "l"(dst)
        : "memory"
    );
}

int main() {

    // P2P Setup
    int can_access_peer_0_1;
    int can_access_peer_1_0;
    CUDACHECK(cudaDeviceCanAccessPeer(&can_access_peer_0_1, 0, 1));
    CUDACHECK(cudaDeviceCanAccessPeer(&can_access_peer_1_0, 1, 0));
    cout << "Device 0 can access device 1: " << can_access_peer_0_1 << endl;
    cout << "Device 1 can access device 0: " << can_access_peer_1_0 << endl;
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaDeviceEnablePeerAccess(1, 0));
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaDeviceEnablePeerAccess(0, 0));

    // Step 1: Allocate device memory (two separate locations, each 4 bytes)
    int *d_src, *d_dst;
    
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMalloc((void**)&d_src, sizeof(int)));

    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMalloc((void**)&d_dst, sizeof(int)));

    // Step 2: Copy an integer from host to device (d_src)
    int h_value = 0xdeadbeef;
    CUDACHECK(cudaSetDevice(0)); // since 0 has the source, this becomes p2p write
    CUDACHECK(cudaMemcpy(d_src, &h_value, sizeof(int), cudaMemcpyHostToDevice));
    
    // Step 3: Launch the kernel to copy value from d_src to d_dst
    CUDACHECK(cudaSetDevice(0)); // since 0 has the source, this becomes p2p write
    copyKernel<<<1, 1>>>(d_src, d_dst);
    CUDACHECK(cudaDeviceSynchronize());  // Ensure kernel execution is completed

    // Step 4: Copy d_dst back to the host
    int h_result;
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMemcpy(&h_result, d_dst, sizeof(int), cudaMemcpyDeviceToHost));

    // Step 5: Verify correctness
    if (h_value == h_result) {
        std::cout << "Success! Value correctly copied: " << h_result << std::endl;
    } else {
        std::cerr << "Error! Mismatch: expected " << h_value << ", but got " << h_result << std::endl;
    }

    // Cleanup
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaFree(d_src));
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaFree(d_dst));

    return 0;
}
