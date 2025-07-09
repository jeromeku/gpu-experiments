#include "multi-gpu.cuh"

int main() {

    // Constants
    constexpr int n_dev = 2; // Number of GPUs
    constexpr float n_data = 1; // number of data points to transfer

#ifdef DEBUG
    // Check number of GPUs
    int deviceCount = 0;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < n_dev) {
        fprintf(stderr, "Error: Not enough GPUs available\n");
        return 1;
    }
#endif

    // Initialize NCCL communicators
    ncclComm_t comms[n_dev];
    int devs[n_dev];
    for (int i = 0; i < n_dev; ++i) devs[i] = i;
    NCCLCHECK(ncclCommInitAll(comms, n_dev, devs));

    // Allocate and initialize host array
    float *host_arr = (float*)malloc(n_data * sizeof(float));
    for (int i = 0; i < n_data; ++i) {
        host_arr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // Allocate device buffers and streams
    float **dev_arrs = (float**)malloc(n_dev * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(n_dev * sizeof(cudaStream_t));

    // Initialize device buffers
    for (int i = 0; i < n_dev; ++i) {
        CUDACHECK(cudaSetDevice(i));

        // Allocate device memory
        // asm volatile ("nop; nop; nop; nop; nop" ::: "memory");  // Unique marker
        CUDACHECK(cudaMalloc(&dev_arrs[i], n_data * sizeof(float)));
        // asm volatile ("nop; nop; nop; nop; nop" ::: "memory");  // Unique marker

        // Create CUDA stream
        CUDACHECK(cudaStreamCreate(&streams[i]));
    }

    // Copy input data to device 0
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMemcpy(dev_arrs[0], host_arr, n_data * sizeof(float), cudaMemcpyHostToDevice));

    // We could set to zero, but we will not
    // CUDACHECK(cudaMemset(...));

    // Start profiling
    // cudaProfilerStart();

    // P2P communication
    NCCLCHECK(ncclGroupStart()); // needed for multiple send/recv pairs
    NCCLCHECK(ncclSend(dev_arrs[0], n_data, ncclFloat32, 1, comms[0], streams[0]));
    NCCLCHECK(ncclRecv(dev_arrs[1], n_data, ncclFloat32, 0, comms[1], streams[1]));
    NCCLCHECK(ncclGroupEnd());

    // Wait for all operations to complete 
    for (int i = 0; i < n_dev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

#ifdef DEBUG
    float *copied_arr = (float*)malloc(n_data * sizeof(float));
    CUDACHECK(cudaMemcpy(copied_arr, dev_arrs[1], n_data * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_data; ++i) {
        printf("%f %f\n", copied_arr[i], host_arr[i]);
        assert(copied_arr[i] == host_arr[i]);
    }
    free(copied_arr);
#endif

    // We really should cleanup, but we will not
    // for (int i = 0; i < n_dev; ++i) {
    //     CUDACHECK(cudaSetDevice(i));
    //     CUDACHECK(cudaFree(dev_arrs[i]));
    //     CUDACHECK(cudaStreamDestroy(streams[i]));
    //     ncclCommDestroy(comms[i]);
    // }
    // free(host_arr);
    // free(dev_mats);
    // free(streams);

    return 0;
}
