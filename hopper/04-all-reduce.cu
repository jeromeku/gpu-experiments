#include "multi-gpu.cuh"

int main() {

    // Constants
    constexpr int n_dev = 8; // Number of GPUs
    constexpr int size = 256 * 1024 * 1024;
    constexpr int nelem = size / sizeof(float);

    // Check number of GPUs
    int deviceCount = 0;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < n_dev) {
        fprintf(stderr, "Error: Not enough GPUs available\n");
        return 1;
    }

    // Initialize NCCL communicators
    ncclComm_t comms[n_dev];
    int devs[n_dev];
    for (int i = 0; i < n_dev; ++i) devs[i] = i;
    NCCLCHECK(ncclCommInitAll(comms, n_dev, devs));

    // Allocate and initialize host matrices
    float **host_mats = (float**)malloc(n_dev * sizeof(float*));
    for (int i = 0; i < n_dev; ++i) {
        host_mats[i] = (float*)malloc(nelem * sizeof(float));
        for (int j = 0; j < nelem; ++j) {
            host_mats[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
    
    // Allocate device buffers and streams
    float **dev_mats = (float**)malloc(n_dev * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(n_dev * sizeof(cudaStream_t));
    
    // Initialize devices
    for (int i = 0; i < n_dev; ++i) {
        CUDACHECK(cudaSetDevice(i));

        // Allocate device memory
        CUDACHECK(cudaMalloc(&dev_mats[i], nelem * sizeof(float)));

        // Create CUDA stream
        CUDACHECK(cudaStreamCreate(&streams[i]));

        // Copy input data
        CUDACHECK(cudaMemcpyAsync(dev_mats[i], host_mats[i], nelem * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
    }

    // Wait for all async memcpy operations to complete 
    for (int i = 0; i < n_dev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

    // Start profiling
    // cudaProfilerStart();

    // All reduce (average operation)
    auto start = std::chrono::high_resolution_clock::now();
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < n_dev; ++i) {
        NCCLCHECK(ncclAllReduce((const void*)dev_mats[i], (void*)dev_mats[i], nelem, ncclFloat32, ncclSum, comms[i], streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Time for %d MiB allreduce: %f ms\n", size / 1024 / 1024, elapsed.count() * 1e3);

    // Wait for all operations to complete 
    for (int i = 0; i < n_dev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

    // cudaProfilerStop();

#ifdef DEBUG

    // Create expected matrix
    float *expected = (float*)malloc(nelem * sizeof(float));
    for (int i = 0; i < nelem; ++i) {
        expected[i] = 0;
        for (int j = 0; j < n_dev; ++j) {
            expected[i] += host_mats[j][i];
        }
        expected[i] /= n_dev;
    }

    // Copy result to host
    for (int i = 0; i < n_dev; ++i) {
        CUDACHECK(cudaMemcpy(host_mats[i], dev_mats[i], nelem * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Verify result
    bool correct = true;
    constexpr float tol = 1e-5;
    for (int i = 0; i < n_dev; ++i) {
        for (int j = 0; j < nelem; ++j) {
            if (abs(host_mats[i][j] - expected[j]) > tol) {
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }
    printf("All reduce %s\n", correct ? "PASSED" : "FAILED");
    free(expected);

#endif

    // Cleanup GPU
    for (int i = 0; i < n_dev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(dev_mats[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }

    // Cleanup host
    for (int i = 0; i < n_dev; ++i) {
        free(host_mats[i]);
    }
    free(host_mats);
    free(dev_mats);
    free(streams);

    return 0;
}