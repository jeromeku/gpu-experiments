#include "multi-gpu.cuh"

static constexpr int nDev = 7;

int main() {
    // Create a communicator of 4 devs
    ncclComm_t comms[nDev];
    int size =  1024;
    int devs[nDev];
    for (int i = 0; i < nDev; ++i)
        devs[i] = i;

    // alloc & init device buffers (pointers to gl memory which host cannot directly read/write)
    float** sendbuff = (float**)malloc(nDev * sizeof(float*));
    float** recvbuff = (float**)malloc(nDev * sizeof(float*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nDev);

    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc((void**)sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(s+i));
    }

    // initialize nccl
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

    // Start profiling
    // cudaProfilerStart();

    // call nccl API. Using Group is required when using multiple devices in a single thread
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i) {
        NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
            comms[i], s[i]));
    }
    NCCLCHECK(ncclGroupEnd());

    // Stop profiling
    // cudaProfilerStop();

    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }

    //free device buffers
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }

    // End of program. Destory comm obj
    for(int i = 0; i < nDev; ++i)
        ncclCommDestroy(comms[i]);

    printf("Success \n");
    return 0;
}
