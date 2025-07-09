/*
    Example code for 1-device-per-process multi-GPU communication WITHOUT MPI
    
    - You must manually run this code on each process
    - Compile with `nvcc -o test 18-mpi-less-multiprocess.cu -lnccl -lcudart`
    - Run with `./test <device> <myRank> <nRanks>` (ex. `./test 0 0 4`)
 */

#include "multi-gpu.cuh"

void printHex(const char* message, const int length)
{
    for (int i = 0; i < length; i++)
    {
        if (i > 0) printf(":");
        printf("%02X", (unsigned char)message[i]);
    }
    printf("\n");
}

void decodeHex(char *dst, const char* message, const int length)
{
    for (int i = 0; i < length; i++)
    {
        sscanf(message + 3 * i, "%2x", dst + i);
    }
}

void printHash(const float *arr, size_t length) {
    uint32_t hash = 2166136261u; // FNV-1a 32-bit hash basis
    for (size_t i = 0; i < length; i++) {
        uint32_t bits;
        memcpy(&bits, &arr[i], sizeof(uint32_t)); // Get bit representation of float
        hash ^= bits;
        hash *= 16777619; // FNV prime
    }
    printf("Hash: %u\n", hash);   
}

int main(int argc, char* argv[])
{
    if (argc != 4 && argc != 5) {
        printf("Usage: %s <device> <myRank> <nRanks> [<ncclUniqueId>]\n", argv[0]);
        printf("       (run myRank = 0 first to get the ncclUniqueId, and use it on all other ranks)\n");
        return -1;
    }
    
    int device = atoi(argv[1]);
    int myRank = atoi(argv[2]);
    int nRanks = atoi(argv[3]);
    printf("device: %d, myRank: %d, nRanks: %d\n", device, myRank, nRanks);
    
    if (myRank != 0 && argc != 5) {
        printf("Usage: %s <device> <myRank> <nRanks> <ncclUniqueId>\n", argv[0]);
        printf("       (run myRank = 0 first to get the ncclUniqueId, and use it on all other ranks)\n");
        return -1;
    }

    int size = 4 * 1024 * 1024; // number of float32 elems

    ncclUniqueId id;
    ncclComm_t comm;
    float *sendbuff, *recvbuff;
    cudaStream_t s;

    // get NCCL unique ID at rank 0; other ranks will get it from stdin
    if (myRank == 0 && argc == 4) {
        ncclGetUniqueId(&id); // ncclUniqueId is a struct with 128 bytes of char
        printHex(id.internal, 128);
    } else {
        decodeHex(id.internal, argv[4], 128);
        printHex(id.internal, 128);
    }

    CUDACHECK(cudaSetDevice(device));
    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));

    // initialize NCCL
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    // communicate using NCCL
    NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum, comm, s));

    // completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));

    // Retrieve values and print hash to check
    float *sendbuff_host = (float*)malloc(size * sizeof(float));
    float *recvbuff_host = (float*)malloc(size * sizeof(float));
    CUDACHECK(cudaMemcpy(sendbuff_host, sendbuff, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(recvbuff_host, recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost));
    printHash(sendbuff_host, size);
    printHash(recvbuff_host, size);

    // free buffers
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));
    ncclCommDestroy(comm);

    printf("[Rank %d] Done \n", myRank);

    return 0;
}
