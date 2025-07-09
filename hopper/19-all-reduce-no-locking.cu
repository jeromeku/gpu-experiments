/*
    Code for running multi-GPU NCCL kernel WITHOUT thread blocking, to enable NCU profliing
    
    - You must manually run this code on each process
    - Run with myRank=0 first, this will print the generated ncclUniqueId in hex
      (using a fixed ncclUniqueId doesn't work, not sure why yet)
    - Run other ranks with the printed ncclUniqueId as the last argument
 */

#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include "cuda_runtime.h"
#include "nccl.h"

#define CUDACHECK(cmd) do {                                   \
    cudaError_t err = cmd;                                    \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

#define NCCLCHECK(cmd) do {                                   \
    ncclResult_t res = cmd;                                   \
    if (res != ncclSuccess) {                                 \
        fprintf(stderr, "Failed, NCCL error %s:%d '%s'\n",    \
            __FILE__, __LINE__, ncclGetErrorString(res));     \
            exit(EXIT_FAILURE);                               \
    }                                                         \
} while(0)

constexpr int nelem = 1024 * 1024; // we do a simple allReduce on 1024 x 1024 matrices
constexpr int ncclUniqueIdSize = 128; // a single ncclUniqueId is 128 bytes

void printHex(const char* message, const int length)
{
    // Used to print ncclUniqueId in hex
    for (int i = 0; i < length; i++)
    {
        if (i > 0) printf(":");
        printf("%02X", (unsigned char)message[i]);
    }
    printf("\n");
}

void decodeHex(char *dst, const char* message, const int length)
{
    // Used to decode ncclUniqueId printed by printHex()
    for (int i = 0; i < length; i++)
    {
        sscanf(message + 3 * i, "%2x", (unsigned int*)(dst + i));
    }
}

void printHash(const float *arr, size_t length) {
    // Used to verify AllReduce result
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
    /*
     * 1. Read arguments
     */

    if (argc != 4 && argc != 5) {
        printf("Usage: %s <device> <myRank> <nRanks> [<ncclUniqueId>]\n", argv[0]);
        printf("       (run myRank = 0 first to get the ncclUniqueId, and use it on all other ranks)\n");
        return 1;
    }
    
    int device = atoi(argv[1]);
    int myRank = atoi(argv[2]);
    int nRanks = atoi(argv[3]);
    printf("device: %d, myRank: %d, nRanks: %d\n", device, myRank, nRanks);
    
    if (myRank != 0 && argc != 5) {
        printf("Usage: %s <device> <myRank> <nRanks> <ncclUniqueId>\n", argv[0]);
        printf("       (run myRank = 0 first to get the ncclUniqueId, and use it on all other ranks)\n");
        return 1;
    }

    int deviceCount = 0;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < nRanks) {
        printf("Not enough GPUs available\n");
        return 1;
    }

    /*
     * 2. Allocate & initialize the matrix on host & device
     */

    // Host matrix
    float *hostMat = (float*)malloc(nelem * sizeof(float));
    srand(static_cast<unsigned int>(time(nullptr))); // random seed
    for (int i = 0; i < nelem; ++i) {
        hostMat[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // Print the first 5 elements to manually verify
    for (int i = 0; i < 5; ++i) {
        printf("%f ", hostMat[i]);
    }
    printf("\n");

    // Device matrix
    float *devMat;
    CUDACHECK(cudaSetDevice(device));
    CUDACHECK(cudaMalloc(&devMat, nelem * sizeof(float)));
    CUDACHECK(cudaMemcpy(devMat, hostMat, nelem * sizeof(float), cudaMemcpyHostToDevice));

    /*
     * 3. Initialize NCCL
     */

    ncclUniqueId id;
    ncclComm_t comm;
    cudaStream_t stream;

    // get NCCL unique ID at rank 0; other ranks will get it from stdin
    if (myRank == 0) {
        ncclGetUniqueId(&id); // ncclUniqueId is a struct with 128 bytes of char
        printHex(id.internal, ncclUniqueIdSize);
    } else {
        decodeHex(id.internal, argv[4], ncclUniqueIdSize);
        printHex(id.internal, ncclUniqueIdSize);
    }

    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
    CUDACHECK(cudaStreamCreate(&stream));

    // **Run NCCL AllReduce (sum) kernel**
    // Since this is not within a ncclGroup, this will run directly without queueing
    NCCLCHECK(ncclAllReduce((const void*)devMat, (void*)devMat, nelem, ncclFloat, ncclSum, comm, stream));

    // Wait for the kernel to complete
    CUDACHECK(cudaStreamSynchronize(stream));

    /*
     * 4. Verify & cleanup
     */

    // Retrieve values and print hash to check
    CUDACHECK(cudaSetDevice(device));
    CUDACHECK(cudaMemcpy(hostMat, devMat, nelem * sizeof(float), cudaMemcpyDeviceToHost));
    printHash(hostMat, nelem);

    // Print the first 5 elements to manually verify
    for (int i = 0; i < 5; ++i) {
        printf("%f ", hostMat[i]);
    }
    printf("\n");

    // Free
    CUDACHECK(cudaFree(devMat));
    ncclCommDestroy(comm);
    free(hostMat);

    printf("[Rank %d] Done \n", myRank);

    return 0;
}
