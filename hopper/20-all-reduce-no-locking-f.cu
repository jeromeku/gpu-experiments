/*
    Code for running multi-GPU NCCL kernel WITHOUT thread blocking, to enable NCU profliing
    
    - You must manually run this code on each process
    - Run with myRank=0 first, this will save the generated ncclUniqueId in hex
      (using a fixed ncclUniqueId doesn't work, not sure why yet)
    - Run other ranks, it will read the ncclUniqueId automatically
 */

#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
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

constexpr int nelem = 4096 * 4096; // we do a simple allReduce on 4096 x 4096 matrices
constexpr int ncclUniqueIdSize = 128; // a single ncclUniqueId is 128 bytes

void saveHex(const char* message, const int length)
{
    // Used to print ncclUniqueId in hex
    // Save ncclUniqueId to a temporary file
    FILE *file = fopen("/scratch/stuart/ncclUniqueId.txt", "w");
    if (file == NULL) {
        fprintf(stderr, "Failed to open file for writing\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < length; i++) {
        if (i > 0) fprintf(file, ":");
        fprintf(file, "%02X", (unsigned char)message[i]);
    }
    fclose(file);
}

void readHex(char *dst, const int length)
{
    // Read ncclUniqueId from temporary file
    FILE *file = fopen("/scratch/stuart/ncclUniqueId.txt", "r");
    if (file == NULL) {
        fprintf(stderr, "Failed to open file for reading\n");
        exit(EXIT_FAILURE);
    }
    char message[length * 3];
    if (fgets(message, sizeof(message), file) == NULL) {
        fprintf(stderr, "Failed to read ncclUniqueId from file\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fclose(file);
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

    if (argc != 4) {
        printf("Usage: %s <device> <myRank> <nRanks>\n", argv[0]);
        printf("       (run myRank = 0 first to save the ncclUniqueId, and then run all the other ranks)\n");
        return 1;
    }

    int device = atoi(argv[1]);
    int myRank = atoi(argv[2]);
    int nRanks = atoi(argv[3]);
    printf("device: %d, myRank: %d, nRanks: %d\n", device, myRank, nRanks);

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
        saveHex(id.internal, ncclUniqueIdSize);
    } else {
        readHex(id.internal, ncclUniqueIdSize);
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
