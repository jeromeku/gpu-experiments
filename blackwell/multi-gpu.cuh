#ifndef MULTI_GPU_CUH
#define MULTI_GPU_CUH

// C
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <assert.h>

// C++
#include <iostream>
#include <memory>
#include <chrono>

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

// CUDA driver API
#define CUCHECK(cmd) do {                                     \
    CUresult err = cmd;                                       \
    if (err != CUDA_SUCCESS) {                                \
        const char *errStr;                                   \
        cuGetErrorString(err, &errStr);                       \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, errStr);                      \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// CUDA runtime API
#define CUDACHECK(cmd) do {                                   \
    cudaError_t err = cmd;                                    \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

/*
    Example usage:

    benchmark("Sorting a large vector", [] {
        std::vector<int> v(1'000'000);
        std::iota(v.begin(), v.end(), 0);  // Fill with 0,1,2,...999999
        std::sort(v.begin(), v.end(), std::greater<>()); // Sort in descending order
    });
*/
template <typename Func>
void benchmark(const char* message, Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << message << ": " << elapsed.count() << " ms" << std::endl;
}

#endif // MULTI_GPU_CUH
