#include "gpu-experiments.cuh"

/*

Generic notes on tcgen05 instructions

- Tensor memory (TM) is on-chip memory (likely SRAM)
- It is organized as 2D matrix (rows are called "lanes" and columns are called as is)
    - On sm_100a, this is 128 x 512 per CTA, each cell 32-bit in size
- TM address is 32-bit, where first 16 significant bits are lane index and next 16 are column index
- TM must be allocated by a single warp in a CTA
    - Allocation is done in columns only (all lanes in the columns are allocated)
    - Granularity is (1) powers of 2 and (2) at least 32
- Supported matrix multiply and accumulate shapes: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-shape
    - For 1-CTA dense (MX)FP8 matrix multiply, K is always 32
    - M and N are specified in the instruction descriptor
- Data movement shapes are in format lane x bits: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-movement-shape
    - Each movement type (16x32b, 32x32b, ...) has its unique way of how values across registers are spread throughout TM
- A warp in a warpgroup can access only 1/4 of the lanes, and all columns of the TM
    - Warp 0: lanes 0-31
    - Warp 1: lanes 32-63
    - Warp 2: lanes 64-95
    - Warp 3: lanes 96-127

*/

// Global dimension
constexpr int N = 128;
constexpr int M = 128;
constexpr int K = 32;

// Tile dimension
constexpr int TILE_N = 128;
constexpr int TILE_M = 128;
constexpr int TILE_K = 32;

// Quantization
constexpr int Q_BLOCK = 32;
constexpr int NUM_BLOCKS = K / Q_BLOCK;
constexpr float DEST_MAX = 448.0;

// Kernel
constexpr int SM_COUNT = 148;
constexpr int WARP_THREADS = 32;
constexpr int WARPGROUP_WARPS = 4;
constexpr int WARPGROUP_THREADS = WARP_THREADS * WARPGROUP_WARPS;
constexpr int NUM_WARPGROUPS = 2;
constexpr int NUM_THREADS = WARPGROUP_THREADS * NUM_WARPGROUPS;
constexpr int MAX_SHARED_MEMORY = 227000; // Hopper/Blackwell
constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1000;


__global__ void kernel(
    const __grid_constant__ __nv_fp8_e4m3 * const A_fp8,
    const __grid_constant__ __nv_fp8_e8m0 * const A_sc,
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ __nv_fp8_e4m3 * const B_fp8,
    const __grid_constant__ __nv_fp8_e8m0 * const B_sc,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ float * const C
) { 
    // Retrieve thread info
    int lane_id = threadIdx.x % WARP_THREADS;
    int warp_id = threadIdx.x / WARP_THREADS;
    int warpgroup_id = threadIdx.x / WARPGROUP_THREADS;

    // Allocate shared memory
    extern __shared__ int __shm[];
    __shared__ uint64_t mbarrier;
    __shared__ uint32_t tm_addr_shared;

    // Assign shared tiles. TMA swizzle require 1024 alignment max
    uint64_t __shm_ptr = reinterpret_cast<uint64_t>(&__shm[0]);
    __nv_fp8_e4m3 *A_fp8_shm = reinterpret_cast<__nv_fp8_e4m3 *>(((__shm_ptr + 1023) / 1024) * 1024);
    __shm_ptr += sizeof(__nv_fp8_e4m3) * TILE_M * TILE_K;
    __nv_fp8_e8m0 *A_sc_shm = reinterpret_cast<__nv_fp8_e8m0 *>(((__shm_ptr + 1023) / 1024) * 1024);
    __shm_ptr += sizeof(__nv_fp8_e8m0) * TILE_M * NUM_BLOCKS;
    __nv_fp8_e4m3 *B_fp8_shm = reinterpret_cast<__nv_fp8_e4m3 *>(((__shm_ptr + 1023) / 1024) * 1024);
    __shm_ptr += sizeof(__nv_fp8_e4m3) * TILE_N * TILE_K;
    __nv_fp8_e8m0 *B_sc_shm = reinterpret_cast<__nv_fp8_e8m0 *>(((__shm_ptr + 1023) / 1024) * 1024);
    __shm_ptr += sizeof(__nv_fp8_e8m0) * TILE_N * NUM_BLOCKS;
    float *C_shm = reinterpret_cast<float *>(((__shm_ptr + 1023) / 1024) * 1024);
    __shm_ptr += sizeof(float) * TILE_M * TILE_N;
    if (__shm_ptr >= DYNAMIC_SHARED_MEMORY) {
        if (threadIdx.x == 0) printf("ERROR: Exceeded maximum dynamic shared memory.");
        asm volatile("trap;");
    }

    // Initialize mbarriers
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
            :
            : "l"(__cvta_generic_to_shared(&mbarrier)), "r"(1)
        );
    }
    __syncthreads();

    // Allocate Tensor Memory (TM) for 1-CTA group 
    uint32_t tm_addr = 0;
    uint32_t n_cols = 32; // must be unsigned 32b
    if (warp_id == 0) { // must be performed by a single warp in the CTA
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;"
            :
            : "l"((uint64_t)&tm_addr_shared), "r"(n_cols)
        ); // __syncwarp() naturally happens here
        // After relinquish_alloc_permit, it becomes illegal for this CTA to call tcgen05.alloc
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    __syncthreads();
    tm_addr = tm_addr_shared; // Move from shared memory into register

    // Main work begins here
    if (warpgroup_id == 1) {
        // Producer warpgroup
    } else {
        // Consumer warpgroup
    }

    // De-allocate TM for 1-CTA group
    if (warp_id == 0) { // must be performed by a single warp in the CTA
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :
            : "r"(tm_addr), "r"(n_cols)
        );
    }
}


int main() {
    static_assert(K % Q_BLOCK == 0, "K must be divisible by Q_BLOCK");
    std::cout << "M = " << M << ", N = " << N << ", K = " << K << ", Q_BLOCK = " << Q_BLOCK << std::endl;

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[N * K];
    __nv_fp8_e4m3 *h_A_fp8 = new __nv_fp8_e4m3[M * K];
    __nv_fp8_e4m3 *h_B_fp8 = new __nv_fp8_e4m3[N * K];
    __nv_fp8_e8m0 *h_A_sc = new __nv_fp8_e8m0[M * NUM_BLOCKS];
    __nv_fp8_e8m0 *h_B_sc = new __nv_fp8_e8m0[N * NUM_BLOCKS];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];
    std::cout << "Allocated host memory" << std::endl;

    // Initialize matrices with random values
    std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < N * K; ++i) h_B[i] = dis(gen);
    std::cout << "Initialized matrices" << std::endl;

    // Matrix A quantization
    for (int i = 0; i < M; i++) {
        for (int block = 0; block < NUM_BLOCKS; block++) {
            // Get block absolute maximum
            float amax = fabsf(h_A[i * K + block * Q_BLOCK]);
            for (int j = 1; j < Q_BLOCK; j++)
                amax = fmaxf(amax, h_A[i * K + block * Q_BLOCK + j]);

            // ceilf(log2f(amax / DEST_MAX)) with round to +inf & clamp to [2^-127, 2^127]
            h_A_sc[i * NUM_BLOCKS + block] = __nv_fp8_e8m0(amax / DEST_MAX); 
            // printf("actual: %f, stored: %d, val: %f\n", amax / DEST_MAX, 
            //     *reinterpret_cast<uint8_t *>(&h_A_sc[i * NUM_BLOCKS + block]) - 127,
            //     powf(2., *reinterpret_cast<uint8_t *>(&h_A_sc[i * NUM_BLOCKS + block]) - 127));

            // Quantize
            for (int j = 0; j < Q_BLOCK; j++) {
                float quantized_fp32 = h_A[i * K + block * Q_BLOCK + j] / 
                    powf(2., *reinterpret_cast<uint8_t *>(&h_A_sc[i * NUM_BLOCKS + block]) - 127);
                h_A_fp8[i * K + block * Q_BLOCK + j] = __nv_fp8_e4m3(quantized_fp32);
            }
        }
    }

    // Matrix B quantization
    for (int i = 0; i < N; i++) {
        for (int block = 0; block < NUM_BLOCKS; block++) {
            // Get block absolute maximum
            float amax = fabsf(h_B[i * K + block * Q_BLOCK]);
            for (int j = 1; j < Q_BLOCK; j++)
                amax = fmaxf(amax, h_B[i * K + block * Q_BLOCK + j]);

            // this does ceilf(log2f(amax / DEST_MAX)) with round to +inf & clamp to [2^-127, 2^127]
            h_B_sc[i * NUM_BLOCKS + block] = __nv_fp8_e8m0(amax / DEST_MAX); 
            // printf("actual: %f, stored: %d, val: %f\n", amax / DEST_MAX, 
            //     *reinterpret_cast<uint8_t *>(&h_B_sc[i * NUM_BLOCKS + block]) - 127,
            //     powf(2., *reinterpret_cast<uint8_t *>(&h_B_sc[i * NUM_BLOCKS + block]) - 127));

            // Quantize
            for (int j = 0; j < Q_BLOCK; j++) {
                float quantized_fp32 = h_B[i * K + block * Q_BLOCK + j] / 
                    powf(2., *reinterpret_cast<uint8_t *>(&h_B_sc[i * NUM_BLOCKS + block]) - 127);
                h_B_fp8[i * K + block * Q_BLOCK + j] = __nv_fp8_e4m3(quantized_fp32);
            }
        }
    }

    // Sanity check: dequantize and check errors
    // for (int i = 0; i < M; i++) {
    //     for (int block = 0; block < NUM_BLOCKS; block++) {
    //         for (int j = 0; j < Q_BLOCK; j++) {
    //             float dequantized_fp32 = float(h_A_fp8[i * K + block * Q_BLOCK + j]) * 
    //                 powf(2., *reinterpret_cast<uint8_t *>(&h_A_sc[i * NUM_BLOCKS + block]) - 127);
    //             float error = fabsf(h_A[i * K + block * Q_BLOCK + j] - dequantized_fp32);
    //             printf("A: %f, dequantized: %f, error: %f\n", h_A[i * K + block * Q_BLOCK + j], dequantized_fp32, error);
    //         }
    //     }
    // }
    // for (int i = 0; i < N; i++) {
    //     for (int block = 0; block < NUM_BLOCKS; block++) {
    //         for (int j = 0; j < Q_BLOCK; j++) {
    //             float dequantized_fp32 = float(h_B_fp8[i * K + block * Q_BLOCK + j]) * 
    //                 powf(2., *reinterpret_cast<uint8_t *>(&h_B_sc[i * NUM_BLOCKS + block]) - 127);
    //             float error = fabsf(h_B[i * K + block * Q_BLOCK + j] - dequantized_fp32);
    //             printf("B: %f, dequantized: %f, error: %f\n", h_B[i * K + block * Q_BLOCK + j], dequantized_fp32, error);
    //         }
    //     }
    // }

    // Run reference GEMM
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[i * K + k] * h_B[j * N + k];
            }
            h_C_ref[i * N + j] = sum;
        }
    }
    std::cout << "Performed CPU matrix multiplication" << std::endl;

    // Allocate device memory
    __nv_fp8_e4m3 *d_A_fp8;
    __nv_fp8_e4m3 *d_B_fp8;
    __nv_fp8_e8m0 *d_A_sc;
    __nv_fp8_e8m0 *d_B_sc;
    float *d_C;
    CUDACHECK(cudaMalloc(&d_A_fp8, M * K * sizeof(__nv_fp8_e4m3)));
    CUDACHECK(cudaMalloc(&d_B_fp8, K * N * sizeof(__nv_fp8_e4m3)));
    CUDACHECK(cudaMalloc(&d_A_sc, M * K / Q_BLOCK * sizeof(__nv_fp8_e8m0)));
    CUDACHECK(cudaMalloc(&d_B_sc, K / Q_BLOCK * N * sizeof(__nv_fp8_e8m0)));
    CUDACHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    std::cout << "Allocated device memory" << std::endl;

    // Copy data to device
    CUDACHECK(cudaMemcpy(d_A_fp8, h_A_fp8, M * K * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B_fp8, h_B_fp8, N * K * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_A_sc, h_A_sc, M * NUM_BLOCKS * sizeof(__nv_fp8_e8m0), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B_sc, h_B_sc, N * NUM_BLOCKS * sizeof(__nv_fp8_e8m0), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(d_C, 999999999.0f, M * N * sizeof(float))); // useful for checking errors
    std::cout << "Copied data to device" << std::endl;

    // Create tensor map descriptor for matrix A
    constexpr int tma_dim = 5; // always use all 5 dimensions
    constexpr int swizzle_bytes = 32; // should change accordingly on TILE_K
    constexpr int swizzle_elements = swizzle_bytes / sizeof(__nv_fp8_e4m3);
    constexpr CUtensorMapSwizzle tma_swizzle = 
        swizzle_bytes == 32  ? CU_TENSOR_MAP_SWIZZLE_32B  :
        swizzle_bytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B  :
        swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : 
                               CU_TENSOR_MAP_SWIZZLE_NONE;
    static_assert(K % swizzle_elements == 0);
    CUtensorMap A_tmap;
    uint64_t A_gmem_shape [5] = {
        (uint64_t)swizzle_elements,
        (uint64_t)M, 
        (uint64_t)K / swizzle_elements, 
        1, 
        1
    };
    uint64_t A_gmem_stride[4] = {
        (uint64_t)K * sizeof(__nv_fp8_e4m3), 
        (uint64_t)swizzle_bytes, 
        (uint64_t)M * K * sizeof(__nv_fp8_e4m3), // never utilized
        (uint64_t)M * K * sizeof(__nv_fp8_e4m3)  // never utilized
    };
    uint32_t A_smem_shape [5] = {
        swizzle_elements, 
        TILE_M, 
        TILE_K / swizzle_elements, 
        1, 
        1
    };
    uint32_t A_smem_stride[5] = {1, 1, 1, 1, 1};
    CUCHECK(cuTensorMapEncodeTiled(
        &A_tmap,
        CU_TENSOR_MAP_DATA_TYPE_UINT8, // there is no FP8 TMA type
        tma_dim,
        (void *)d_A_fp8,
        (uint64_t *)A_gmem_shape,
        (uint64_t *)A_gmem_stride, 
        (uint32_t *)A_smem_shape,
        (uint32_t *)A_smem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, // don't need this
        tma_swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, // don't need this
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE // don't need this
    ));

    // Create tensor map descriptor for matrix B
    CUtensorMap B_tmap;
    uint64_t B_gmem_shape [5] = {
        (uint64_t)swizzle_elements,
        (uint64_t)N, 
        (uint64_t)K / swizzle_elements, 
        1, 
        1
    };
    uint64_t B_gmem_stride[4] = {
        (uint64_t)K * sizeof(__nv_fp8_e4m3), 
        (uint64_t)swizzle_bytes, 
        (uint64_t)N * K * sizeof(__nv_fp8_e4m3), // never utilized
        (uint64_t)N * K * sizeof(__nv_fp8_e4m3)  // never utilized
    };
    uint32_t B_smem_shape [5] = {
        swizzle_elements, 
        TILE_N, 
        TILE_K / swizzle_elements, 
        1, 
        1
    };
    uint32_t B_smem_stride[5] = {1, 1, 1, 1, 1};
    CUCHECK(cuTensorMapEncodeTiled(
        &B_tmap,
        CU_TENSOR_MAP_DATA_TYPE_UINT8, // there is no FP8 TMA type
        tma_dim,
        (void *)d_B_fp8,
        (uint64_t *)B_gmem_shape,
        (uint64_t *)B_gmem_stride, 
        (uint32_t *)B_smem_shape,
        (uint32_t *)B_smem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, // don't need this
        tma_swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, // don't need this
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE // don't need this
    ));

    // Launch kernel
    std::cout << "Launching kernel..." << std::endl;
    dim3 grid(SM_COUNT, 1, 1);
    dim3 block(NUM_THREADS, 1, 1);
    CUDACHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SHARED_MEMORY
    ));

    // Warmup
    for (int i = 0; i < 5; i++) {
        kernel<<<grid, block, DYNAMIC_SHARED_MEMORY>>>(
            d_A_fp8, d_A_sc, A_tmap, d_B_fp8, d_B_sc, B_tmap, d_C);
        CUDACHECK(cudaDeviceSynchronize());
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 20; i++)
        kernel<<<grid, block, DYNAMIC_SHARED_MEMORY>>>(
            d_A_fp8, d_A_sc, A_tmap, d_B_fp8, d_B_sc, B_tmap, d_C);
    cudaDeviceSynchronize(); // no CUDACHECK here
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> diff = end - start;
    double duration_us = diff.count() / 20.0;
    std::cout << "Kernel execution time: " << duration_us << " us" << std::endl;

    // Calculate TFLOPs
    double flops = double(2.0) * M * N * K;
    double tflops = (flops / duration_us) / 1e6;
    std::cout << "Achieved performance: " << tflops << " TFLOPs" << std::endl;

    // Copy result back to host
    CUDACHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Copied results back to host" << std::endl;

    // Check results
    float max_error = 0.0f;
    float average_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        max_error = std::max(max_error, error);
        average_error += error;
    }
    average_error /= M * N;
    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Average error: " << average_error << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_A_fp8;
    delete[] h_B_fp8;
    delete[] h_A_sc;
    delete[] h_B_sc;
    delete[] h_C;
    delete[] h_C_ref;
    cudaFree(d_A_fp8);
    cudaFree(d_B_fp8);
    cudaFree(d_A_sc);
    cudaFree(d_B_sc);
    cudaFree(d_C);

    return 0;
}
