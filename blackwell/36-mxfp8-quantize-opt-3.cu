/*
    Observation 1:
        DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) 
        DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1)
        DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) 
        DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1)

        It was the reason why the kernels suddenly became slow! Like REALLY slow, like 40% slower!
        It changes the way CUDA schedules blocks in a way that reduces the occupancy. Probably because
        it schedules at the granularity of clusters instead of individual blocks. I think there is also
        extra launch setup/teardown overhead for cluster mode.

    Observation 2:
        cp.async.bulk is surprisingly slow compared to cp.async.bulk.tensor, even if we are storing 1D tensor.
        Thus, we should just always use cp.async.bulk.tensor, and never cp.async.bulk
*/

#include <kittens.cuh>
#include <pybind11/pybind11.h>

using namespace kittens;
namespace py = pybind11;

static constexpr int M = 204800;
static constexpr int N = 2048;
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 128;
static constexpr int QUANT_SIZE = 32;

__global__ __launch_bounds__(TILE_M)
void kernel(
    const bf16 *A_bf16,
    const __grid_constant__ CUtensorMap A_bf16_tmap,
    fp8e4m3 *A_fp8,
    const __grid_constant__ CUtensorMap A_fp8_tmap,
    fp8e8m0 *A_sc,
    const __grid_constant__ CUtensorMap A_sc_tmap
) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    uint64_t __shm_base = reinterpret_cast<uint64_t>(&__shm[0]);
    bf16 *A_bf16_smem = reinterpret_cast<bf16*>(((__shm_base + 1023) / 1024) * 1024);
    fp8e4m3 *A_fp8_smem = reinterpret_cast<fp8e4m3*>(A_bf16_smem);
    fp8e8m0 *A_sc_smem = reinterpret_cast<fp8e8m0*>(A_fp8_smem + TILE_M * TILE_N); // naturally fulfills alignment

    // Initialize mbarriers
    __shared__ semaphore inputs_arrived;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
    }
    __syncthreads();

    // Calculate tile index
    int row = blockIdx.y * TILE_M;
    int col = blockIdx.x * TILE_N;

    // Initiate the load
    if (threadIdx.x == 0) {
        tma::expect_bytes(inputs_arrived, TILE_M * TILE_N * sizeof(bf16));
        asm volatile("{cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3}], [%4];}"
            :: "l"(__cvta_generic_to_shared(A_bf16_smem)), "l"(&A_bf16_tmap), "r"(col), "r"(row), "l"(__cvta_generic_to_shared(&inputs_arrived))
            : "memory");
    }

    // Wait for the load to complete
    asm volatile("{fence.proxy.async.shared::cta;}" ::: "memory"); // make writes to smem visible
    wait(inputs_arrived, 0);

    // Perform quantization
    // TODO
    if (threadIdx.x == 0) {
        // for (int i = 0 ; i < 512; i++)
        //     A_sc_smem[i] = fp8e8m0(1);
        // for (int i = 0 ; i < 128 * 128 ; i++)
        //     A_fp8_smem[i] = fp8e4m3(1);
    }

    // Store
    asm volatile("{fence.proxy.async.shared::cta;}" ::: "memory"); // make writes to smem visible
    if (threadIdx.x == 0) {
        // Since this is the only store, don't wait for completion
        asm volatile("{cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%1, %2}], [%3];}"
            :: "l"(&A_fp8_tmap), "r"(col), "r"(row), "l"(__cvta_generic_to_shared(A_fp8_smem))
            : "memory");
        asm volatile("{cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3}], [%4];}"
            :: "l"(&A_sc_tmap), "r"(0), "r"(col / TILE_N), "r"(row / TILE_M), "l"(__cvta_generic_to_shared(A_sc_smem))
            : "memory");
        asm volatile("{cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3}], [%4];}"
            :: "l"(&A_sc_tmap), "r"(TILE_N * TILE_M / QUANT_SIZE / 2), "r"(col / TILE_N), "r"(row / TILE_M), "l"(__cvta_generic_to_shared(A_sc_smem + TILE_M * TILE_N / QUANT_SIZE / 2))
            : "memory");
    }
}

template <typename T>
__host__ static inline T *get_data_ptr(py::object tensor) {
    // Assumes the following about `tensor`
    // - is a torch.Tensor object
    // - is contiguous
    // - is on device
    // - has the correct shape
    return reinterpret_cast<T *>(tensor.attr("data_ptr")().cast<uintptr_t>());
}

__host__ static inline void launch_kernel(py::object &A_bf16, py::object &A_fp8, py::object &A_sc) {
    CUtensorMap A_bf16_tmap, A_fp8_tmap, A_sc_tmap;

    static constexpr int A_bf16_rank = 2;
    static constexpr int A_fp8_rank = 2;
    static constexpr int A_sc_rank = 3;

    uint64_t A_bf16_shape[A_bf16_rank] = {N, M}; // inner-dim first
    uint64_t A_bf16_stride[A_bf16_rank - 1] = {N * sizeof(bf16)};
    uint32_t A_bf16_smem_shape[A_bf16_rank] = {TILE_N, TILE_M};
    uint32_t A_bf16_smem_stride[A_bf16_rank] = {1, 1};
    
    uint64_t A_fp8_shape[A_fp8_rank] = {N, M};
    uint64_t A_fp8_stride[A_fp8_rank - 1] = {N * sizeof(fp8e4m3)};
    uint32_t A_fp8_smem_shape[A_fp8_rank] = {TILE_N, TILE_M};
    uint32_t A_fp8_smem_stride[A_fp8_rank] = {1, 1};

    uint64_t A_sc_shape[A_sc_rank] = {TILE_N * TILE_M / QUANT_SIZE, N / TILE_N, M / TILE_M};
    uint64_t A_sc_stride[A_sc_rank - 1] = {TILE_N * TILE_M / QUANT_SIZE * sizeof(fp8e8m0), N * TILE_M / QUANT_SIZE * sizeof(fp8e4m3)};
    uint32_t A_sc_smem_shape[A_sc_rank] = {TILE_N * TILE_M / QUANT_SIZE / 2, 1, 1}; // divide into 2 TMA stores
    uint32_t A_sc_smem_stride[A_sc_rank] = {1, 1, 1};

    CUCHECK(cuTensorMapEncodeTiled(
        &A_bf16_tmap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        A_bf16_rank,
        (void *)get_data_ptr<bf16>(A_bf16),
        &A_bf16_shape[0],
        &A_bf16_stride[0],
        &A_bf16_smem_shape[0],
        &A_bf16_smem_stride[0],
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    CUCHECK(cuTensorMapEncodeTiled(
        &A_fp8_tmap,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        A_fp8_rank,
        (void *)get_data_ptr<fp8e4m3>(A_fp8),
        &A_fp8_shape[0],
        &A_fp8_stride[0],
        &A_fp8_smem_shape[0],
        &A_fp8_smem_stride[0],
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    CUCHECK(cuTensorMapEncodeTiled(
        &A_sc_tmap,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        A_sc_rank,
        (void *)get_data_ptr<fp8e8m0>(A_sc),
        &A_sc_shape[0],
        &A_sc_stride[0],
        &A_sc_smem_shape[0],
        &A_sc_smem_stride[0],
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    static constexpr int DYNAMIC_SMEM = TILE_M * TILE_N * sizeof(bf16) + 1024;
    dim3 grid = dim3(N / TILE_N, M / TILE_M);
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SMEM));
    kernel<<<grid, TILE_M, DYNAMIC_SMEM>>>(
        get_data_ptr<bf16>(A_bf16), A_bf16_tmap, get_data_ptr<fp8e4m3>(A_fp8), A_fp8_tmap, get_data_ptr<fp8e8m0>(A_sc), A_sc_tmap
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &launch_kernel);
}
