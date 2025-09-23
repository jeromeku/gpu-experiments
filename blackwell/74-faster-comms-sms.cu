#include "kittens.cuh"
#include "ops/thread/memory/util/tma.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 1;
    static constexpr int NUM_BLOCKS = 148;
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int BLOCK_SIZE = 128;
    static constexpr int PIPELINE_STAGES = 6;

    using shared_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1, shared_tile>, NUM_DEVICES, false>;

    parallel_layout tensor;
    const int dev_idx;
    const int sm_id;
};

__device__ inline void kernel(const globals &G) {
    unsigned int sm_id;
    asm volatile("{mov.u32 %0, %%smid;}" : "=r"(sm_id));

    // Should be 148, although PTX docs says it could be larger
    // unsigned int n_sms;
    // asm volatile("{mov.u32 %0, %%nsmid;}" : "=r"(n_sms));

    if (G.dev_idx == 0 && sm_id == G.sm_id) {
        extern __shared__ int __shm[];
        tma_swizzle_allocator allocator((int*)&__shm[0]);
        globals::shared_tile (&tile)[globals::PIPELINE_STAGES] = allocator.allocate<globals::shared_tile, globals::PIPELINE_STAGES>();

        __shared__ semaphore inputs_arrived[globals::PIPELINE_STAGES];
        __shared__ semaphore inputs_finished[globals::PIPELINE_STAGES];
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int i = 0; i < globals::PIPELINE_STAGES; ++i) {
                init_semaphore(inputs_arrived[i], 0, 1);
                init_semaphore(inputs_finished[i], 0, 1);
            }
        }
        __syncthreads();

        const int num_row_blocks = G.tensor.rows() / globals::BLOCK_SIZE;
        const int num_col_blocks = G.tensor.cols() / globals::BLOCK_SIZE;

        int stage = 0;
        unsigned int phasebits = 0xFFFF'0000;

        if (threadIdx.x == 0) {
            for (int row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
                for (int col_block_idx = 0; col_block_idx < num_col_blocks; ++col_block_idx) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);
                    tma::expect_bytes(inputs_arrived[stage], sizeof(globals::shared_tile));
                    tma::load_async(tile[stage], G.tensor[0], {row_block_idx, col_block_idx}, inputs_arrived[stage]);
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        } else if (threadIdx.x == 32) {
            for (int row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
                for (int col_block_idx = 0; col_block_idx < num_col_blocks; ++col_block_idx) {
                    wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    tma::store_async(G.tensor[1], tile[stage], {row_block_idx, col_block_idx});
                    tma::store_async_read_wait();
                    arrive(inputs_finished[stage], 1);
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        }
    }
}

void entrypoint(kittens::py::TKParallelTensor &tensor, int sm_id) {
    globals G {
        .tensor = kittens::py::parallel_tensor_to_pgl<typename globals::parallel_layout>(tensor),
        .dev_idx = tensor.local_rank_,
        .sm_id = sm_id
    };

    kittens::py::launch_kernel<config, globals, kernel>(G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("p2p_from_dev0_to_dev1", &entrypoint);
}
