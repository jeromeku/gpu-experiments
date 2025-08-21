#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int SM_COUNT = 148;
    static constexpr int STATIC_SHARED_MEMORY = 128;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 56;
    static constexpr int CONSUMER_REGISTERS = 224;

    static constexpr int PIPELINE_STAGES = 3;
};

struct globals {
    static constexpr int BLOCK_SIZE = 128;

    using tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>;

    gl<bf16, 1, 1, -1, -1, tile> A;

    __host__ inline dim3 grid() { return dim3(config::SM_COUNT); }
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

constexpr int NUM_ITERS = 16;

__device__ inline void print_time(const char *name, uint64_t *start, uint64_t *end) {
    double avg = 0;
    for (int i = 0; i < NUM_ITERS; i++) {
        if (threadIdx.x == 0) printf("%s Iteration %d: %llu cycles\n", name, i, end[i] - start[i]);
        avg += end[i] - start[i];
    }

    avg /= NUM_ITERS;
    if (threadIdx.x == 0) printf("%s Average: %llf cycles\n", name, avg);
}

__global__ __launch_bounds__(config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    // Shared memory declaration
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    // Allocate shared and tensor memory
    globals::tile &A = allocator.allocate<globals::tile>();
    tensor_allocator<1, 1> tm_allocator {};

    // Set up mbarriers
    __shared__ semaphore inputs_arrived;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
    }
    __syncthreads();

    uint64_t start[NUM_ITERS];
    uint64_t end[NUM_ITERS];

    for (int i = 0; i < NUM_ITERS; i++) {
        start[i] = clock64();
        tma::expect_bytes(inputs_arrived, sizeof(globals::tile));
        tma::load_async(A, G.A, {0, 0}, inputs_arrived);
        wait(inputs_arrived, 0);
        end[i] = clock64();
    }

    print_time("TMA Load", start, end);

    




    // // Main divergence
    // if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
    //     // Producer group
    //     warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();
    //     int ctarank = cluster_ctarank();

    //     // Sub divergence
    //     if (warp_id == 3 && lane_id == 0) {
    //         // Producer group -- loaders
    //         for (int block_idx = clusterIdx().x; block_idx < num_blocks; block_idx += gridDim.x / config::CLUSTER_SIZE) {

    //             for (int i = 0; i < num_iters_per_block; ++i) {
    //                 tma::cluster::wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
    //                 if (stage == last_stage) {
    //                     arrive(outputs_arrived);
    //                     last_stage = -1;
    //                 }
    //                 tma::cluster::expect_bytes(inputs_arrived[stage], sizeof(globals::pipeline_inputs), 0);
    //                 tma::cluster::load_async(inputs[stage].A, G.A, {row_block_idx * 2 + ctarank, i}, inputs_arrived[stage], (uint16_t)(1 << ctarank), 0);
    //                 tma::cluster::load_async(inputs[stage].B, G.B, {col_block_idx * 2 + ctarank, i}, inputs_arrived[stage], (uint16_t)(1 << ctarank), 0);
    //                 update_phasebit<1>(phasebits, stage);
    //                 if (i == num_iters_per_block - 1) {
    //                     last_stage = stage;
    //                 }
    //                 stage = (stage + 1) % config::PIPELINE_STAGES;
    //             }
    //         }
    //         tma::cluster::wait(inputs_finished[last_stage], get_phasebit<1>(phasebits, last_stage));
    //         arrive(outputs_arrived);
    //     } else if (lane_id == 0 && ctarank == 0 && warp_id == 0) {
    //         // Producer group -- launchers
    //         auto tm = tm_allocator.allocate<tt<float, globals::ROW_BLOCK / 2, globals::COL_BLOCK>>(0);
    //         for (int block_idx = clusterIdx().x; block_idx < num_blocks; block_idx += gridDim.x / config::CLUSTER_SIZE) {
    //             tma::cluster::wait(tensors_finished, get_phasebit<1>(phasebits, config::PIPELINE_STAGES));
    //             update_phasebit<1>(phasebits, config::PIPELINE_STAGES);
    //             {
    //                 tma::cluster::wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
    //                 mm2_ABt(tm, inputs[stage].A, inputs[stage].B, inputs_finished[stage]);
    //                 update_phasebit<0>(phasebits, stage);
    //                 stage = (stage + 1) % config::PIPELINE_STAGES;
    //             }
    //             for (int i = 1; i < num_iters_per_block; ++i) {
    //                 tma::cluster::wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
    //                 mma2_ABt(tm, inputs[stage].A, inputs[stage].B, inputs_finished[stage]);
    //                 update_phasebit<0>(phasebits, stage);
    //                 stage = (stage + 1) % config::PIPELINE_STAGES;
    //             }
    //         }
    //     }
    // } else {
    //     // Consumer group
    //     warpgroup::increase_registers<config::CONSUMER_REGISTERS>();
    //     int ctarank = cluster_ctarank();
    //     auto tm = tm_allocator.allocate<tt<float, globals::ROW_BLOCK / 2, globals::COL_BLOCK>>(0);

    //         // Wait for the last matmul to complete
    //         wait(outputs_arrived, get_phasebit<0>(phasebits, config::PIPELINE_STAGES));
    //         update_phasebit<0>(phasebits, config::PIPELINE_STAGES);

    //         // Load the output from tensor memory into registers
    //         rt_bf<globals::ROW_BLOCK / 8, globals::COL_BLOCK / 8> C[8];
    //         #pragma unroll
    //         for (int i = 0; i < 8; i++)
    //             consumer::load_async(C[i], tm.subtile<tt<float, globals::ROW_BLOCK / 2, globals::COL_BLOCK / 8>>(0, i * globals::COL_BLOCK / 8));
    //         tensor_load_wait();
    //         consumer::sync(1);
    //         if (consumer::laneid() == 0)
    //             tma::cluster::arrive(tensors_finished, 0);

    //         // Store to global memory
    //         #pragma unroll
    //         for (int i = 0; i < 8; i++) {
    //             consumer::store(outputs.C, C[i]);
    //             consumer::sync(1);
    //             consumer::tma::store_async(G.C, outputs.C, {row_block_idx * 2 + ctarank, col_block_idx * 8 + i});
    //             consumer::tma::store_async_read_wait();
    //             consumer::sync(1);
    //         }
    //     }
    // }
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "";
    kittens::py::bind_kernel<kernel>(m, "kernel", &globals::A);
}
