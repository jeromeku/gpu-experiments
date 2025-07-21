#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

// ThunderKittens macro check
#if !defined(KITTENS_HOPPER) || !defined(KITTENS_BLACKWELL)
    #error "KITTENS_HOPPER and KITTENS_BLACKWELL macros must be defined for Blackwell compilation"
#endif

// Kernel configuration
struct config {
    static constexpr int SM_COUNT = 148;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 2;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 40;
    static constexpr int CONSUMER_REGISTERS = 232;

    static constexpr int PIPELINE_STAGES = 4;
};

// Kernel globals
struct globals {
    // 1. The block size should be equivalent to the quantization block size
    // 2. The block should be square (for transpose)
    static constexpr int BLOCK_SIZE = 128;
    static constexpr int QUANT_BLOCK_SIZE = 32;

    using A_tile_bf16 = st_bf<BLOCK_SIZE, BLOCK_SIZE>;
    using A_tile_fp8 = st_fp8e4m3<BLOCK_SIZE, BLOCK_SIZE>;
    using A_sc_vec = sv<fp8e8m0, BLOCK_SIZE * BLOCK_SIZE / QUANT_BLOCK_SIZE>;

    gl<bf16, 1, -1, -1, -1, A_tile_bf16> A_bf16;
    gl<fp8e4m3, 1, -1, -1, -1, A_tile_fp8> A_fp8;
    gl<float, -1, -1, -1, -1, A_sc_vec> A_sc;

    __host__ inline dim3 grid() { return dim3(config::SM_COUNT); }
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }

    struct pipeline_inputs {
        A_tile_bf16 A_bf16;
    };

    struct pipeline_outputs {
        A_tile_fp8 A_fp8;
        A_sc_vec A_sc;
    };
};

// Kernel implementation
__global__  __launch_bounds__(config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    // Shared memory declaration
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    // Warpgroup configuration
    int lane_id = warp::laneid();
    int warp_id = warpgroup::warpid();
    int warpgroup_id = warpgroup::groupid();
    using consumer = group<config::CONSUMER_WARPGROUPS * WARPGROUP_WARPS>;

    // Allocate shared memory
    static_assert(sizeof(globals::pipeline_inputs) * config::PIPELINE_STAGES + sizeof(globals::pipeline_outputs) * config::PIPELINE_STAGES <= config::DYNAMIC_SHARED_MEMORY);
    globals::pipeline_inputs (&inputs)[config::PIPELINE_STAGES] = allocator.allocate<globals::pipeline_inputs, config::PIPELINE_STAGES>();
    globals::pipeline_outputs (&outputs)[config::PIPELINE_STAGES] = allocator.allocate<globals::pipeline_outputs, config::PIPELINE_STAGES>();

    // Set up mbarriers
    __shared__ semaphore inputs_arrived[config::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[config::PIPELINE_STAGES];
    if (threadIdx.x == 0) {
        for (int i = 0; i < config::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 2);
        }
    }
    __syncthreads();

    // Pipeline configuration
    int num_groups = G.A_bf16.depth();
    int num_blocks_per_row = G.A_bf16.cols() / globals::BLOCK_SIZE;
    int num_blocks_per_col = G.A_bf16.rows() / globals::BLOCK_SIZE;
    int num_blocks_per_group = num_blocks_per_row * num_blocks_per_col;
    int num_blocks = num_groups * num_blocks_per_group;

    // Declare phasebits for semaphore waits
    uint32_t phasebits = 0xFFFF0000;

    // Main divergence
    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        // Producer group
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        // Main loop
        int stage = 0;
        if (warp_id == 0 && lane_id == 0) {
            for (int block_idx = blockIdx.x; block_idx < num_blocks; block_idx += gridDim.x) {
                // Compute block indices
                int group_idx = block_idx / num_blocks_per_group;
                int group_local_block_idx = block_idx % num_blocks_per_group;
                int row_block_idx = group_local_block_idx / num_blocks_per_row;
                int col_block_idx = group_local_block_idx % num_blocks_per_row;

                // Wait for shared memory to be free
                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                update_phasebit<1>(phasebits, stage);

                // Load inputs into shared memory
                tma::expect_bytes(inputs_arrived[stage], sizeof(globals::pipeline_inputs));
                tma::load_async(inputs[stage].A_bf16, G.A_bf16, {group_idx, row_block_idx, col_block_idx}, inputs_arrived[stage]);

                // Update phasebit and stage
                stage = (stage + 1) % config::PIPELINE_STAGES;
            }
        }
    } else {
        // Consumer group
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

        // Main loop
        int stage = 0;
        for (int block_idx = blockIdx.x; block_idx < num_blocks; block_idx += gridDim.x) {
            // Compute block indices
            int group_idx = block_idx / num_blocks_per_group;
            int group_local_block_idx = block_idx % num_blocks_per_group;
            int row_block_idx = group_local_block_idx / num_blocks_per_row;
            int col_block_idx = group_local_block_idx % num_blocks_per_row;

            // Wait for inputs to arrive at shared memory
            wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
            update_phasebit<0>(phasebits, stage);

            // Load input
            rt_bf<globals::BLOCK_SIZE / 8, globals::BLOCK_SIZE> A_bf16;
            consumer::load(A_bf16, inputs[stage].A_bf16[warpgroup_id]);
            consumer::sync(1);
            consumer::arrive(inputs_finished[stage]);

            // Quantize
            // I think I can do this by reinterpret-casting tiles into 4 array
            rt_fl<globals::BLOCK_SIZE / 8, globals::BLOCK_SIZE> A_fl, A_fl_abs;
            rt_fp8e4m3<globals::BLOCK_SIZE / 8, globals::BLOCK_SIZE> A_fp8;
            col_vec<rt_fl<globals::BLOCK_SIZE / 8, globals::BLOCK_SIZE>> scale;
            warp::copy(A_fl, A_bf16);
            warp::abs(A_fl_abs, A_fl);
            warp::row_max(scale, A_fl_abs);
            warp::max(scale, scale, 0.000000000001f); // avoid division by zero
            warp::mul(scale, scale, 0.002232142857f); // 1 / 448
            warp::div_row(A_fl, A_fl, scale);
            warp::copy(A_fp8, A_fl);

            // Store results to shared memory
            consumer::store(outputs[stage].A_fp8[warpgroup_id], A_fp8);
            consumer::store(outputs[stage].A_sc[warpgroup_id], scale);
            consumer::sync(1);

            // Store results to global memory
            // TODO: why the fuck do I have C memory per stage? we only use single stage at a time. Reduce this and increase pipeline depth
            if (consumer::laneid() == 0) {
                tma::store_async(G.A_fp8, outputs[stage].A_fp8[warpgroup_id], {group_idx, row_block_idx, col_block_idx});
                tma::store_async(G.A_sc, outputs[stage].A_sc[warpgroup_id], {group_idx, col_block_idx, row_block_idx}); // column-major
                tma::store_async_read_wait();
            }
            consumer::sync(1);

            // Update phasebit and stage
            stage = (stage + 1) % config::PIPELINE_STAGES;
        }
    }
}

// Python bindings
PYBIND11_MODULE(_C, m) {
    m.doc() = "";
    kittens::py::bind_kernel<kernel>(m, "kernel",
        &globals::A_bf16,
        &globals::A_fp8,
        &globals::A_sc
    );
}
