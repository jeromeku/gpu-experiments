#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

namespace bf16_mha_fwd {

struct config {
    static constexpr int CLUSTER_SIZE = 2;

    static constexpr int NUM_BLOCKS = 148;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int NUM_CONSUMERS = 2;
    static constexpr int WARPGROUPS_PER_CONSUMER = 2;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPGROUPS = NUM_CONSUMERS * WARPGROUPS_PER_CONSUMER + NUM_PRODUCERS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 64;
    static constexpr int CONSUMER_REGISTERS = 104;
};

struct globals {
    static constexpr int BLOCK_SIZE = 128;

    static constexpr int QK_DIM = 128;
    static constexpr int VO_DIM = 128;

    static constexpr int PIPELINE_STAGES = 2;

    using Q_tile = st_bf<BLOCK_SIZE, QK_DIM>;
    using K_tile = st_bf<BLOCK_SIZE / 2, QK_DIM>;
    using V_tile = st_bf<BLOCK_SIZE, VO_DIM / 2>;
    using L_vec  = col_vec<st_fl<BLOCK_SIZE, VO_DIM>>;
    using O_tile = st_bf<BLOCK_SIZE, VO_DIM>;
    using A_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>;

    using Q_gl = gl<bf16,  -1, -1, -1, -1, Q_tile>; // B, H, N, D
    using K_gl = gl<bf16,  -1, -1, -1, -1, K_tile>;
    using V_gl = gl<bf16,  -1, -1, -1, -1, V_tile>;
    using L_gl = gl<float, -1, -1, -1, -1, L_vec>;
    using O_gl = gl<bf16,  -1, -1, -1, -1, O_tile>;

    Q_gl Q;
    K_gl K;
    V_gl V;
    L_gl L;
    O_gl O;

    __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
    __host__ inline dim3 grid()  { return dim3(config::NUM_BLOCKS); }
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }

    struct task_info {
        int batch_idx;
        int head_idx;
        int Q_block_idx;
        int KV_block_start;
        int KV_block_end;
    };
};

__device__ static inline globals::task_info get_task_info(const globals &G, int task_idx) {
    constexpr int Q_block_size = 2 * config::NUM_CONSUMERS * globals::BLOCK_SIZE;
    const int num_QO_blocks = G.K.rows() / globals::BLOCK_SIZE;

    const int B = G.Q.batch();
    const int H = G.Q.depth();
    const int num_Q_blocks = (G.Q.rows() + Q_block_size - 1) / Q_block_size;

    globals::task_info task_info;
    task_info.batch_idx = task_idx / (H * num_Q_blocks);
    task_idx -= task_info.batch_idx * H * num_Q_blocks;
    task_info.head_idx  = task_idx / num_Q_blocks;
    task_idx -= task_info.head_idx * num_Q_blocks;
    task_info.Q_block_idx = task_idx;

    task_info.KV_block_start = 0;
    task_info.KV_block_end = num_QO_blocks;

    if (task_info.batch_idx >= B)
        return { -1, -1, -1 };
    else
        return task_info;
}

struct rescale_add {
    template<typename T> static __device__ inline T op(const T &a, const T &b) {
        if constexpr (std::is_same_v<T, float2>) {
            constexpr float2 scale = {1.44269504089f*0.08838834764f, 1.44269504089f*0.08838834764f};
            float2 c;
            asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(*(uint64_t*)&c) : "l"(*(uint64_t*)&a), "l"(*(uint64_t*)&scale), "l"(*(uint64_t*)&b));
            return c;
        }
        else {
            static_assert(sizeof(T) == 999, "Currently unsupported type");
        }
    }
};

__global__ __cluster_dims__(config::CLUSTER_SIZE) __launch_bounds__(config::NUM_THREADS, 1)
static void kernel(const __grid_constant__ globals G) {
    // Declare shared memory
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);

    // Allocate shared memory
    static_assert(
        sizeof(globals::Q_tile) * config::NUM_CONSUMERS +
        sizeof(globals::K_tile) * globals::PIPELINE_STAGES +
        sizeof(globals::V_tile) * globals::PIPELINE_STAGES +
        sizeof(globals::L_vec) * config::NUM_CONSUMERS +
        sizeof(globals::A_tile) * config::NUM_CONSUMERS <= config::DYNAMIC_SHARED_MEMORY
    );
    globals::Q_tile (&Q_smem)[config::NUM_CONSUMERS]    = sm_allocator.allocate<globals::Q_tile, config::NUM_CONSUMERS>();
    globals::K_tile (&K_smem)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::K_tile, globals::PIPELINE_STAGES>();
    globals::V_tile (&V_smem)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::V_tile, globals::PIPELINE_STAGES>();
    globals::L_vec  (&L_smem)[config::NUM_CONSUMERS]    = sm_allocator.allocate<globals::L_vec, config::NUM_CONSUMERS>();
    globals::O_tile (&O_smem)[config::NUM_CONSUMERS]    = *reinterpret_cast<globals::O_tile(*)[config::NUM_CONSUMERS]>(&Q_smem[0]);
    globals::A_tile (&A_smem)[config::NUM_CONSUMERS]    = sm_allocator.allocate<globals::A_tile, config::NUM_CONSUMERS>();

    // Allocate tensor memory
    tensor_allocator<1, config::CLUSTER_SIZE> tm_allocator {};
    using tm_t = tt<float, globals::BLOCK_SIZE, globals::BLOCK_SIZE>;
    tm_t QK_tm[2] = {
        tm_allocator.allocate<tm_t>(globals::BLOCK_SIZE * 0),
        tm_allocator.allocate<tm_t>(globals::BLOCK_SIZE * 1)
    };
    tm_t AV_tm[2] = {
        tm_allocator.allocate<tm_t>(globals::BLOCK_SIZE * 2),
        tm_allocator.allocate<tm_t>(globals::BLOCK_SIZE * 3)
    };

    // Set up mbarriers
    __shared__ semaphore Q_arrived;
    __shared__ semaphore K_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore V_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore K_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore V_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore A_unloaded[config::NUM_CONSUMERS];
    __shared__ semaphore A_loaded[config::NUM_CONSUMERS];
    __shared__ semaphore QK_finished[config::NUM_CONSUMERS];
    __shared__ semaphore AV_finished[config::NUM_CONSUMERS];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < globals::PIPELINE_STAGES; ++i) {
            init_semaphore(K_arrived[i], 0, config::CLUSTER_SIZE);
            init_semaphore(V_arrived[i], 0, config::CLUSTER_SIZE);
            init_semaphore(K_finished[i], 0, 1);
            init_semaphore(V_finished[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::NUM_CONSUMERS; ++i) {
            init_semaphore(Q_arrived, 0, config::CLUSTER_SIZE * config::NUM_CONSUMERS);
            init_semaphore(A_unloaded[i], 0, config::CLUSTER_SIZE);
            init_semaphore(A_loaded[i], 0, config::CLUSTER_SIZE);
            init_semaphore(QK_finished[i], 0, 1);
            init_semaphore(AV_finished[i], 0, 1);
        }
    }
    everyone::tma::cluster::sync();

    // Pipeline configuration
    const int cluster_id = clusterIdx().x;
    const int cta_id = cluster_ctarank();

    // Constants
    constexpr float SQRT_D_INV = 0.08838834764f; // 1 / sqrt(128)
    constexpr float NEG_SQRT_D = -11.313708499f; // -sqrt(128)
    constexpr float LOG2E = 1.44269504089f;

    // Main divergence
    if (warpgroup::groupid() == config::NUM_WARPGROUPS - 1) {
        // Producer group
        // warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();
        const int warp_id = warpgroup::warpid();
        const int lane_id = warp::laneid();

        // Declare stage and phasebits for semaphore waits
        int stage = 0;
        uint32_t phasebits = 0xFFFF0000;

        if (warp_id == 0 && lane_id == 0) {
            // K loader
            for (int task_idx = cluster_id; true; task_idx += gridDim.x / 2) {
                globals::task_info task_info = get_task_info(G, task_idx);
                if (task_info.batch_idx == -1) break;

                for (int i = task_info.KV_block_start; i < task_info.KV_block_end; ++i) {
                    wait(K_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);

                    tma::cluster::expect(K_arrived[stage], 0, K_smem[stage]);
                    tma::cluster::load_async(K_smem[stage], G.K, 
                                             {task_info.batch_idx, task_info.head_idx, 2 * i + cta_id, 0},
                                             K_arrived[stage], (uint16_t)(1 << cta_id), 0);

                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        } else if (warp_id == 1 && lane_id == 0) {
            // V loader
            for (int task_idx = cluster_id; true; task_idx += gridDim.x / 2) {
                globals::task_info task_info = get_task_info(G, task_idx);
                if (task_info.batch_idx == -1) break;

                for (int i = task_info.KV_block_start; i < task_info.KV_block_end; ++i) {
                    wait(V_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);

                    tma::cluster::expect(V_arrived[stage], 0, V_smem[stage]);
                    tma::cluster::load_async(V_smem[stage], G.V, 
                                             {task_info.batch_idx, task_info.head_idx, i, cta_id},
                                             V_arrived[stage], (uint16_t)(1 << cta_id), 0);

                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        } else if (cta_id == 0 && warp_id == 2 && lane_id == 0) {
            // QK launcher
            for (int task_idx = cluster_id; true; task_idx += gridDim.x / 2) {
                globals::task_info task_info = get_task_info(G, task_idx);
                if (task_info.batch_idx == -1) break;

                tma::cluster::wait(Q_arrived, get_phasebit<0>(phasebits, globals::PIPELINE_STAGES));
                update_phasebit<0>(phasebits, globals::PIPELINE_STAGES);

                for (int i = task_info.KV_block_start; i < task_info.KV_block_end; ++i) {
                    tma::cluster::wait(K_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);

                    #pragma unroll
                    for (int consumer_id = 0; consumer_id < config::NUM_CONSUMERS; ++consumer_id) {
                        tma::cluster::wait(A_unloaded[consumer_id], get_phasebit<1>(phasebits, globals::PIPELINE_STAGES + consumer_id));
                        update_phasebit<1>(phasebits, globals::PIPELINE_STAGES + consumer_id);
                        mm2_ABt(QK_tm[consumer_id], Q_smem[consumer_id], K_smem[stage], QK_finished[consumer_id]);
                    }

                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        } else if (cta_id == 0 && warp_id == 3 && lane_id == 0) {
            // AV launcher
            for (int task_idx = cluster_id; true; task_idx += gridDim.x / 2) {
                globals::task_info task_info = get_task_info(G, task_idx);
                if (task_info.batch_idx == -1) break;

                for (int i = task_info.KV_block_start; i < task_info.KV_block_end; ++i) {
                    tma::cluster::wait(V_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);

                    #pragma unroll
                    for (int consumer_id = 0; consumer_id < config::NUM_CONSUMERS; ++consumer_id) {
                        tma::cluster::wait(A_loaded[consumer_id], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES + consumer_id));
                        update_phasebit<0>(phasebits, globals::PIPELINE_STAGES + consumer_id);
                        mma2_AB(AV_tm[consumer_id], A_smem[consumer_id], V_smem[stage], AV_finished[consumer_id]);
                    }

                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        }
    } else {
        // Consumer group
        // warpgroup::increase_registers<config::CONSUMER_REGISTERS>();
        using all_consumers = group<config::NUM_CONSUMERS * config::WARPGROUPS_PER_CONSUMER * WARPGROUP_WARPS>;
        using consumer = group<config::WARPGROUPS_PER_CONSUMER * WARPGROUP_WARPS>;
        const int consumer_id = consumer::groupid();
        constexpr int ROWS_PER_WARP = globals::BLOCK_SIZE / config::WARPGROUPS_PER_CONSUMER / WARPGROUP_WARPS;

        // Declare stage and phasebits for semaphore waits
        int K_stage = 0;
        int V_stage = 0;
        uint32_t phasebits = 0xFFFF0000;

        for (int task_idx = cluster_id; true; task_idx += gridDim.x / 2) {
            globals::task_info task_info = get_task_info(G, task_idx);
            if (task_info.batch_idx == -1) break;

            // Load Q
            if (consumer::laneid() == 0) {
                tma::cluster::expect(Q_arrived, 0, Q_smem[consumer_id]);
                tma::cluster::load_async(Q_smem[consumer_id], G.Q,
                                         {task_info.batch_idx, task_info.head_idx, 
                                          2 * config::NUM_CONSUMERS * task_info.Q_block_idx + 
                                          config::NUM_CONSUMERS * cta_id + consumer_id, 0},
                                         Q_arrived, (uint16_t)(1 << cta_id), 0);
            }

            rt_fl<ROWS_PER_WARP, globals::VO_DIM> O_reg;
            col_vec<rt_fl<ROWS_PER_WARP, globals::BLOCK_SIZE>> max_vec, norm_vec;
            warp::zero(O_reg);
            warp::neg_infty(max_vec);
            warp::zero(norm_vec);

            for (int i = task_info.KV_block_start; i < task_info.KV_block_end; ++i) {
                tma::cluster::wait(QK_finished[consumer_id], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES + 0));
                update_phasebit<0>(phasebits, globals::PIPELINE_STAGES + 0);
                if (all_consumers::laneid() == 0) arrive(K_finished[K_stage]);
                K_stage = (K_stage + 1) % globals::PIPELINE_STAGES;

                rt_fl<ROWS_PER_WARP, globals::BLOCK_SIZE> A_fl_reg;
                consumer::load_async(A_fl_reg, QK_tm[consumer_id]);
                tensor_load_wait();
                consumer::sync(1 + consumer_id);
                if (consumer::laneid() == 0)
                    tma::cluster::arrive(A_unloaded[consumer_id], 0);

                // Perform softmax
                rt_bf<ROWS_PER_WARP, globals::BLOCK_SIZE> A_bf_reg;
                col_vec<rt_fl<ROWS_PER_WARP, globals::BLOCK_SIZE>> max_vec_last_scaled, max_vec_scaled;
                warp::mul(max_vec_last_scaled, max_vec, LOG2E * SQRT_D_INV);
                warp::row_max(max_vec, A_fl_reg, max_vec);
                warp::mul(max_vec_scaled, max_vec, -LOG2E * SQRT_D_INV);
                warp::row_map<rescale_add>(A_fl_reg, A_fl_reg, max_vec_scaled);
                warp::exp2(A_fl_reg, A_fl_reg);
                warp::add(max_vec_last_scaled, max_vec_scaled, max_vec_last_scaled);
                warp::exp2(max_vec_last_scaled, max_vec_last_scaled);
                warp::mul(norm_vec, max_vec_last_scaled, norm_vec);
                warp::row_sum(norm_vec, A_fl_reg, norm_vec);
                warp::copy(A_bf_reg, A_fl_reg);

                if (i > task_info.KV_block_start) {
                    tma::cluster::wait(AV_finished[consumer_id], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES + 1));
                    update_phasebit<0>(phasebits, globals::PIPELINE_STAGES + 1);
                    if (all_consumers::laneid() == 0) arrive(V_finished[V_stage]);
                    V_stage = (V_stage + 1) % globals::PIPELINE_STAGES;
                }

                consumer::load_async(O_reg, AV_tm[consumer_id]);
                consumer::store(A_smem[consumer_id], A_bf_reg);
                warp::mul_row(O_reg, O_reg, max_vec_last_scaled);
                consumer::store_async(AV_tm[consumer_id], O_reg);
                tensor_store_wait();
                consumer::sync(1 + consumer_id);
                if (consumer::laneid() == 0)
                    tma::cluster::arrive(A_loaded[consumer_id], 0);
            }

            // Wait for the last AV to finished
            tma::cluster::wait(AV_finished[consumer_id], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES + 1));
            update_phasebit<0>(phasebits, globals::PIPELINE_STAGES + 1);
            if (all_consumers::laneid() == 0) arrive(V_finished[V_stage]);
            V_stage = (V_stage + 1) % globals::PIPELINE_STAGES;

            consumer::load_async(O_reg, AV_tm[consumer_id]);
            warp::div_row(O_reg, O_reg, norm_vec);
            consumer::store(O_smem[consumer_id], O_reg);
            consumer::sync(1 + consumer_id);
            if (consumer::laneid() == 0) {
                tma::store_async(G.O, O_smem[consumer_id], 
                                 {task_info.batch_idx, task_info.head_idx, 
                                  2 * config::NUM_CONSUMERS * task_info.Q_block_idx + 
                                  config::NUM_CONSUMERS * cta_id + consumer_id, 0});
            }

            warp::mul(max_vec, max_vec, SQRT_D_INV);
            warp::log(norm_vec, norm_vec);
            warp::add(norm_vec, norm_vec, max_vec);
            warp::mul(norm_vec, norm_vec, NEG_SQRT_D);

            consumer::store(L_smem[consumer_id], norm_vec);
            consumer::sync(1 + consumer_id);
            if (consumer::laneid() == 0) {
                tma::store_async(G.L, L_smem[consumer_id], 
                                 {task_info.batch_idx, task_info.head_idx, 0, 
                                  2 * config::NUM_CONSUMERS * task_info.Q_block_idx + 
                                  config::NUM_CONSUMERS * cta_id + consumer_id});
                tma::store_async_read_wait();
            }
            consumer::sync(1 + consumer_id);
        }
    }

    // For TM deallocation
    everyone::tma::cluster::sync();
}

} // namespace bf16_mha_fwd

namespace bf16_mha_bwd_prep {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;
    static constexpr int NUM_THREADS = 128;
};

struct globals {
    static constexpr int BLOCK_SIZE = 128;
    static constexpr int VO_DIM = 128;

    using O_grad_tile = st_bf<BLOCK_SIZE, VO_DIM>;
    using O_tile = st_bf<BLOCK_SIZE, VO_DIM>;
    using D_vec = col_vec<st_fl<BLOCK_SIZE, VO_DIM>>;

    using O_grad_gl = gl<bf16, -1, -1, -1, -1, O_grad_tile>;
    using O_gl = gl<bf16, -1, -1, -1, -1, O_tile>;
    using D_gl = gl<float, -1, -1, -1, -1, D_vec>;

    O_grad_gl O_grad;
    O_gl O;
    D_gl D;

    __host__ inline int dynamic_shared_memory() { return sizeof(O_grad_tile) + sizeof(O_tile) + sizeof(D_vec) + 1024; }
    __host__ inline dim3 grid()  { return dim3(O_grad.rows() / BLOCK_SIZE, O_grad.depth(), O_grad.batch()); }
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
};

__global__ __launch_bounds__(config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    // Declare shared memory
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);

    // Allocate shared memory
    globals::O_grad_tile &O_grad_smem = sm_allocator.allocate<globals::O_grad_tile>();
    globals::O_tile &O_smem = sm_allocator.allocate<globals::O_tile>();
    globals::D_vec &D_smem = sm_allocator.allocate<globals::D_vec>();

    // Retrieve indices
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int QO_block_idx = blockIdx.x;

    // Set up mbarriers
    __shared__ semaphore inputs_arrived;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, sizeof(globals::O_grad_tile) + sizeof(globals::O_tile));
        tma::load_async(O_grad_smem, G.O_grad, {batch_idx, head_idx, QO_block_idx, 0}, inputs_arrived);
        tma::load_async(O_smem, G.O, {batch_idx, head_idx, QO_block_idx, 0}, inputs_arrived);
    }
    __syncthreads();

    // Wait and load
    rt_fl<globals::BLOCK_SIZE / WARPGROUP_WARPS, globals::VO_DIM> O_grad_reg;
    rt_fl<globals::BLOCK_SIZE / WARPGROUP_WARPS, globals::VO_DIM> O_reg;
    wait(inputs_arrived, 0);
    warpgroup::load(O_grad_reg, O_grad_smem);
    warpgroup::load(O_reg, O_smem);

    // Compute
    col_vec<rt_fl<globals::BLOCK_SIZE / WARPGROUP_WARPS, globals::VO_DIM>> D_reg;
    warp::mul(O_grad_reg, O_grad_reg, O_reg);
    warp::row_sum(D_reg, O_grad_reg, D_reg);

    // Store to SM
    warpgroup::store(D_smem, D_reg);
    __syncthreads();

    // Store to GMEM
    if (threadIdx.x == 0) {
        tma::store_async(G.D, D_smem, {batch_idx, head_idx, 0, QO_block_idx});
    }
}

} // namespace bf16_mha_bwd_prep

namespace bf16_mha_bwd {

struct config {
    static constexpr int CLUSTER_SIZE = 1;

    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int NUM_CONSUMERS = 2;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPGROUPS = NUM_CONSUMERS + NUM_PRODUCERS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int BLOCK_SIZE = 64;

    static constexpr int QK_DIM = 128;
    static constexpr int VO_DIM = 128;

    static constexpr int PIPELINE_STAGES = 2;

    using Q_tile  = st_bf<BLOCK_SIZE, QK_DIM>;
    using K_tile  = st_bf<BLOCK_SIZE, QK_DIM>;
    using V_tile  = st_bf<BLOCK_SIZE, VO_DIM>;
    using O_grad_tile = st_bf<BLOCK_SIZE, VO_DIM>;
    using Q_grad_tile = st_fl<BLOCK_SIZE, QK_DIM>;
    using K_grad_tile = st_fl<BLOCK_SIZE, QK_DIM>;
    using V_grad_tile = st_fl<BLOCK_SIZE, VO_DIM>;
    using L_vec  = row_vec<st_fl<BLOCK_SIZE, BLOCK_SIZE>>;
    using D_vec  = row_vec<st_fl<BLOCK_SIZE, BLOCK_SIZE>>;
    using SP_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>; 

    using Q_gl = gl<bf16,  -1, -1, -1, -1, Q_tile>;
    using K_gl = gl<bf16,  -1, -1, -1, -1, K_tile>;
    using V_gl = gl<bf16,  -1, -1, -1, -1, V_tile>;
    using O_grad_gl = gl<bf16,  -1, -1, -1, -1, O_grad_tile>;
    using Q_grad_gl = gl<float, -1, -1, -1, -1, Q_grad_tile>;
    using K_grad_gl = gl<float, -1, -1, -1, -1, K_grad_tile>;
    using V_grad_gl = gl<float, -1, -1, -1, -1, V_grad_tile>;
    using L_gl = gl<float, -1, -1, -1, -1, L_vec>;
    using D_gl = gl<float, -1, -1, -1, -1, D_vec>; 

    Q_gl Q;
    K_gl K;
    V_gl V;
    O_grad_gl O_grad;
    Q_grad_gl Q_grad;
    K_grad_gl K_grad;
    V_grad_gl V_grad;
    L_gl L;
    D_gl D;

    __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
    __host__ inline dim3 grid()  { return dim3(K.rows() / (BLOCK_SIZE * 2), Q.depth(), Q.batch()); }
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
};

__global__ __launch_bounds__(config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    // Set up indices
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int KV_block_idx = blockIdx.x;
    const int num_QO_blocks = G.Q.rows() / globals::BLOCK_SIZE; // Q is 64x128, KV is 128x128
    const int warpgroup_id = warpgroup::groupid();

    // Declare shared memory
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);

    // Allocate shared memory
    static_assert(
        sizeof(globals::Q_tile) * globals::PIPELINE_STAGES +
        sizeof(globals::K_tile) * config::NUM_CONSUMERS +
        sizeof(globals::V_tile) * config::NUM_CONSUMERS +
        sizeof(globals::O_grad_tile) * globals::PIPELINE_STAGES +
        sizeof(globals::Q_grad_tile) +
        sizeof(globals::L_vec) * globals::PIPELINE_STAGES +
        sizeof(globals::D_vec) * globals::PIPELINE_STAGES +
        sizeof(globals::SP_tile) * config::NUM_CONSUMERS <= config::DYNAMIC_SHARED_MEMORY
    );
    globals::Q_tile (&Q_smem)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::Q_tile, globals::PIPELINE_STAGES>();
    globals::K_tile (&K_smem)[config::NUM_CONSUMERS] = sm_allocator.allocate<globals::K_tile, config::NUM_CONSUMERS>();
    globals::V_tile (&V_smem)[config::NUM_CONSUMERS] = sm_allocator.allocate<globals::V_tile, config::NUM_CONSUMERS>();
    globals::O_grad_tile (&O_grad_smem)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::O_grad_tile, globals::PIPELINE_STAGES>(); 
    globals::Q_grad_tile (&Q_grad_smem) = sm_allocator.allocate<globals::Q_grad_tile>();
    globals::K_grad_tile (&K_grad_smem)[config::NUM_CONSUMERS] = *reinterpret_cast<globals::K_grad_tile(*)[config::NUM_CONSUMERS]>(&Q_smem[0].data[0]);
    globals::V_grad_tile (&V_grad_smem)[config::NUM_CONSUMERS] = *reinterpret_cast<globals::V_grad_tile(*)[config::NUM_CONSUMERS]>(&V_smem[0].data[0]);
    globals::L_vec (&L_smem)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::L_vec, globals::PIPELINE_STAGES>();
    globals::D_vec (&D_smem)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::D_vec, globals::PIPELINE_STAGES>(); 
    globals::SP_tile (&dS_smem)[config::NUM_CONSUMERS] = sm_allocator.allocate<globals::SP_tile, config::NUM_CONSUMERS>();

    // Allocate tensor memory
    tensor_allocator<1, config::CLUSTER_SIZE> tm_allocator {};

    // Set up mbarriers and launch initial loads
    __shared__ semaphore Q_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore KV_arrived[config::NUM_CONSUMERS];
    __shared__ semaphore LD_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore O_grad_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore Q_grad_ready;
    __shared__ semaphore matmul_finished[config::NUM_CONSUMERS];
    __shared__ semaphore compute_done[globals::PIPELINE_STAGES];
    if (threadIdx.x == 0) {
        // Initialize mbarriers
        #pragma unroll
        for (int i = 0; i < globals::PIPELINE_STAGES; i++) {
            init_semaphore(Q_arrived[i], 0, 1);
            init_semaphore(LD_arrived[i], 0, 1);
            init_semaphore(O_grad_arrived[i], 0, 1); 
            init_semaphore(compute_done[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < config::NUM_CONSUMERS; i++) {
            init_semaphore(KV_arrived[i], 0, 1);
            init_semaphore(matmul_finished[i], 0, 1);
        }
        init_semaphore(Q_grad_ready, 0, 1);

        // Load K and V
        #pragma unroll
        for (int i = 0; i < config::NUM_CONSUMERS; i++) {
            tma::expect_bytes(KV_arrived[i], sizeof(K_smem[i]) + sizeof(V_smem[i]));
            tma::load_async(K_smem[i], G.K, {batch_idx, head_idx, (KV_block_idx * config::NUM_CONSUMERS) + i, 0}, KV_arrived[i]);
            tma::load_async(V_smem[i], G.V, {batch_idx, head_idx, (KV_block_idx * config::NUM_CONSUMERS) + i, 0}, KV_arrived[i]);
        }
    }
    __syncthreads();

    // Main divergence
    if (warpgroup_id == config::NUM_CONSUMERS) {
        // Loader/storer group
        warpgroup::decrease_registers<24>();
        const int warp_id = warpgroup::warpid();
        const int lane_id = warp::laneid();

        // Declare stage and phasebits for semaphore waits
        int stage = 0;
        uint32_t phasebits = 0xFFFF0000;

        if (warp_id == 0 && lane_id == 0) { // Loader (TODO: separate L/D/Q/dO and apply per signaling)
            for (int i = 0; i < num_QO_blocks; i++) {
                wait(compute_done[stage], get_phasebit<1>(phasebits, stage));
                update_phasebit<1>(phasebits, stage);

                // Load L and D
                tma::expect_bytes(LD_arrived[stage], sizeof(L_smem[stage]) + sizeof(D_smem[stage]));
                tma::load_async(L_smem[stage], G.L, {batch_idx, head_idx, 0, i}, LD_arrived[stage]);
                tma::load_async(D_smem[stage], G.D, {batch_idx, head_idx, 0, i}, LD_arrived[stage]);

                // Load Q
                tma::expect_bytes(Q_arrived[stage], sizeof(Q_smem[stage]));
                tma::load_async(Q_smem[stage], G.Q, {batch_idx, head_idx, i, 0}, Q_arrived[stage]);

                // Load O_grad
                tma::expect_bytes(O_grad_arrived[stage], sizeof(O_grad_smem[stage]));
                tma::load_async(O_grad_smem[stage], G.O_grad, {batch_idx, head_idx, i, 0}, O_grad_arrived[stage]);

                stage = (stage + 1) % globals::PIPELINE_STAGES;
            }
        } else if (warp_id == 1 && lane_id == 0) { // Storer (TODO: is this even necessary)
            for (int i = 0; i < num_QO_blocks; i++) {
                wait(compute_done[stage], get_phasebit<0>(phasebits, stage));
                update_phasebit<0>(phasebits, stage);

                tma::store_add_async(G.Q_grad, Q_grad_smem, {batch_idx, head_idx, i, 0});
                tma::store_async_read_wait();
                arrive(Q_grad_ready);

                stage = (stage + 1) % globals::PIPELINE_STAGES;
            }
        }
    } else {
        // Consumer group
        if (warpgroup_id == 0)
            warpgroup::increase_registers<256>();
        else
            warpgroup::increase_registers<224>();

        // Constants
        constexpr float SQRT_D_INV = 0.08838834764f; // 1 / sqrt(128)
        constexpr float LOG2E = 1.44269504089f;

        // Declare stage and phasebits for semaphore waits
        int stage = 0;
        uint32_t phasebits = 0xFFFF0000;

        // Declare K and V TMEM
        auto K_grad_tm = tm_allocator.allocate<tt<float, 64, 128>>(warpgroup_id, 0);
        auto V_grad_tm = tm_allocator.allocate<tt<float, 64, 128>>(warpgroup_id, 128);

        // Wait for K & V to be loaded
        wait(KV_arrived[warpgroup_id], 0);

        for (int i = 0; i < num_QO_blocks; i++) {
            // Wait for L & D to be loaded
            wait(LD_arrived[stage], get_phasebit<0>(phasebits, stage));

            // Broadcast the L vec row (1x64) to all of the rows in the S^T tile (64x64)
            // This makes S_t = -LSE * sqrt(d)
            rt_fl<16, 64> SP_t_reg;
            #pragma unroll
            for(int ii = 0; ii < 4; ii++) {
                int base_col = 16 * ii + 2 * (warp::laneid() % 4);
                SP_t_reg.tiles[0][ii].data[0] = *(float2*)&L_smem[stage][base_col + 0];
                SP_t_reg.tiles[0][ii].data[1] = *(float2*)&L_smem[stage][base_col + 0];
                SP_t_reg.tiles[0][ii].data[2] = *(float2*)&L_smem[stage][base_col + 8];
                SP_t_reg.tiles[0][ii].data[3] = *(float2*)&L_smem[stage][base_col + 8];
            }

            // S_t = (QK^T)^T - LSE * sqrt(d)
            auto S_t_tm = tm_allocator.allocate<tt<float, 64, 64>>(warpgroup_id, 256);
            warpgroup::store_async(S_t_tm, SP_t_reg);
            tensor_store_wait();
            wait(Q_arrived[stage], get_phasebit<0>(phasebits, stage));
            warpgroup::mma_ABt(S_t_tm, K_smem[warpgroup_id], Q_smem[stage]);

            // dP_t = (dO @ V^T)^T
            auto dP_t_tm = tm_allocator.allocate<tt<float, 64, 64>>(warpgroup_id, 320);
            wait(O_grad_arrived[stage], get_phasebit<0>(phasebits, stage));
            warpgroup::mm_ABt(dP_t_tm, V_smem[warpgroup_id], O_grad_smem[stage], matmul_finished[warpgroup_id]);

            // Wait for the matmuls to finish and load
            rt_fl<16, 64> dSP_t_reg;
            wait(matmul_finished[warpgroup_id], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES));
            update_phasebit<0>(phasebits, globals::PIPELINE_STAGES);
            warpgroup::load_async(SP_t_reg, S_t_tm);
            warpgroup::load_async(dSP_t_reg, dP_t_tm);

            // S_t = ( (QK^T)^T / sqrt(d) - LSE ) * log2(e)
            warp::mul(SP_t_reg, SP_t_reg, SQRT_D_INV * LOG2E);
            // P_t = S_t = exp( (QK^T)^T / sqrt(d) - LSE )
            warp::exp2(SP_t_reg, SP_t_reg);

            // Move P_t to TMEM
            auto &P_t_bf_tm = reinterpret_cast<tt<bf16, 64, 64>&>(S_t_tm);
            rt_bf<16, 64> SP_t_bf_reg;
            warp::copy(SP_t_bf_reg, SP_t_reg);
            warpgroup::store_async(P_t_bf_tm, SP_t_bf_reg);

            // dP_t = (dO @ V^T)^T - D
            #pragma unroll
            for(int ii = 0; ii < 4; ii++) {
                int base_col = 16 * ii + 2 * (warp::laneid() % 4);
                dSP_t_reg.tiles[0][ii].data[0] = base_ops::sub::template op<float2>(dSP_t_reg.tiles[0][ii].data[0], *(float2*)&D_smem[stage][base_col + 0]);
                dSP_t_reg.tiles[0][ii].data[1] = base_ops::sub::template op<float2>(dSP_t_reg.tiles[0][ii].data[1], *(float2*)&D_smem[stage][base_col + 0]);
                dSP_t_reg.tiles[0][ii].data[2] = base_ops::sub::template op<float2>(dSP_t_reg.tiles[0][ii].data[2], *(float2*)&D_smem[stage][base_col + 8]);
                dSP_t_reg.tiles[0][ii].data[3] = base_ops::sub::template op<float2>(dSP_t_reg.tiles[0][ii].data[3], *(float2*)&D_smem[stage][base_col + 8]);
            }
            // dS_t = P_t * ( (dO @ V^T)^T - D )
            warp::mul(dSP_t_reg, SP_t_reg, dSP_t_reg);
            // dS_t = P_t * ( (dO @ V^T)^T - D ) / sqrt(d)
            warp::mul(dSP_t_reg, dSP_t_reg, SQRT_D_INV);

            // dV = P^T @ O_grad
            tensor_store_wait();
            warpgroup::sync(warpgroup_id + 2);
            if (i == 0)
                warpgroup::mm_AB(V_grad_tm, P_t_bf_tm, O_grad_smem[stage]);
            else
                warpgroup::mma_AB(V_grad_tm, P_t_bf_tm, O_grad_smem[stage]);

            // dK = dS^T @ Q
            auto &dS_t_bf_tm = reinterpret_cast<tt<bf16, 64, 64>&>(dP_t_tm);
            rt_bf<16, 64> dSP_t_bf_reg;
            warp::copy(dSP_t_bf_reg, dSP_t_reg);
            warpgroup::store_async(dS_t_bf_tm, dSP_t_bf_reg);
            tensor_store_wait();
            warpgroup::sync(warpgroup_id + 2);
            if (i == 0)
                warpgroup::mm_AB(K_grad_tm, dS_t_bf_tm, Q_smem[stage], matmul_finished[warpgroup_id]);
            else
                warpgroup::mma_AB(K_grad_tm, dS_t_bf_tm, Q_smem[stage], matmul_finished[warpgroup_id]);

            warpgroup::store(dS_smem[warpgroup_id], dSP_t_reg);
            wait(matmul_finished[warpgroup_id], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES));
            update_phasebit<0>(phasebits, globals::PIPELINE_STAGES);
            group<config::NUM_CONSUMERS * WARPGROUP_WARPS>::sync(1);

            if (warpgroup_id == 0) {
                // dQ = dS_t @ K
                auto Q_grad_tm = tm_allocator.allocate<tt<float, 64, 128>>(0, 384); // just used by warpgroup 0
                warpgroup::mm_AtB(Q_grad_tm, dS_smem[0], K_smem[0]);
                warpgroup::mma_AtB(Q_grad_tm, dS_smem[1], K_smem[1], matmul_finished[0]);

                // Wait for matrix multiply to complete and SMEM to be empty
                wait(matmul_finished[0], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES));
                update_phasebit<0>(phasebits, globals::PIPELINE_STAGES);
                wait(Q_grad_ready, get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
                update_phasebit<1>(phasebits, globals::PIPELINE_STAGES);

                // Store Q_grad and signal
                rt_fl<16, 128> Q_grad_reg;
                warpgroup::load_async(Q_grad_reg, Q_grad_tm);
                warpgroup::store(Q_grad_smem, Q_grad_reg);
                warpgroup::sync(warpgroup_id + 2);
                warpgroup::arrive(compute_done[stage]);
            }

            update_phasebit<0>(phasebits, stage);
            stage = (stage + 1) % globals::PIPELINE_STAGES;
        }

        // Load finished K_grad and V_grad
        rt_fl<16, 128> K_grad_reg;
        rt_fl<16, 128> V_grad_reg;
        warpgroup::load_async(K_grad_reg, K_grad_tm);
        warpgroup::load_async(V_grad_reg, V_grad_tm);

        // Store V_grad
        warpgroup::store(V_grad_smem[warpgroup_id], V_grad_reg);
        warpgroup::sync(2 + warpgroup_id);
        warpgroup::tma::store_add_async(G.V_grad, V_grad_smem[warpgroup_id], {batch_idx, head_idx, (KV_block_idx * config::NUM_CONSUMERS) + warpgroup_id, 0});

        // Store K_grad
        wait(Q_grad_ready, get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
        update_phasebit<1>(phasebits, globals::PIPELINE_STAGES); // TODO: remove
        group<config::NUM_CONSUMERS * WARPGROUP_WARPS>::sync(1);
        warpgroup::store(K_grad_smem[warpgroup_id], K_grad_reg);
        warpgroup::sync(2 + warpgroup_id);
        warpgroup::tma::store_add_async(G.K_grad, K_grad_smem[warpgroup_id], {batch_idx, head_idx, (KV_block_idx * config::NUM_CONSUMERS) + warpgroup_id, 0});
    }
}

} // namespace bf16_mha_bwd

PYBIND11_MODULE(_C, m) {
    m.doc() = "";
    py::bind_kernel<bf16_mha_fwd::kernel>(m, "bf16_mha_fwd",
        &bf16_mha_fwd::globals::Q,
        &bf16_mha_fwd::globals::K,
        &bf16_mha_fwd::globals::V,
        &bf16_mha_fwd::globals::L,
        &bf16_mha_fwd::globals::O
    );
    py::bind_kernel<bf16_mha_bwd_prep::kernel>(m, "bf16_mha_bwd_prep",
        &bf16_mha_bwd_prep::globals::O_grad,
        &bf16_mha_bwd_prep::globals::O,
        &bf16_mha_bwd_prep::globals::D
    );
    py::bind_kernel<bf16_mha_bwd::kernel>(m, "bf16_mha_bwd",
        &bf16_mha_bwd::globals::Q,
        &bf16_mha_bwd::globals::K,
        &bf16_mha_bwd::globals::V,
        &bf16_mha_bwd::globals::O_grad,
        &bf16_mha_bwd::globals::Q_grad,
        &bf16_mha_bwd::globals::K_grad,
        &bf16_mha_bwd::globals::V_grad,
        &bf16_mha_bwd::globals::L,
        &bf16_mha_bwd::globals::D
    );
}
