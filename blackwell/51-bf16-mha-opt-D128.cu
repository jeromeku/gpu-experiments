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

    static constexpr int NUM_WARPGROUPS = 4;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 64;
    static constexpr int CONSUMER_REGISTERS = 104;
};

struct globals {
    static constexpr int BLOCK_SIZE = 128;

    static constexpr int QK_DIM = 128;
    static constexpr int VO_DIM = 128;

    static constexpr int PIPELINE_STAGES = 3;
    static constexpr int NUM_PIPELINES = 2;

    using Q_tile = st_bf<BLOCK_SIZE, QK_DIM>;          // 32768 B
    using K_tile = st_bf<BLOCK_SIZE / 2, QK_DIM>;      // 16384 B
    using V_tile = st_bf<BLOCK_SIZE, VO_DIM / 2>;      // 16384 B
    using L_vec  = col_vec<st_fl<BLOCK_SIZE, VO_DIM>>; // 512   B
    using O_tile = st_bf<BLOCK_SIZE, VO_DIM>;          // 32768 B

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

__device__ inline globals::task_info get_task_info(const globals &G, int task_idx) {
    constexpr int Q_block_size = config::CLUSTER_SIZE * globals::NUM_PIPELINES * globals::BLOCK_SIZE;

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
    task_info.KV_block_end = G.K.rows() / globals::BLOCK_SIZE;

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
            asm volatile("{fma.rn.f32x2 %0, %1, %2, %3;}" : "=l"(*(uint64_t*)&c) : "l"(*(uint64_t*)&a), "l"(*(uint64_t*)&scale), "l"(*(uint64_t*)&b));
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
        sizeof(globals::Q_tile) * globals::NUM_PIPELINES +   // 65536 B
        sizeof(globals::K_tile) * globals::PIPELINE_STAGES + // 49152 B
        sizeof(globals::V_tile) * globals::PIPELINE_STAGES + // 49152 B
        sizeof(globals::L_vec) * globals::NUM_PIPELINES <=   // 1024  B
        config::DYNAMIC_SHARED_MEMORY
    );
    globals::Q_tile (&Q_smem)[globals::NUM_PIPELINES]    = sm_allocator.allocate<globals::Q_tile, globals::NUM_PIPELINES>();
    globals::K_tile (&K_smem)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::K_tile, globals::PIPELINE_STAGES>();
    globals::V_tile (&V_smem)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::V_tile, globals::PIPELINE_STAGES>();
    globals::L_vec  (&L_smem)[globals::NUM_PIPELINES]    = sm_allocator.allocate<globals::L_vec, globals::NUM_PIPELINES>();
    globals::O_tile (&O_smem)[globals::NUM_PIPELINES]    = *reinterpret_cast<globals::O_tile(*)[globals::NUM_PIPELINES]>(&Q_smem[0]);

    // Allocate tensor memory
    tensor_allocator<1, config::CLUSTER_SIZE> tm_allocator {};
    using tm_fl_t = tt<float, globals::BLOCK_SIZE, globals::BLOCK_SIZE>;
    using tm_bf_t = tt<bf16, globals::BLOCK_SIZE, globals::BLOCK_SIZE>;

    // Set up mbarriers
    __shared__ semaphore Q_arrived;
    __shared__ semaphore Q_finished;
    __shared__ semaphore K_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore K_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore V_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore V_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore S_arrived[globals::NUM_PIPELINES];
    __shared__ semaphore P_arrived[globals::NUM_PIPELINES];
    __shared__ semaphore O_arrived[globals::NUM_PIPELINES];
    __shared__ semaphore O_scaled[globals::NUM_PIPELINES];
    if (threadIdx.x == 0) {
        init_semaphore(Q_arrived, 0, config::CLUSTER_SIZE);
        init_semaphore(Q_finished, 0, globals::NUM_PIPELINES);
        #pragma unroll
        for (int i = 0; i < globals::PIPELINE_STAGES; ++i) {
            init_semaphore(K_arrived[i], 0, config::CLUSTER_SIZE);
            init_semaphore(V_arrived[i], 0, config::CLUSTER_SIZE);
            init_semaphore(K_finished[i], 0, globals::NUM_PIPELINES);
            init_semaphore(V_finished[i], 0, globals::NUM_PIPELINES);
        }
        #pragma unroll
        for (int i = 0; i < globals::NUM_PIPELINES; ++i) {
            init_semaphore(S_arrived[i], 0, 1);
            init_semaphore(P_arrived[i], 0, config::CLUSTER_SIZE);
            init_semaphore(O_arrived[i], 0, 1);
            init_semaphore(O_scaled[i], 0, config::CLUSTER_SIZE);
        }
    }
    everyone::tma::cluster::sync();

    // Pipeline configuration
    const int cluster_id = clusterIdx().x;
    const int cta_id = cluster_ctarank();
    const int warp_id = ::warpid();
    const int lane_id = warp::laneid();
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF'0000;

    // Constants
    constexpr float SQRT_D_INV = 0.08838834764f; // 1 / sqrt(128)
    constexpr float NEG_SQRT_D = -11.313708499f; // -sqrt(128)
    constexpr float LOG2E = 1.44269504089f;

    if (warp_id < 8) { // 2 softmax groups
        const int pipeline_id = warpgroup::groupid();

        tm_fl_t S_tm = tm_allocator.allocate<tm_fl_t>(globals::BLOCK_SIZE * pipeline_id);
        tm_bf_t P_tm = tm_allocator.allocate<tm_bf_t>(globals::BLOCK_SIZE * pipeline_id);

        for (int task_idx = cluster_id; true; task_idx += gridDim.x / config::CLUSTER_SIZE) {
            globals::task_info task_info = get_task_info(G, task_idx);
            if (task_info.batch_idx == -1) break;

            col_vec<rt_fl<globals::BLOCK_SIZE / WARPGROUP_WARPS, globals::BLOCK_SIZE>> max_vec, norm_vec;
            warp::neg_infty(max_vec);
            warp::zero(norm_vec);

            for (int i = task_info.KV_block_start; i < task_info.KV_block_end; ++i) {
                tma::cluster::wait(S_arrived[pipeline_id], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES));
                update_phasebit<0>(phasebits, globals::PIPELINE_STAGES);
                warpgroup::arrive(K_finished[stage]);
                stage = (stage + 1) % globals::PIPELINE_STAGES;

                rt_fl<globals::BLOCK_SIZE / WARPGROUP_WARPS, globals::BLOCK_SIZE> SP_reg;
                warpgroup::load_async(SP_reg, S_tm);
                tensor_load_wait();

                // Perform softmax
                col_vec<rt_fl<globals::BLOCK_SIZE / WARPGROUP_WARPS, globals::BLOCK_SIZE>> max_vec_last_scaled, max_vec_scaled;
                // warp::mul(max_vec_last_scaled, max_vec, LOG2E * SQRT_D_INV);
                // warp::row_max(max_vec, SP_reg, max_vec);
                // warp::mul(max_vec_scaled, max_vec, -LOG2E * SQRT_D_INV);
                // warp::row_map<rescale_add>(SP_reg, SP_reg, max_vec_scaled);
                // warp::exp2(SP_reg, SP_reg);
                // warp::add(max_vec_last_scaled, max_vec_scaled, max_vec_last_scaled);
                // warp::exp2(max_vec_last_scaled, max_vec_last_scaled);
                // warp::mul(norm_vec, max_vec_last_scaled, norm_vec);
                // warp::row_sum(norm_vec, SP_reg, norm_vec);

                // Move P to TMEM
                // rt_bf<globals::BLOCK_SIZE / WARPGROUP_WARPS, globals::BLOCK_SIZE> P_bf_reg;
                // warp::copy(P_bf_reg, SP_reg);
                // warpgroup::store_async(P_tm, P_bf_reg);
                // tensor_store_wait();
                warpgroup::tma::cluster::arrive(P_arrived[pipeline_id], 0);
            }
        }
    } else if (warp_id < 12) { // O rescale group
        tm_fl_t O_tm[globals::NUM_PIPELINES] = {
            tm_allocator.allocate<tm_fl_t>(globals::BLOCK_SIZE * (globals::NUM_PIPELINES + 0)),
            tm_allocator.allocate<tm_fl_t>(globals::BLOCK_SIZE * (globals::NUM_PIPELINES + 1))
        };

        for (int task_idx = cluster_id; true; task_idx += gridDim.x / config::CLUSTER_SIZE) {
            globals::task_info task_info = get_task_info(G, task_idx);
            if (task_info.batch_idx == -1) break;

            for (int i = task_info.KV_block_start; i < task_info.KV_block_end; ++i) {
                #pragma unroll
                for (int pipeline_id = 0; pipeline_id < globals::NUM_PIPELINES; ++pipeline_id) {
                    if (i > task_info.KV_block_start) {
                        tma::cluster::wait(O_arrived[pipeline_id], get_phasebit<0>(phasebits, pipeline_id));
                        update_phasebit<0>(phasebits, pipeline_id);
                        warpgroup::arrive(V_finished[stage]);
                        stage = (stage + 1) % globals::PIPELINE_STAGES;
    
                        // rt_fl<globals::BLOCK_SIZE / WARPGROUP_WARPS, globals::VO_DIM> O_reg;
                        // warpgroup::load_async(O_reg, O_tm[pipeline_id]);
                        // // warp::mul_row(O_reg, O_reg, max_vec_last_scaled);
                        // warpgroup::store_async(O_tm[pipeline_id], O_reg);
                        // tensor_store_wait();
                        // warpgroup::sync(1 + pipeline_id);
                    }
                    warpgroup::tma::cluster::arrive(O_scaled[pipeline_id], 0);
                }
            }

            // Wait for the last PV MMA
            #pragma unroll
            for (int pipeline_id = 0; pipeline_id < globals::NUM_PIPELINES; ++pipeline_id) {
                tma::cluster::wait(O_arrived[pipeline_id], get_phasebit<0>(phasebits, pipeline_id));
                update_phasebit<0>(phasebits, pipeline_id);
                warpgroup::arrive(V_finished[stage]);
                stage = (stage + 1) % globals::PIPELINE_STAGES;

                rt_fl<globals::BLOCK_SIZE / WARPGROUP_WARPS, globals::VO_DIM> O_reg;
                warpgroup::load_async(O_reg, O_tm[pipeline_id]);
                // warp::div_row(O_reg, O_reg, norm_vec);
                warpgroup::store(O_smem[pipeline_id], O_reg);
                warpgroup::sync(pipeline_id + 1);
                warpgroup::tma::store_async(G.O, O_smem[pipeline_id], 
                                            {task_info.batch_idx, task_info.head_idx, 
                                            config::CLUSTER_SIZE * globals::NUM_PIPELINES * task_info.Q_block_idx + 
                                            globals::NUM_PIPELINES * cta_id + pipeline_id, 0});
                warpgroup::tma::store_async_read_wait();
                warpgroup::arrive(Q_finished);

                // warp::mul(max_vec, max_vec, SQRT_D_INV);
                // warp::log(norm_vec, norm_vec);
                // warp::add(norm_vec, norm_vec, max_vec);
                // warp::mul(norm_vec, norm_vec, NEG_SQRT_D);

                // warpgroup::store(L_smem[pipeline_id], norm_vec);
                warpgroup::sync(1 + pipeline_id);
                warpgroup::tma::store_async(G.L, L_smem[pipeline_id], 
                                    {task_info.batch_idx, task_info.head_idx, 0, 
                                    config::CLUSTER_SIZE * globals::NUM_PIPELINES * task_info.Q_block_idx + 
                                    globals::NUM_PIPELINES * cta_id + pipeline_id});
                warpgroup::tma::store_async_read_wait();
                warpgroup::sync(pipeline_id + 1);
            }
        }
    } else if (warp_id == 12 && lane_id == 0) { // Loader group
        for (int task_idx = cluster_id; true; task_idx += gridDim.x / config::CLUSTER_SIZE) {
            globals::task_info task_info = get_task_info(G, task_idx);
            if (task_info.batch_idx == -1) break;

            // Load Q
            wait(Q_finished, get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
            update_phasebit<1>(phasebits, globals::PIPELINE_STAGES);
            tma::cluster::expect_bytes(Q_arrived, sizeof(globals::Q_tile) * globals::NUM_PIPELINES, 0);
            #pragma unroll
            for (int pipeline_id = 0; pipeline_id < 2; pipeline_id++) {
                tma::cluster::load_async(Q_smem[pipeline_id], G.Q,
                                         {task_info.batch_idx, task_info.head_idx, 
                                         config::CLUSTER_SIZE * globals::NUM_PIPELINES * task_info.Q_block_idx + 
                                         globals::NUM_PIPELINES * cta_id + pipeline_id, 0},
                                         Q_arrived, (uint16_t)(1 << cta_id), 0);
            }

            // Stream K & V
            for (int i = task_info.KV_block_start; i < task_info.KV_block_end; ++i) {
                wait(K_finished[stage], get_phasebit<1>(phasebits, stage));
                tma::cluster::expect(K_arrived[stage], 0, K_smem[stage]);
                tma::cluster::load_async(K_smem[stage], G.K, 
                                         {task_info.batch_idx, task_info.head_idx, 2 * i + cta_id, 0},
                                         K_arrived[stage], (uint16_t)(1 << cta_id), 0);

                wait(V_finished[stage], get_phasebit<1>(phasebits, stage));
                tma::cluster::expect(V_arrived[stage], 0, V_smem[stage]);
                tma::cluster::load_async(V_smem[stage], G.V, 
                                         {task_info.batch_idx, task_info.head_idx, i, cta_id},
                                         V_arrived[stage], (uint16_t)(1 << cta_id), 0);

                update_phasebit<1>(phasebits, stage);
                stage = (stage + 1) % globals::PIPELINE_STAGES;
            }
        }
    } else if (warp_id == 13 && lane_id == 0 && cta_id == 0) { // MMA launcher group
        tm_fl_t S_tm[globals::NUM_PIPELINES] = {
            tm_allocator.allocate<tm_fl_t>(globals::BLOCK_SIZE * 0),
            tm_allocator.allocate<tm_fl_t>(globals::BLOCK_SIZE * 1)
        };
        tm_bf_t P_tm[globals::NUM_PIPELINES] = { // overlaps with S_tm
            tm_allocator.allocate<tm_bf_t>(globals::BLOCK_SIZE * 0),
            tm_allocator.allocate<tm_bf_t>(globals::BLOCK_SIZE * 1)
        };
        tm_fl_t O_tm[globals::NUM_PIPELINES] = {
            tm_allocator.allocate<tm_fl_t>(globals::BLOCK_SIZE * 2),
            tm_allocator.allocate<tm_fl_t>(globals::BLOCK_SIZE * 3)
        };

        for (int task_idx = cluster_id; true; task_idx += gridDim.x / config::CLUSTER_SIZE) {
            globals::task_info task_info = get_task_info(G, task_idx);
            if (task_info.batch_idx == -1) break;

            tma::cluster::wait(Q_arrived, get_phasebit<0>(phasebits, globals::PIPELINE_STAGES + globals::NUM_PIPELINES));
            update_phasebit<0>(phasebits, globals::PIPELINE_STAGES + globals::NUM_PIPELINES);

            for (int i = task_info.KV_block_start; i < task_info.KV_block_end; ++i) {
                // Launch S = QK^T
                tma::cluster::wait(K_arrived[stage], get_phasebit<0>(phasebits, stage));
                #pragma unroll
                for (int pipeline_id = 0; pipeline_id < globals::NUM_PIPELINES; ++pipeline_id) {
                    tma::cluster::wait(O_arrived[pipeline_id], get_phasebit<1>(phasebits, globals::PIPELINE_STAGES + pipeline_id));
                    mm2_ABt(S_tm[pipeline_id], Q_smem[pipeline_id], K_smem[stage], S_arrived[pipeline_id]);
                }

                // Launch O = PV
                tma::cluster::wait(V_arrived[stage], get_phasebit<0>(phasebits, stage));
                #pragma unroll
                for (int pipeline_id = 0; pipeline_id < globals::NUM_PIPELINES; ++pipeline_id) {
                    tma::cluster::wait(P_arrived[pipeline_id], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES + pipeline_id));
                    tma::cluster::wait(O_scaled[pipeline_id], get_phasebit<0>(phasebits, globals::PIPELINE_STAGES + pipeline_id));
                    {
                        constexpr int trans_b = 0;
                        constexpr int M = globals::BLOCK_SIZE * config::CLUSTER_SIZE;                    
                        constexpr int K = globals::BLOCK_SIZE;
                        constexpr int N = globals::VO_DIM;                    
                        constexpr int red_dim = 16; // fixed for BF16
                        using B_type = globals::V_tile;

                        uint32_t idesc = kittens::detail::tcgen05::instruction_descriptor<float, bf16, M, N, 0, trans_b, false>();
                        st_descriptor<ducks::st_descriptor::detail::get_st<B_type>, trans_b> b_desc(V_smem[stage]);
                        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");

                        if (i == task_info.KV_block_start)
                            kittens::detail::tcgen05::template tt_st<bf16, 0, config::CLUSTER_SIZE>(
                                O_tm[pipeline_id].addr,
                                P_tm[pipeline_id].template chunk_addr<0>(0),
                                b_desc.chunk_descriptor(0),
                                idesc
                            );
                        else
                        kittens::detail::tcgen05::template tt_st<bf16, 1, config::CLUSTER_SIZE>(
                            O_tm[pipeline_id].addr,
                            P_tm[pipeline_id].template chunk_addr<0>(0),
                            b_desc.chunk_descriptor(0),
                            idesc
                        );
                        #pragma unroll
                        for(int i = 1; i < K / red_dim; i++) {
                            kittens::detail::tcgen05::template tt_st<bf16, 1, config::CLUSTER_SIZE>(
                                O_tm[pipeline_id].addr,
                                P_tm[pipeline_id].template chunk_addr<0>(i),
                                b_desc.chunk_descriptor(i),
                                idesc
                            );
                        }
                        kittens::detail::tcgen05::commit<config::CLUSTER_SIZE>(O_arrived[pipeline_id]);
                    }
                }

                // Step
                #pragma unroll
                for (int pipeline_id = 0; pipeline_id < globals::NUM_PIPELINES; ++pipeline_id) {
                    update_phasebit<0>(phasebits, globals::PIPELINE_STAGES + pipeline_id);
                    update_phasebit<1>(phasebits, globals::PIPELINE_STAGES + pipeline_id);
                }
                update_phasebit<0>(phasebits, stage);
                stage = (stage + 1) % globals::PIPELINE_STAGES;
            }
        }
    } else if (warp_id == 14 && lane_id == 0) {

    } else if (warp_id == 15 && lane_id == 0) {

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
