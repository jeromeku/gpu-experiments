/*
    Goal: implement fast single-node all2all collective communication
*/

#include <torch/extension.h>

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 148; // TODO: vary this
    static constexpr int STATIC_SHARED_MEMORY = 128;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int NUM_WARPS = 2;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int ROW_BLOCK_SIZE = 16;
    static constexpr int COL_BLOCK_SIZE = 128;
    static constexpr int PIPELINE_STAGES = 16;

    using tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1, tma::descriptor<tile, 2, false>>, NUM_DEVICES, false>;

    parallel_layout src;
    parallel_layout dst;
    const int dev_idx;
};

__device__ inline int4 get_indices(const globals &G, int task_id) {
    int4 indices;

    indices.x = task_id / (G.src.depth() * (G.src.rows() / globals::ROW_BLOCK_SIZE) * (G.src.cols() / globals::COL_BLOCK_SIZE));
    task_id -= indices.x * (G.src.depth() * (G.src.rows() / globals::ROW_BLOCK_SIZE) * (G.src.cols() / globals::COL_BLOCK_SIZE));
    indices.y = task_id / ((G.src.rows() / globals::ROW_BLOCK_SIZE) * (G.src.cols() / globals::COL_BLOCK_SIZE));
    task_id -= indices.y * ((G.src.rows() / globals::ROW_BLOCK_SIZE) * (G.src.cols() / globals::COL_BLOCK_SIZE));
    indices.z = task_id / (G.src.cols() / globals::COL_BLOCK_SIZE);
    task_id -= indices.z * (G.src.cols() / globals::COL_BLOCK_SIZE);
    indices.w = task_id;

    if (indices.x >= G.src.batch())
        return { -1, -1, -1, -1 };
    else
        return indices;
}

template <int SCATTER_AXIS = 2, int GATHER_AXIS = 1>
__device__ inline void kernel(const globals &G) {
    // Shared memory declaration
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    // Warp configuration
    const int warp_id = kittens::warpid();
    const int lane_id = warp::laneid();

    // Allocate shared memory
    static_assert(sizeof(globals::tile) * globals::PIPELINE_STAGES <= config::DYNAMIC_SHARED_MEMORY);
    globals::tile (&tiles)[globals::PIPELINE_STAGES] = allocator.allocate<globals::tile, globals::PIPELINE_STAGES>();

    // Set up mbarriers
    __shared__ semaphore inputs_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[globals::PIPELINE_STAGES];
    if (threadIdx.x == 0) {
        for (int i = 0; i < globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
    }
    __syncthreads();

    // Declare stage and phasebits for semaphore waits
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    // Main divergence
    if (warp_id == 0 && lane_id < globals::PIPELINE_STAGES) {
        for (int task_id = blockIdx.x; true; task_id += gridDim.x) {
            int4 indices = get_indices(G, task_id);
            if (indices.x == -1) break;

            if (lane_id == stage) {
                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                update_phasebit<1>(phasebits, stage);
    
                tma::expect_bytes(inputs_arrived[stage], sizeof(globals::tile));
                asm volatile("{cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3, %4, %5}], [%6];}"
                    :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&tiles[stage][0]))), 
                       "l"(G.src[G.dev_idx].template get_tma<globals::tile, 2>()), 
                       "r"(indices.w * globals::COL_BLOCK_SIZE), "r"(indices.z * globals::ROW_BLOCK_SIZE), "r"(indices.y), "r"(indices.x),
                       "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&inputs_arrived[stage])))
                    : "memory");
            }

            stage = (stage + 1) % globals::PIPELINE_STAGES;
        }
    } else if (warp_id == 1 && lane_id < globals::PIPELINE_STAGES) {
        // Scatter/gather configuration
        // TODO: branch based on actual axis
        const int scatter_blocks_per_dev = G.src.rows() / globals::ROW_BLOCK_SIZE / globals::NUM_DEVICES;
        const int gather_blocks_per_dev = G.dst.depth() / globals::NUM_DEVICES; // important to use dst not src

        for (int task_id = blockIdx.x; true; task_id += gridDim.x) {
            int4 indices = get_indices(G, task_id);
            if (indices.x == -1) break;

            // Decide which device to send to with scatter axis
            int dst_dev_idx = reinterpret_cast<int *>(&indices)[SCATTER_AXIS] / scatter_blocks_per_dev;

            // Decide which location in the scatter axis to send to with destination device index
            int scatter_axis_base = scatter_blocks_per_dev * dst_dev_idx; // will be subtracted

            // Decide which location in the gather axis to send to with current device index
            int gather_axis_base = gather_blocks_per_dev * G.dev_idx; // will be added

            if (lane_id == stage) {
                wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                update_phasebit<0>(phasebits, stage);

                asm volatile("{fence.proxy.async.shared::cta;}" ::: "memory"); // make writes to smem visible
                asm volatile("{cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];}"
                    :: "l"(G.dst[dst_dev_idx].template get_tma<globals::tile, 2>()),
                       "r"(indices.w * globals::COL_BLOCK_SIZE), "r"((indices.z - scatter_axis_base) * globals::ROW_BLOCK_SIZE), "r"(indices.y + gather_axis_base), "r"(indices.x), // TODO
                       "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&tiles[stage][0])))
                    : "memory");
                asm volatile("{cp.async.bulk.commit_group;}");
                asm volatile("{cp.async.bulk.wait_group.read %0;}" :: "n"(0) : "memory");

                arrive(inputs_finished[stage]);
            }

            stage = (stage + 1) % globals::PIPELINE_STAGES;
        }
    }
}

void entrypoint(
    const at::Tensor &dst,
    const KittensIPCPointerSet &dst_ipc_ptrs,
    const at::Tensor &src,
    const KittensIPCPointerSet &src_ipc_ptrs,
    const KittensBroker &broker
) {
    kittens::py::device_check(dst, src);

    // Instantiate globals
    globals G {
        .src = kittens::py::tensor_to_pgl<typename globals::parallel_layout>(src, src_ipc_ptrs, broker),
        .dst = kittens::py::tensor_to_pgl<typename globals::parallel_layout>(dst, dst_ipc_ptrs, broker),
        .dev_idx = broker.local_rank_
    };

    // Run kernel
    kittens::py::launch_kernel<config, globals, kernel<2, 1>>(G);
}

PYBIND11_MODULE(_C, m) {
    pybind11::class_<KittensBroker>(m, "KittensBroker")
        .def(pybind11::init<int,int>())
        .def("gather_ipc_ptrs",
            pybind11::overload_cast<const at::Tensor&>(&KittensBroker::gather_ipc_ptrs),
            pybind11::call_guard<pybind11::gil_scoped_release>(),
            pybind11::return_value_policy::move);
    pybind11::class_<KittensIPCPointerSet>(m, "KittensIPCPointerSet")
        .def(pybind11::init<>());
    m.def("all2all", &entrypoint);
}
