/*
    Goal: implement fast single-node all2all collective communication

    Benchmarks (N: 131072, H: 128, D: 128, 8 GPUs, rank 0 only, Unidirectional NVL bandwidth):

        NCCL: 319.01 GB/s
        Naive 1-stage persistent-grid: 642.21 GB/s
        Naive 2-stage persistent-grid: 630.59 GB/s
        Naive 4-stage persistent-grid: 628.76 GB/s
        Naive 8-stage persistent-grid: 632.25 GB/s
        Naive 16-stage persistent-grid: 627.79 GB/s
        Stage-parllel 1-stage persistent-grid: 645.83 GB/s
        Stage-parllel 2-stage persistent-grid: 615.85 GB/s
        Stage-parllel 4-stage persistent-grid: 641.20 GB/s
        Stage-parllel 8-stage persistent-grid: 627.09 GB/s
        Stage-parllel 16-stage persistent-grid: 557.56 GB/s
        Stage-parllel 32-stage persistent-grid: 496.32 GB/s
        Non-persistent-grid: 693.37 GB/s

    Conclusion:
        - It always seems to be the case that for memory-bound workloads, non-persistent-grid is faster.
*/

#include <torch/csrc/utils/pybind.h>

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 1;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int ROW_BLOCK_SIZE = 16;
    static constexpr int COL_BLOCK_SIZE = 128;
    // static constexpr int PIPELINE_STAGES = 16;

    using tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1, tma::descriptor<tile, 2, false>>, NUM_DEVICES, false>;

    parallel_layout src;
    parallel_layout dst;
    const int dev_idx;

    __host__ inline dim3 grid() const { 
        return dim3(src.batch() *
                    src.depth() * 
                    (src.rows() / globals::ROW_BLOCK_SIZE) *
                    (src.cols() / globals::COL_BLOCK_SIZE)); 
    }
    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(tile) + 1024);
    }
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
    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::tile &tile = allocator.allocate<globals::tile>();

    // Retrieve indices
    int4 indices = get_indices(G, blockIdx.x);

    // Set up mbarrier
    __shared__ semaphore inputs_arrived;
    init_semaphore(inputs_arrived, 0, 1);

    // Initiate the load
    tma::expect_bytes(inputs_arrived, sizeof(globals::tile));
    asm volatile("{cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3, %4, %5}], [%6];}"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&tile[0]))), 
            "l"(G.src[G.dev_idx].template get_tma<globals::tile, 2>()), 
            "r"(indices.w * globals::COL_BLOCK_SIZE), "r"(indices.z * globals::ROW_BLOCK_SIZE), "r"(indices.y), "r"(indices.x),
            "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&inputs_arrived)))
        : "memory");

    // Calculate scatter and gather blocks per device
    const int scatter_blocks_per_dev = G.src.rows() / globals::ROW_BLOCK_SIZE / globals::NUM_DEVICES;
    const int gather_blocks_per_dev = G.dst.depth() / globals::NUM_DEVICES; // important to use dst not src

    // Decide which device to send to with scatter axis
    int dst_dev_idx = reinterpret_cast<int *>(&indices)[SCATTER_AXIS] / scatter_blocks_per_dev;

    // Decide which location in the scatter axis to send to with destination device index
    int scatter_axis_base = scatter_blocks_per_dev * dst_dev_idx; // will be subtracted

    // Decide which location in the gather axis to send to with current device index
    int gather_axis_base = gather_blocks_per_dev * G.dev_idx; // will be added

    // Wait for inputs to be arrived
    wait(inputs_arrived, 0);
    asm volatile("{fence.proxy.async.shared::cta;}" ::: "memory"); // make writes to smem visible

    // Store data to destination device
    asm volatile("{cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];}"
        :: "l"(G.dst[dst_dev_idx].template get_tma<globals::tile, 2>()),
            "r"(indices.w * globals::COL_BLOCK_SIZE), "r"((indices.z - scatter_axis_base) * globals::ROW_BLOCK_SIZE), "r"(indices.y + gather_axis_base), "r"(indices.x), // TODO
            "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&tile[0])))
        : "memory");
}

void entrypoint(
    const at::Tensor &dst,
    const KittensIPCPointerSet &dst_ipc_ptrs,
    const at::Tensor &src,
    const KittensIPCPointerSet &src_ipc_ptrs,
    KittensBroker &broker
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
