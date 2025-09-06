#include <vector>

#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/utils/pybind.h>

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"

at::Tensor create_custom_tensor(
    std::vector<int64_t> &shape,
    at::ScalarType dtype,
    int device_id
) {
    c10::cuda::CUDAGuard device_guard(device_id);

    TORCH_CHECK(!shape.empty(), "Shape must be non-empty");
    TORCH_CHECK(shape.size() <= 4, "Shape must have at most 4 dimensions for TK");
    size_t bytes = c10::elementSize(dtype);
    for (auto dim : shape) {
        TORCH_CHECK(dim > 0, "Size dimensions must be positive");
        bytes *= static_cast<size_t>(dim);
    }

    void* raw_ptr = nullptr;
    CUDACHECK(cudaMalloc(&raw_ptr, bytes));

    auto deleter = [device_id](void* p) {
        if (!p) return;
        c10::cuda::CUDAGuard device_guard(device_id);
        auto stream = c10::cuda::getCurrentCUDAStream().stream();
        CUDACHECK(cudaFreeAsync(p, stream));
    };

    at::TensorOptions options = at::TensorOptions()
        .dtype(dtype)
        .device(at::kCUDA, device_id);

    return at::from_blob(raw_ptr, shape, std::move(deleter), options);
}

PYBIND11_MODULE(_C, m){
    m.def("create_custom_tensor", &create_custom_tensor);
}
