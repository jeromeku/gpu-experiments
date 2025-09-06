#include <vector>

#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/utils/pybind.h>

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"

at::Tensor create_shareable_cuda_tensor(
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

    // Create intra-node-shareable memory allocation.
    // This makes the handle shareable with cuMemExportToShareableHandle/cuMemImportFromShareableHandle
    /*
    TODO
    later must query :
    int deviceSupportsIpcHandle;
#if defined(__linux__)
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
#else
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, device));
#endif
    */
    CUmemAllocationProp prop = {};
    prop.location.id = device_id;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; // intra-node shared
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;

    std::cout << "Bytes: " << bytes << std::endl;
    // Round up to the recommended granularity (usually 2MB)
    size_t granularity  = 0;
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    bytes = (bytes + granularity - 1) / granularity * granularity; // round-up

    std::cout << "Granularity: " << granularity << std::endl;
    std::cout << "Bytes: " << bytes << std::endl;

    // Allocate physical memory
    CUmemGenericAllocationHandle handle;
    CUCHECK(cuMemCreate(&handle, bytes, &prop, 0));

    // Allocate virtual address
    CUdeviceptr raw_ptr;
    CUCHECK(cuMemAddressReserve(&raw_ptr, bytes, 0, 0, 0));

    // Map physical memory & virtual address
    CUCHECK(cuMemMap(raw_ptr, bytes, 0, handle, 0));

    // Set access (TODO: set access for all devices)
    CUmemAccessDesc desc = {};
    desc.location.id = device_id;
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess(raw_ptr, bytes, &desc, 1));

    auto deleter = [device_id, handle, raw_ptr, bytes](void* p) {
        if (!p) return;
        std::cout << "Deleting tensor" << std::endl;
        c10::cuda::CUDAGuard device_guard(device_id);
        auto stream = c10::cuda::getCurrentCUDAStream().stream();
        CUDACHECK(cudaStreamSynchronize(stream));
        CUCHECK(cuMemUnmap(raw_ptr, bytes)); 
        CUCHECK(cuMemAddressFree(raw_ptr, bytes));
        CUCHECK(cuMemRelease(handle));
    };

    at::TensorOptions options = at::TensorOptions()
        .dtype(dtype)
        .device(at::kCUDA, device_id);

    return at::from_blob(reinterpret_cast<void *>(raw_ptr), shape, std::move(deleter), options);
}

PYBIND11_MODULE(_C, m){
    m.def("create_custom_tensor", &create_custom_tensor);
}
