#include "multi-gpu.cuh"

using bf16 = __nv_bfloat16;

__global__ void writeKernel(bf16 *ptr, int nelem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nelem)
        ptr[idx] = __int2bfloat16_rd(idx);
}

__global__ void readKernel(bf16 *ptr, int nelem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 100)
        printf("ptr[%d] = %f ", idx, __bfloat162float(ptr[idx]));
}

int main() {

    // cuInit must be called before any Driver API calls, and argument SBZ
    CUCHECK(cuInit(0));

    // A generic allocation handle representing a multicast object
    CUmemGenericAllocationHandle mcHandle;

    // Describe the allocation handle
    CUmulticastObjectProp mcProp;
    mcProp.flags = 0;
    mcProp.handleTypes = 0;
    mcProp.numDevices = 2;

    size_t granularity;
    CUCHECK(cuMulticastGetGranularity(&granularity, &mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    
    size_t size = 64 * 1024 * 1024;
    mcProp.size = size;
    size_t nelem = size / sizeof(bf16);
    
    CUCHECK(cuMulticastCreate(&mcHandle, &mcProp));
    CUCHECK(cuMulticastAddDevice(mcHandle, /*dev=*/0));
    CUCHECK(cuMulticastAddDevice(mcHandle, /*dev=*/1));

    bf16 *dev0ptr;
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMalloc(&dev0ptr, size));
    CUCHECK(cuMulticastBindAddr(mcHandle, 0, (CUdeviceptr)dev0ptr, size, 0));

    bf16 *dev1ptr;
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMalloc(&dev1ptr, size));
    CUCHECK(cuMulticastBindAddr(mcHandle, 0, (CUdeviceptr)dev1ptr, size, 0));

    CUDACHECK(cudaSetDevice(0));
    CUdeviceptr vaPtr0;
    CUCHECK(cuMemAddressReserve(&vaPtr0, size, granularity, 0, 0));
    CUCHECK(cuMemMap(vaPtr0, size, 0, mcHandle, 0));
    CUmemAccessDesc desc0[1];
    desc0[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    desc0[0].location.id = 0; /* device ID */
    desc0[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    CUCHECK(cuMemSetAccess(vaPtr0, size, desc0, 1));

    CUDACHECK(cudaSetDevice(1));
    CUdeviceptr vaPtr1;
    CUCHECK(cuMemAddressReserve(&vaPtr1, size, granularity, 0, 0));
    CUCHECK(cuMemMap(vaPtr1, size, 0, mcHandle, 0));
    CUmemAccessDesc desc1[1];
    desc1[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    desc1[0].location.id = 1; /* device ID */
    desc1[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    CUCHECK(cuMemSetAccess(vaPtr1, size, desc1, 1));

    CUDACHECK(cudaSetDevice(0));
    writeKernel<<<nelem / 256, 256>>>((bf16*)vaPtr0, nelem);
    CUDACHECK(cudaDeviceSynchronize());

    CUDACHECK(cudaSetDevice(1));
    readKernel<<<nelem / 256, 256>>>((bf16*)dev1ptr, nelem);
    CUDACHECK(cudaDeviceSynchronize());

    printf("\nDone\n");    
    return 0;
}