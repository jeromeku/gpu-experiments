#include "multi-gpu.cuh"

__global__ void writeKernel(float *ptr, int nelem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nelem)
        ptr[idx] = idx;
}

__global__ void readKernel(float *ptr, int nelem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 100)
        printf("ptr[%d] = %f ", idx, ptr[idx]);
}

int main() {

    // cuInit must be called before any Driver API calls, and argument SBZ
    CUCHECK(cuInit(0));

    // A generic allocation handle representing a multicast object
    CUmemGenericAllocationHandle mcHandle;

    // Describe the allocation handle
    CUmulticastObjectProp mcProp;
    mcProp.flags = 0; // SBZ; field for extensions in the future
    /*
    CU_MEM_HANDLE_TYPE_NONE = 0x0
        Does not allow any export mechanism. >
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1
        Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int)
    CU_MEM_HANDLE_TYPE_WIN32 = 0x2
        Allows a Win32 NT handle to be used for exporting. (HANDLE)
    CU_MEM_HANDLE_TYPE_WIN32_KMT = 0x4
        Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE)
    CU_MEM_HANDLE_TYPE_FABRIC = 0x8
        Allows a fabric handle to be used for exporting. (CUmemFabricHandle)
    */
    mcProp.handleTypes = 0; // Bitmask of CUmemAllocationHandleTypes
    mcProp.numDevices = 2; // number of participating devices

    // Minimum size 64MB??
    // There are restriction on sizes
    size_t granularity;
    CUCHECK(cuMulticastGetGranularity(&granularity, &mcProp, CU_MULTICAST_GRANULARITY_MINIMUM));
    printf("Minimum multicast granularity: %luMiB\n", granularity / 1024 / 1024);
    CUCHECK(cuMulticastGetGranularity(&granularity, &mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    printf("Recommended multicast granularity: %luMiB\n", granularity / 1024 / 1024);
    
    // Use the recommended value (2MiB on H100 machines apparently)
    size_t size = 2 * 1024 * 1024;
    mcProp.size = size; // maximum amount of memory that can be bound per device
    size_t nelem = size / sizeof(float);
    
    // Allocate the allocation handle as described by mcProp
    CUCHECK(cuMulticastCreate(&mcHandle, &mcProp));

    // Add devices to this mc obj
    // Devices must be added before memory binding
    CUCHECK(cuMulticastAddDevice(mcHandle, /*dev=*/0));
    CUCHECK(cuMulticastAddDevice(mcHandle, /*dev=*/1));

    // Memory binding: 
    //   1. Allocate memory thru cuMemCreate + cuMemMap (+ cuMemSetAccess)
    //      or cudaMalloc(Async).
    //   2. Bind the virtual address of ^ with cuMulticastBindMem
    //      or cuMulticastBindAddr
    float *dev0ptr;
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMalloc(&dev0ptr, size));
    CUCHECK(cuMulticastBindAddr(
        mcHandle, // the mc handle
        0, // mcOffset; the offset in mc VA range; mcOffset + size cannot be larger than size of mc object
        (CUdeviceptr)dev0ptr, // memptr; the pointer to VA to bind; must be on one of the added devices
        size, // size; bind will be on dev0ptr[0:size]; cannot be larger than allocated size
        0 // flags; SBZ
    )); // mcOffset, memptr, size must be multiple of granularity
    // Or you can directly map CUmemGenericAllocationHandle (from cuMemCreate)
    // to mc Object with cuMulticastBindMem

    // Let's also bind an area from device 1
    float *dev1ptr;
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMalloc(&dev1ptr, size));
    CUCHECK(cuMulticastBindAddr(mcHandle, 0, (CUdeviceptr)dev1ptr, size, 0));


    // The fun parts
    // 
    // 1. Let's copy some data to dev0, and see how dev1 reads it
    // printf("\nExp 1:\n\n");
    float *hostptr0 = (float*)malloc(size);
    float *hostptr1 = (float*)malloc(size);
    // for (int i = 0; i < nelem; ++i) {
    //     hostptr0[i] = (float)i;
    //     if (i < 100)
    //         printf("%f ", hostptr0[i]);
    // }   printf("\n");
    // CUDACHECK(cudaSetDevice(0));
    // CUDACHECK(cudaMemcpy(dev0ptr, hostptr0, nelem, cudaMemcpyHostToDevice));
    // CUDACHECK(cudaDeviceSynchronize());
    // CUDACHECK(cudaSetDevice(1));
    // CUDACHECK(cudaMemcpy(hostptr1, dev1ptr, nelem, cudaMemcpyDeviceToHost));
    // // So we didn't explicitly write anything to device 1; if it matches it must be multicast stuff happening
    // for (int i = 0; i < nelem; ++i) {
    //     if (i < 100)
    //         printf("%f ", hostptr1[i]);
    //     // assert(hostptr0[i] == hostptr1[i]);
    // }   printf("\n");
    // Okay so apparently this doesn't work; the values didn't get broadcasted at all.


    // 2. Instead, let's try to write a kernel that writes to these addresses.
    // printf("\n\n\nExp 2:\n\n");
    // CUDACHECK(cudaSetDevice(0));
    // writeKernel<<<nelem / 256, 256>>>(dev0ptr, nelem);
    // CUDACHECK(cudaDeviceSynchronize());
    // CUDACHECK(cudaSetDevice(1));
    // readKernel<<<nelem / 256, 256>>>(dev1ptr, nelem);
    // CUDACHECK(cudaDeviceSynchronize());
    // printf("\n");
    // Okay this doesn't work either. Should I try binding a new VA address to multicast handle?


    // 3. Let's bind virtual addresses to the mc handles
    CUDACHECK(cudaSetDevice(0));
    CUdeviceptr vaPtr0;
    CUCHECK(cuMemAddressReserve(&vaPtr0, size, granularity, 0, 0));
    CUCHECK(cuMemMap(vaPtr0, size, 0 /*SBZ*/, mcHandle, 0 /*SBZ*/));
    CUmemAccessDesc desc0[1];
    desc0[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    desc0[0].location.id = 0; /* device ID */
    desc0[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    CUCHECK(cuMemSetAccess(vaPtr0, size, desc0, 1));

    CUDACHECK(cudaSetDevice(1));
    CUdeviceptr vaPtr1;
    CUCHECK(cuMemAddressReserve(&vaPtr1, size, granularity, 0, 0));
    CUCHECK(cuMemMap(vaPtr1, size, 0 /*SBZ*/, mcHandle, 0 /*SBZ*/));
    CUmemAccessDesc desc1[1];
    desc1[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    desc1[0].location.id = 1; /* device ID */
    desc1[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    CUCHECK(cuMemSetAccess(vaPtr1, size, desc1, 1));

    printf("\n\n\nExp 3:\n\n");
    CUDACHECK(cudaSetDevice(0));
    writeKernel<<<nelem / 256, 256>>>((float*)vaPtr0, nelem);
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaSetDevice(1));
    readKernel<<<nelem / 256, 256>>>((float*)vaPtr1, nelem);
    CUDACHECK(cudaDeviceSynchronize());
    printf("\n\n");
    // Works!!!!!!


    // 4. Now I'm curious; what about memcpy? Will it work?
    printf("\nExp 4:\n\n");
    for (int i = 0; i < nelem; ++i) {
        hostptr0[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (i < 100)
            printf("%f ", hostptr0[i]);
    }   printf("\n");
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMemcpy((float*)vaPtr0, hostptr0, nelem, cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMemcpy(hostptr1, (float*)vaPtr1, nelem, cudaMemcpyDeviceToHost));
    // So we didn't explicitly write anything to device 1; if it matches it must be multicast stuff happening
    for (int i = 0; i < nelem; ++i) {
        if (i < 100)
            printf("%f ", hostptr1[i]);
        // assert(hostptr0[i] == hostptr1[i]);
    }   printf("\n");
    // Works!!!!!!!!!!!!!!!



    // Memory unbinding: use cuMulticastUnbind and do the usual cleaning
    CUCHECK(cuMulticastUnbind(
        mcHandle, 
        0,          // Device that hosts the memory allocation
        0,          // mcOffset; must match the value used in cuMulticastBindX
        size // size to unbind; must match the value used in cuMulticastBindX
    ));
    CUCHECK(cuMulticastUnbind(mcHandle, 1, 0, size));
    // CUCHECK(cuMemAddressFree(vaPtr0, size)); // does not work...why?
    // CUCHECK(cuMemAddressFree(vaPtr1, size)); // does not work...why?
    CUDACHECK(cudaFree(dev0ptr));
    CUDACHECK(cudaFree(dev1ptr));
    free(hostptr0);
    free(hostptr1);

    printf("Done\n");    
    return 0;
}