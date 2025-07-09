#include "multi-gpu.cuh"

constexpr int NUM_DEVICES = 8;

__global__ void all_reduce_float32_sum(float *data, int nelem);

int run(int verbose = 0) {

    assert(NUM_DEVICES > 1);

    /*
        Set up MC
    */
    CUCHECK(cuInit(0));

    size_t granularity;
    size_t size;

    CUmemGenericAllocationHandle mcHandle;
    CUmulticastObjectProp mcProp = {};
    mcProp.numDevices = NUM_DEVICES;
    mcProp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; // single node
    mcProp.flags = 0; // SBZ

    granularity = 0;
    CUCHECK(cuMulticastGetGranularity(&granularity, &mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED)); 
    size = granularity;
    mcProp.size = size;
    CUCHECK(cuMulticastCreate(&mcHandle, &mcProp));

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUdevice dev;
        CUCHECK(cuDeviceGet(&dev, dev_idx));
        CUCHECK(cuMulticastAddDevice(mcHandle, dev));
    }

    CUmemGenericAllocationHandle memHandles[NUM_DEVICES];
    CUdeviceptr mcPtrs[NUM_DEVICES];
    CUdeviceptr memPtrs[NUM_DEVICES];

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));

        CUmemAllocationProp memProp = {};
        memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        memProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        memProp.location.id = dev_idx;
        memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

        size_t mem_granularity = 0;
        CUCHECK(cuMemGetAllocationGranularity(&mem_granularity, &memProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        if (size % mem_granularity != 0) {
            fprintf(stderr, "Size must be a multiple of mem granularity\n");
            exit(1);
        }

        // Allocate physical memory on the device
        CUCHECK(cuMemCreate(&memHandles[dev_idx], size, &memProp, 0));

        // Bind the physical memory to the multicast handle
        CUCHECK(cuMulticastBindMem(mcHandle, /*mcOffset=*/0, memHandles[dev_idx], /*memOffset=*/0, size, 0));
        
        // Allocate virtual address space for the handles
        CUCHECK(cuMemAddressReserve(&mcPtrs[dev_idx], size, granularity, 0, 0));
        CUCHECK(cuMemAddressReserve(&memPtrs[dev_idx], size, granularity, 0, 0));

        // Bind VAs to the multicast handle and physical memory
        CUCHECK(cuMemMap(mcPtrs[dev_idx], size, 0, mcHandle, 0));
        CUCHECK(cuMemMap(memPtrs[dev_idx], size, 0, memHandles[dev_idx], 0));

        // Remember to set access AFTER mapping
        CUmemAccessDesc desc[1];
        desc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        desc[0].location.id = dev_idx;
        desc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        CUCHECK(cuMemSetAccess(mcPtrs[dev_idx], size, desc, 1));
        CUCHECK(cuMemSetAccess(memPtrs[dev_idx], size, desc, 1));
    }

    /*
        Setup the data
    */
    assert(size % sizeof(float) == 0);

    int nelem = size / sizeof(float);
    float **host_mats = (float**)malloc(NUM_DEVICES * sizeof(float*));
    srand(static_cast<unsigned int>(time(nullptr))); // random seed

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        host_mats[dev_idx] = (float*)malloc(size);
        if (verbose) printf("Device %d: ", dev_idx);
        for (int i = 0; i < nelem; ++i) {
            host_mats[dev_idx][i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            if (i < 10)
                if (verbose) printf("%f ", host_mats[dev_idx][i]);
        }
        if (verbose) printf("... (%d elements)\n", nelem);
    }

    float *expected = (float*)malloc(size);
    if (verbose) printf("Expected: ");
    for (int i = 0; i < nelem; ++i) {
        expected[i] = 0.0f;
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
            expected[i] += host_mats[dev_idx][i];
        }
        if (i < 10)
            if (verbose) printf("%f ", expected[i]);
    }
    if (verbose) printf("... (%d elements)\n", nelem);

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaMemcpy((void*)memPtrs[dev_idx], host_mats[dev_idx], size, cudaMemcpyHostToDevice));
    }

    /*
        Perform the reduction
    */
    CUDACHECK(cudaSetDevice(0));
    all_reduce_float32_sum<<<(nelem / 4 + 255) / 256, 256>>>((float*)mcPtrs[0], nelem);
    CUDACHECK(cudaDeviceSynchronize());

    /* 
        Bring back data
    */
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaMemcpy(host_mats[dev_idx], (void*)memPtrs[dev_idx], size, cudaMemcpyDeviceToHost));
    }

    /*
        Verify the results
    */
    float TOL = 1e-5;
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        if (verbose) printf("Device %d: ", dev_idx);
        for (int i = 0; i < nelem; ++i) {
            if (i < 10)
                if (verbose) printf("%f ", host_mats[dev_idx][i]);
            if (fabs(expected[i] - host_mats[dev_idx][i]) > TOL) {
                fprintf(stderr, "Mismatch at device %d, index %d: expected %f, got %f\n", dev_idx, i, expected[i], host_mats[dev_idx][i]);
                exit(1);
            }
        }
        if (verbose) printf("... (%d elements)\n", nelem);
    }

    /*
        Cleanup and exit
    */

    // Free resources
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));

        // Always free the memory in this order
        CUCHECK(cuMemUnmap(mcPtrs[dev_idx], size));
        CUCHECK(cuMemUnmap(memPtrs[dev_idx], size));
        CUCHECK(cuMemAddressFree(mcPtrs[dev_idx], size));
        CUCHECK(cuMemAddressFree(memPtrs[dev_idx], size));
        CUCHECK(cuMemRelease(memHandles[dev_idx]));
    }

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        free(host_mats[dev_idx]);
    }

    free(host_mats);
    free(expected);

    if (verbose) printf("Done\n");
    return 0;
}

int main() {
    run(1); // warmup

    for (int i = 2; i <= 130; ++i) {
        printf("Run %d\n", i);
        run();
    }
}

__global__ void all_reduce_float32_sum(float *data, int nelem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 4 < nelem) {
        volatile float4 val;
        float *ptr = data + idx * 4;
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w) : "l"(ptr) : "memory"); // relaxed vs weak?
        asm volatile("fence.proxy.alias;" ::: "memory"); // force memory ordering
        // *ptr = val;
        asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w) : "memory"); // curious: what if I don't use asm here and just use plain assignment?
    }
}
