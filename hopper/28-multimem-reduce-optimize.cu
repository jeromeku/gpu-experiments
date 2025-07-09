

// this did not work out well
// no speed was gained from using for loop inside kernel

#include "multi-gpu.cuh"

constexpr int NUM_DEVICES = 8;
constexpr int SIZE = 1024 * 1024 * 1024;

constexpr int WARPSIZE = 32;
constexpr int STRIDE = 512;
constexpr int NUM_ITERS = 4;

__global__ void all_reduce_float32_sum(float *data, int nelem);
__global__ void all_reduce_float32_sum_coalesced(float *data, int nelem);
__global__ void stupid_load_store(float *dst, float *src, int nelem);

int main() {

    assert(NUM_DEVICES > 1);
    assert(SIZE >= 1024 * 1024 && SIZE % (1024 * 1024) == 0);

    // cuInit must be called before any Driver API calls, and argument SBZ
    CUCHECK(cuInit(0));

    /*
        1. Query Multicast Support
    */
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUdevice dev;
        CUCHECK(cuDeviceGet(&dev, dev_idx));

        int deviceSupportsMultiCast;
        CUCHECK(cuDeviceGetAttribute(
            &deviceSupportsMultiCast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));

        if (!deviceSupportsMultiCast) {
            fprintf(stderr, "Device %d does not support Multicast Objects\n", dev_idx);
            exit(1);
        }
    }

    /*
        2. Create a Multicast Handle with cuMulticastCreate.
    */
    CUmulticastObjectProp mcProp = {};
    mcProp.numDevices = NUM_DEVICES;
    mcProp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; // single node
    mcProp.flags = 0; // SBZ
    // mcProp.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC; // multiple nodes/processes

    size_t granularity = 0;
    CUCHECK(cuMulticastGetGranularity( // or CU_MULTICAST_GRANULARITY_MINIMUM
        &granularity, &mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED)); 

    // 2MiB on H100 machines apparently
    printf("Recommended multicast granularity: %luMiB\n", granularity / 1024 / 1024);
    size_t size = (1 + (SIZE - 1) / granularity) * granularity; // round UP to nearest granularity
    mcProp.size = size;

    // Create Multicast Object (no devices and no physical memory associated yet)
    CUmemGenericAllocationHandle mcHandle;
    CUCHECK(cuMulticastCreate(&mcHandle, &mcProp));

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUdevice dev;
        CUCHECK(cuDeviceGet(&dev, dev_idx));
        CUCHECK(cuMulticastAddDevice(mcHandle, dev));
    }

    /*
        5. For each participating GPU bind physical memory allocated with 
           cuMemCreate as described above to the Multicast Handle. All 
           devices need to be added to the Multicast Team before binding 
           memory on any device.
    */
    CUmemGenericAllocationHandle memHandles[NUM_DEVICES];

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));

        CUmemAllocationProp memProp = {};
        memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        memProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        memProp.location.id = dev_idx;
        memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

        size_t mem_granularity = 0;
        CUCHECK(cuMemGetAllocationGranularity( // or CU_MEM_ALLOC_GRANULARITY_MINIMUM
            &mem_granularity, &memProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        printf("Recommended mem granularity (dev %d): %luMiB\n", dev_idx, mem_granularity / 1024 / 1024);
        if (size % mem_granularity != 0) {
            fprintf(stderr, "Size must be a multiple of mem granularity\n");
            exit(1);
        }

        // Allocate physical memory on the device
        CUCHECK(cuMemCreate(&memHandles[dev_idx], size, &memProp, 0));

        // Bind the physical memory to the multicast handle
        CUCHECK(cuMulticastBindMem(
            mcHandle, /*mcOffset=*/0, memHandles[dev_idx], /*memOffset=*/0, size, 0));
    }

    /*
        6. Reserve an address range, map the Multicast Handle and set Access 
           Rights as described above for regular Unicast mappings. Unicast 
           and Multicast mappings to the same physical memory are possible.
    */
    CUdeviceptr mcPtrs[NUM_DEVICES];
    CUdeviceptr memPtrs[NUM_DEVICES];

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
    
        CUCHECK(cuMemAddressReserve(&mcPtrs[dev_idx], size, granularity, 0, 0));
        CUCHECK(cuMemAddressReserve(&memPtrs[dev_idx], size, granularity, 0, 0));

        // Bind VAs to the multicast handle and physical memory
        // This way, we can choose to write to the same physical location
        // either with or without multicasting
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
        printf("Device %d: ", dev_idx);
        for (int i = 0; i < nelem; ++i) {
            host_mats[dev_idx][i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            if (i < 10)
                printf("%f ", host_mats[dev_idx][i]);
        }
        printf("... (%d elements)\n", nelem);
    }

    float *expected = (float*)malloc(size);
    printf("Expected: ");
    for (int i = 0; i < nelem; ++i) {
        expected[i] = 0.0f;
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
            expected[i] += host_mats[dev_idx][i];
        }
        for (int iter = 1; iter < NUM_ITERS; ++iter) {
            expected[i] *= NUM_DEVICES; 
        }
        if (i < 10)
            printf("%f ", expected[i]);
    }
    printf("... (%d elements)\n", nelem);

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaMemcpy((void*)memPtrs[dev_idx], host_mats[dev_idx], size, cudaMemcpyHostToDevice));
    }

    /*
        Perform the reduction
    */
    printf("Performing reduction...\n");


    // We will measure performance
    // Unfortunately, cudaEvent does not work with multimem for some reason
    // TODO: I think multimem.ld_reduce does lazy execution; must compare the entire flow

    // As a reference, first do a stupid load-store, check time
    CUDACHECK(cudaSetDevice(1));
    float *dummy;
    CUDACHECK(cudaMalloc(&dummy, size));
    auto start = std::chrono::high_resolution_clock::now();
    stupid_load_store<<<(nelem + 255) / 256, 256>>>((float*)dummy, (float*)memPtrs[1], nelem);
    CUDACHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Time for %d MiB plain load-store: %f ms\n", SIZE / 1024 / 1024, elapsed.count() * 1e3);
    CUDACHECK(cudaFree(dummy));
    
    // Perfrorm the reduction
    // WLOG, we will perform the reduction on the first device
    CUDACHECK(cudaSetDevice(0));
    
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        // all_reduce_float32_sum<<<(nelem + 1023) / 1024, 256>>>((float*)mcPtrs[0], nelem);
        all_reduce_float32_sum_coalesced<<<(nelem + 1024 * STRIDE - 1) / (1024 * STRIDE), 256>>>((float*)mcPtrs[0], nelem);
        CUDACHECK(cudaDeviceSynchronize());
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    double avg_time = elapsed.count() / NUM_ITERS;
    printf("Time for %d MiB allreduce using multimem: %lf ms\n", SIZE / (1024 * 1024), avg_time * 1e3);
    printf("NVLink BW: %lf GB/s\n", 4 * (SIZE / (1024. * 1024. * 1024.)) / avg_time); // Or 3 maybe?

    // Do one more load-store to check if this is lazily happening
    CUDACHECK(cudaSetDevice(2));
    CUDACHECK(cudaMalloc(&dummy, size));
    start = std::chrono::high_resolution_clock::now();
    stupid_load_store<<<(nelem + 255) / 256, 256>>>((float*)dummy, (float*)memPtrs[2], nelem);
    CUDACHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    printf("Time for %d MiB load-store after multimem: %f ms\n", SIZE / 1024 / 1024, elapsed.count() * 1e3);
    CUDACHECK(cudaFree(dummy));

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
    float TOL = 1e-2;
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        printf("Device %d: ", dev_idx);
        for (int i = 0; i < nelem; ++i) {
            if (i < 10)
                printf("%f ", host_mats[dev_idx][i]);
            if (fabs(expected[i] - host_mats[dev_idx][i]) > TOL) {
                fprintf(stderr, "Mismatch at device %d, index %d: expected %f, got %f\n", dev_idx, i, expected[i], host_mats[dev_idx][i]);
                exit(1);
            }
        }
        printf("... (%d elements)\n", nelem);
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

        free(host_mats[dev_idx]);
    }

    free(host_mats);
    free(expected);

    printf("Done\n");
    return 0;
}

__global__ void all_reduce_float32_sum(float *data, int nelem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 4 < nelem) {
        volatile float4 val;
        float *ptr = data + idx * 4;
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w) : "l"(ptr) : "memory"); // relaxed vs weak?
        asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w) : "memory"); // curious: what if I don't use asm here and just use plain assignment?
    }
}

__global__ void all_reduce_float32_sum_coalesced(float *data, const int nelem) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARPSIZE;
    int lane_id = threadIdx.x % WARPSIZE;

    /*
        math
        - each thread does   4 floats per iteration
        - each thread does   4 * WARPSIZE floats per iteration
        - each warp does     4 * STRIDE * WARPSIZE floats
        - all threads in a warp must do coalesced access
    */
    constexpr int nelem_per_iter = 4;
    constexpr int nelem_per_warp_per_iter = nelem_per_iter * WARPSIZE;
    constexpr int nelem_per_warp = STRIDE * nelem_per_warp_per_iter;
    int start_idx = nelem_per_warp * warp_id;

    for (int i = 0; i < STRIDE; ++i) {
        int idx = start_idx + i * nelem_per_warp_per_iter + lane_id * nelem_per_iter;
        if (idx < nelem) {
            volatile float4 val;
            float4 *ptr = (float4 *)(data + idx);
            asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w) : "l"(ptr) : "memory");
            asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w) : "memory");
        }
        __syncthreads();
    }
}

__global__ void stupid_load_store(float *dst, float *src, int nelem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nelem) {
        dst[idx] = src[idx];
    }
}
