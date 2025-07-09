/*
    Three multimem instructions are available in ptx
    - multimem.ld_reduce
    - multimem.st
    - multimem.red

    These instructions and only these instructions operate on 
    multimem addresses (other instructions / non-global addr result in UB).
    They will access all of the multiple 
    operations which the multimem address points to

    Multimem addresses are created by multicast object API from CUDA Driver
    (21-multicast-object.cu)

    Apparently this is only supported on NVLink-connected GPUs
*/

/*
    CUDA Programming Guide on working with multicast objects:

    1. Query Multicast Support
    2. Create a Multicast Handle with cuMulticastCreate.
    3. Share the Multicast Handle with all processes that control a 
       GPU which should participate in a Multicast Team. This works 
       with cuMemExportToShareableHandle as described above.
    4. Add all GPUs that should participate in the Multicast Team with 
       cuMulticastAddDevice.
    5. For each participating GPU bind physical memory allocated with 
       cuMemCreate as described above to the Multicast Handle. All 
       devices need to be added to the Multicast Team before binding 
       memory on any device.
    6. Reserve an address range, map the Multicast Handle and set Access 
       Rights as described above for regular Unicast mappings. Unicast 
       and Multicast mappings to the same physical memory are possible. 
       See the Virtual Aliasing Support section above how to ensure 
       consistency between multiple mappings to the same physical memory.
    7. Use the multimem PTX instructions with the multicast mappings.
*/

#include "multi-gpu.cuh"

constexpr int NUM_DEVICES = 4;
constexpr int SIZE = 64 * 1024 * 1024; // 64MiB

void print_device_mem(int dev_idx, CUdeviceptr ptr, size_t size) {
    uint8_t* host_mem = new uint8_t[size];
    CUDACHECK(cudaSetDevice(dev_idx));
    CUDACHECK(cudaMemcpy(host_mem, (void*)ptr, size, cudaMemcpyDeviceToHost));
    printf("Device %d: ", dev_idx);
    for (int i = 0; i < size; ++i) {
        printf("%02x ", host_mem[i]);
    }
    printf("\n");
    delete[] host_mem;
}

void write_random_data(int dev_idx, CUdeviceptr ptr, size_t size) {
    uint8_t* host_mem = new uint8_t[size];
    for (int i = 0; i < size; ++i) {
        host_mem[i] = rand() % 256;
    }
    CUDACHECK(cudaSetDevice(dev_idx));
    CUDACHECK(cudaMemcpy((void*)ptr, host_mem, size, cudaMemcpyHostToDevice));
    delete[] host_mem;
}

int main() {

    assert(NUM_DEVICES > 1);
    assert(SIZE >= 1024 * 1024 && SIZE % (1024 * 1024) == 0);

    // cuInit must be called before any Driver API calls, and argument SBZ
    CUCHECK(cuInit(0));

    // Now following the CUDA programming guide...
    printf("Setting up Multicast Objects...\n");

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

    /*
        3. Share the Multicast Handle with all processes that control a 
           GPU which should participate in a Multicast Team. This works 
           with cuMemExportToShareableHandle as described above.
    */
    // We skip this step because we are working on a single node

    /*
        4. Add all GPUs that should participate in the Multicast Team with 
           cuMulticastAddDevice.
    */
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

    // Setup is complete. Now we can run stuff on the devices
    printf("Setup complete. Running experiments...\n");

    /*
        Exp 1. Make sure that what we understand is correct (sanity checks)
    */
    printf("\nExp 1. Sanity checks\n\n");
    
    // First, initialize each physical location to random data
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        write_random_data(dev_idx, memPtrs[dev_idx], 16);
    }

    // Read from physical handles
    printf("The following should NOT match:\n");
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        print_device_mem(dev_idx, memPtrs[dev_idx], 16);
    }

    // Write to mc handle (dev 1)
    write_random_data(1, mcPtrs[1], 16);

    // Multicast uses weak consistency by default, so we need to sync
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaDeviceSynchronize());
    }

    // Read from all handles
    printf("\nThe following SHOULD all match:\n");
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        print_device_mem(dev_idx, mcPtrs[dev_idx], 16);
        print_device_mem(dev_idx, memPtrs[dev_idx], 16);
    }

    /*
        Exp 2. We now know that:
            - Writing to mc handle writes to all physical locations
            - Writing to physical location writes to that location only
            - Reading from physical handle reads from that location only
        
        But, what happens if we read from mc handle but the physical locations have different data?
    */
    printf("\nExp 2. Reading from mc handle when physical locations have different data\n\n");

    // Write to physical memory handle
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        write_random_data(dev_idx, memPtrs[dev_idx], 16);
    }
    
    // Sync, just in case (not sure if really needed)
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaDeviceSynchronize());
    }

    // Read from physical handles
    printf("We wrote the following to each dev:\n");
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        print_device_mem(dev_idx, memPtrs[dev_idx], 16);
    }

    // Read from mc handle
    printf("\nReading from mc handle on each dev:\n");
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        print_device_mem(dev_idx, mcPtrs[dev_idx], 16);
    }
    printf("\nSo apparently, the mc handle reads from the lowest-index device\n");

    /*
        Exp 3. But does that mean reading from mc Handle in higher-index devices
               will be slow?
    */
    printf("\nExp 3. Reading from mc handle on higher-index devices\n\n");

    // Write to physical memory handle, so that there is no caching
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        write_random_data(dev_idx, memPtrs[dev_idx], 16);
    }

    printf("Reading from mc handle on each dev:\n");
    uint8_t* host_mem = new uint8_t[size];
    for (int dev_idx = NUM_DEVICES - 1; dev_idx >= 0; --dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        auto start = std::chrono::high_resolution_clock::now(); // unlike previous, read the entire SIZE
        CUDACHECK(cudaMemcpy(host_mem, (void*)mcPtrs[dev_idx], size, cudaMemcpyDeviceToHost));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        printf("Time taken on device %d: %f ms\n", dev_idx, 1000 * elapsed.count());
    }
    delete[] host_mem;

    printf("\nEnsure that they return the same values:\n");
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        print_device_mem(dev_idx, mcPtrs[dev_idx], 16);
    }

    printf("\nEnsure that reading from physical handles still return different values\n");
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        print_device_mem(dev_idx, memPtrs[dev_idx], 16);
    }

    printf("\nVery interesting... it seems like only the initial read is slow, regardless of device index, even though we are reading from dev 0\n");

    /*
        Exp 4. Another sanity check: write to mcHandle should take longer than writing to physical handle
    */
    printf("\nExp 4. Writing to mc handle vs writing to physical handle\n\n");

    // Write to physical memory handle, to reset stuff (like cache)
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        write_random_data(dev_idx, memPtrs[dev_idx], 16);
    }

    host_mem = new uint8_t[size];
    for (int i = 0; i < size; ++i) {
        host_mem[i] = rand() % 256;
    }
    
    printf("Writing to mc handle on each dev:\n");
    for (int dev_idx = NUM_DEVICES - 1; dev_idx >= 0; --dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        auto start = std::chrono::high_resolution_clock::now(); // unlike previous, read the entire SIZE
        CUDACHECK(cudaMemcpy((void*)mcPtrs[dev_idx], host_mem, size, cudaMemcpyHostToDevice));        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        printf("Time taken on device %d: %f ms\n", dev_idx, 1000 * elapsed.count());
    }

    printf("\nWriting to physical handle on each dev:\n");
    for (int dev_idx = NUM_DEVICES - 1; dev_idx >= 0; --dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        auto start = std::chrono::high_resolution_clock::now(); // unlike previous, read the entire SIZE
        CUDACHECK(cudaMemcpy((void*)memPtrs[dev_idx], host_mem, size, cudaMemcpyHostToDevice));        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        printf("Time taken on device %d: %f ms\n", dev_idx, 1000 * elapsed.count());
    }
    delete[] host_mem;

    printf("\nThey all take about the same time... then do they just do lazy loading?\n");

    /*
        Hypothesis: there exists some sort of specialized HW for multicast objects. 
            - When writing to mcHandle, the dev only writes to its local HW
            - When reading from mcHandle, the participating devs execute an "exchange";
                i.  if there is recent write-to-mcHandle, the data is broadcasted to all 
                    participating devs
                ii. if there was no recent write-to-mcHandle, the data is read from the 
                    lowest-index dev

        Not sure how to check this though. Let's move on
    */

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

    printf("Done\n");
    return 0;
}

/*
    PTX syntax

    // Integer type:

    multimem.ld_reduce{.ldsem}{.scope}{.ss}.op.type      d, [a];
    multimem.st{.stsem}{.scope}{.ss}.type                [a], b;
    multimem.red{.redsem}{.scope}{.ss}.op.type           [a], b;

    .ss =       { .global }
    .ldsem =    { .weak, .relaxed, .acquire }
    .stsem =    { .weak, .relaxed, .release }
    .redsem =   { .relaxed, .release }
    .scope =    { .cta, .cluster, .gpu, .sys }
    .op  =      { .min, .max, .add, .and, .or, .xor }
    .type =     { .b32, .b64,  .u32, .u64, .s32, .s64 }


    // Floating point type:

    multimem.ld_reduce{.ldsem}{.scope}{.ss}.op{.acc_prec}{.vec}.type    d, [a];
    multimem.st{.stsem}{.scope}{.ss}{.vec}.type                         [a], b;
    multimem.red{.redsem}{.scope}{.ss}.redop{.vec}.redtype              [a], b;

    .ss =       { .global }
    .ldsem =    { .weak, .relaxed, .acquire }
    .stsem =    { .weak, .relaxed, .release }
    .redsem =   { .relaxed, .release }
    .scope =    { .cta, .cluster, .gpu, .sys }
    .op  =      { .min, .max, .add }
    .redop  =   { .add }
    .acc_prec = { .acc::f32, .acc::f16 }
    .vec =      { .v2, .v4, .v8 }
    .type=      { .f16, .f16x2, .bf16, .bf16x2, .f32, .f64, .e5m2, .e5m2x2, .e5m2x4, .e4m3, .e4m3x2, .e4m3x4 }
    .redtype =  { .f16, .f16x2, .bf16, .bf16x2, .f32, .f64 }
*/

/*
    Example from https://github.com/NVIDIA/multi-gpu-programming-models/blob/master/multi_node_p2p/jacobi_kernels.cu

    __global__ void all_reduce_norm_barrier_kernel(
        float* l2_norm,
        float* partial_l2_norm_mc,
        unsigned int* arrival_counter_uc, 
        unsigned int* arrival_counter_mc,
        const unsigned int expected_count
    ) {

        // Must be run on a single thread
        assert(blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z == 1);
        
        // Accumulator
        float l2_norm_sum = 0.0;

        // atomic reduction to all replicas
        // this can be conceptually thought of as __threadfence_system(); atomicAdd_system(arrival_counter_mc, 1);
        asm volatile ("multimem.red.release.sys.global.add.u32 [%0], %1" :: "l"(arrival_counter_mc), "n"(1) : "memory");

        // Need a fence between Multicast (mc) and Unicast (uc) access to the same memory `arrival_counter_uc` and `arrival_counter_mc`:
        // - fence.proxy instructions establish an ordering between memory accesses that may happen through different proxies
        // - Value .alias of the .proxykind qualifier refers to memory accesses performed using virtually aliased addresses to the same memory location.
        // from https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar
        asm volatile ("fence.proxy.alias" ::: "memory");

        // spin wait with acquire ordering on UC mapping till all peers have arrived in this iteration
        // Note: all ranks need to reach another barrier after this kernel, such that it is not possible for the barrier to be unblocked by an
        // arrival of a rank for the next iteration if some other rank is slow.
        cuda::atomic_ref<unsigned int,cuda::thread_scope_system> ac(arrival_counter_uc);
        while (expected_count > ac.load(cuda::memory_order_acquire));

        // Atomic load reduction from all replicas. It does not provide ordering so it can be relaxed.
        asm volatile ("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : "=f"(l2_norm_sum) : "l"(partial_l2_norm_mc) : "memory");

        *l2_norm = std::sqrt(l2_norm_sum);
    }
*/
