#include "multi-gpu.cuh"

using namespace std;

__global__ void allReduceFloat32SumRing(
    const int my_rank,         // dev_idx
    const int num_ranks,       // NUM_DEVS
    volatile float *my_data,   // @ current device HBM (nelem * sizeof(float))
    volatile float *next_data, // @ next device HBM    (nelem * sizeof(float))
    volatile int *rx_flag,     // @ current device HBM (blockDim * sizeof(int))
                               //   set to {seed + iter * 2 + 1} by another dev on data arrival, 
                               //   set to {seed + iter * 2} by current after reading
    volatile float *rx_buffer, // @ current device HBM (blockDim * chunk_nelem * sizeof(float))
    volatile int *tx_flag,     // @ next device HBM    (blockDim * sizeof(int))
                               //   set to {seed + iter * 2 + 1} by current dev on data write, 
                               //   set to {seed + iter * 2} by next dev after reading
    volatile float *tx_buffer, // @ next device HBM    (blockDim * chunk_nelem * sizeof(float))
    const int nelem,           // total number of float32 elements
    const int chunk_nelem,     // num elements per chunk := (nelem + NUM_DEVS - 1) / NUM_DEVS
    const int seed
) {

    int tx_chunk_idx, rx_chunk_idx;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int bdim = blockDim.x;
    const int buf_idx = bid * bdim + tid;

    if (buf_idx >= chunk_nelem) // TODO: use 128-bit vector ops
        return;
    
    if (tid == 0)
        rx_flag[bid] = seed;

    // Phase 1. Share-reduction phase
    for (int iter = 0; iter < num_ranks - 1; ++iter) {

        // if (tid == 0)
        //     printf("rank %d iter %d tx\n", my_rank, iter);

        tx_chunk_idx = (my_rank - iter + num_ranks) % num_ranks;
        rx_chunk_idx = (my_rank - 1 - iter + num_ranks) % num_ranks;

        // if (my_rank == 0) {
        //     printf("Rank %d iter %d tx_chunk %d rx_chunk %d\n", my_rank, iter, tx_chunk_idx, rx_chunk_idx);
        // }

        // Send chunk[p - iter]
        while (tx_flag[bid] != seed + iter * 2)
            ; // wait for buffer to become empty
        tx_buffer[buf_idx] = my_data[tx_chunk_idx * chunk_nelem + buf_idx];
        __syncthreads(); // TODO: this is only effective within a single block. Need another method
        if (tid == 0)
            // *tx_flag = seed + iter * 2 + 1;
            // maybe __threadfence_system();
            asm volatile("{st.release.sys.global.L1::no_allocate.s32 [%0], %1;}" :: "l"(tx_flag + bid), "r"(seed + iter * 2 + 1) : "memory");

        // if (tid == 0)
        //     printf("rank %d iter %d rx\n", my_rank, iter);

        // Receive and reduce chunk[p - 1 - iter]
        while (rx_flag[bid] != seed + iter * 2 + 1)
            ; // wait for buffer to fill
        // if (my_rank == 0)
        //     printf("Rank %d iter %d received data %f\n", my_rank, iter, rx_buffer[tid]);
        my_data[rx_chunk_idx * chunk_nelem + buf_idx] = my_data[rx_chunk_idx * chunk_nelem + buf_idx] + rx_buffer[buf_idx];
        __syncthreads();
        if (tid == 0)
            // *rx_flag = seed + (iter + 1) * 2;
            // maybe __threadfence_system();
            asm volatile("{st.release.sys.global.L1::no_allocate.s32 [%0], %1;}" :: "l"(rx_flag + bid), "r"(seed + (iter + 1) * 2) : "memory");
    }

    // printf("Rank %d: first phase done\n", my_rank);

    // Wait for the next device to complete the previous phase
    int new_seed = seed + (num_ranks - 1) * 2;
    while (tx_flag[bid] != new_seed)
        ; // TODO: is this necessary?

    // printf("Rank %d Waiting for other device to complete phase 1 finished\n", my_rank);
    
    
    // Phase 2. Share-only phase
    for (int iter = 0; iter < num_ranks - 1; ++iter) {

        tx_chunk_idx = (my_rank + 1 - iter + num_ranks) % num_ranks;

        // Send chunk[p + 1 - iter]; no need to use buffer here, just write directly
        while (rx_flag[bid] < new_seed + iter) // one extra load on the first iteration (on first iteration, this always evaluates to true)
            ; // wait for data to be received from prev device (on first iteration, this should always be true, since we aren't receiving any data)
              // It's okay if rx_flax is greater than the right value. As long as the current chunk data arrived, doesn't matter if latter ones arrived as well.
        // if (my_rank == 0)
        //     printf("Rank %d -> rank %d on chunk %d. Writing %f\n", my_rank, (my_rank + 1) % num_ranks, tx_chunk_idx, my_data[tx_chunk_idx * chunk_nelem + tid]);
        
        // volatile float val;
        // volatile float *ptr = &my_data[tx_chunk_idx * chunk_nelem + tid];
        // asm volatile("{ld.relaxed.sys.global.L1::no_allocate.f32 %0, [%1];}": "=f"(val) : "l"(ptr) :"memory");
        // asm volatile("{ld.acquire.sys.global.L1::no_allocate.f32 %0, [%1];}": "=f"(val) : "l"(ptr) :"memory");
        // asm volatile("{st.relaxed.sys.global.L1::no_allocate.f32 [%0], %1;}" :: "l"(&next_data[tx_chunk_idx * chunk_nelem + tid]), "f"(val) : "memory");
        // asm volatile("{st.release.sys.global.L1::no_allocate.f32 [%0], %1;}" :: "l"(&next_data[tx_chunk_idx * chunk_nelem + tid]), "f"(val) : "memory");

        // next_data[tx_chunk_idx * chunk_nelem + tid] = val;
        next_data[tx_chunk_idx * chunk_nelem + buf_idx] = my_data[tx_chunk_idx * chunk_nelem + buf_idx];
        __syncthreads();
        if (tid == 0)
            // *tx_flag = new_seed + iter + 1; // number of chunks written = *tx_flag - new_seed
            // By using release store, we effectly set a memory barrier here, ensuring that (flag written == value was written)
            // maybe __threadfence_system();
            asm volatile("{st.release.sys.global.L1::no_allocate.s32 [%0], %1;}" :: "l"(tx_flag + bid), "r"(new_seed + iter + 1) : "memory");
    }
}

static constexpr int NUM_DEVS = 8;
static constexpr int SIZE = 1024 * 1024 * 1024; // 1024 MiB

int main() {

    // P2P Setup: Device (n) must be able to access device (n + 1) (but not the other way around)
    for (int dev_idx = 0; dev_idx < NUM_DEVS; ++dev_idx) {
        int next_dev_idx = (dev_idx + 1) % NUM_DEVS;

        int can_access;
        CUDACHECK(cudaDeviceCanAccessPeer(&can_access, dev_idx, next_dev_idx)); // can (dev_idx) access (dev_idx + 1)?
        if (!can_access) {
            printf("Error: Device %d cannot access device %d\n", dev_idx, next_dev_idx);
            return 1;
        }

        CUDACHECK(cudaSetDevice(dev_idx));
        // cudaDeviceEnablePeerAccess is unidirectional;
        // this allows current device -> peer device (passed as arg) access
        CUDACHECK(cudaDeviceEnablePeerAccess(next_dev_idx, 0));
    }

    /*
        Setup the data
    */
    assert(SIZE % sizeof(float) == 0);

    int nelem = SIZE / sizeof(float);
    float **host_mats = (float**)malloc(NUM_DEVS * sizeof(float*));
    float **dev_mats = (float**)malloc(NUM_DEVS * sizeof(float*));
    srand(static_cast<unsigned int>(time(nullptr))); // random seed

    printf("\n");
    for (int dev_idx = 0; dev_idx < NUM_DEVS; ++dev_idx) {
        host_mats[dev_idx] = (float*)malloc(SIZE);
        printf("Device %d: ", dev_idx);
        for (int i = 0; i < nelem; ++i) {
            host_mats[dev_idx][i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            if (i < 4)
                printf("%f ", host_mats[dev_idx][i]);
        }
        printf("... (%d elements)\n", nelem);

        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaMalloc((void**)&dev_mats[dev_idx], SIZE));
        CUDACHECK(cudaMemcpy(dev_mats[dev_idx], host_mats[dev_idx], SIZE, cudaMemcpyHostToDevice));
    }

    float *expected = (float*)malloc(SIZE);
    printf("\nExpected: ");
    for (int i = 0; i < nelem; ++i) {
        expected[i] = 0.0f;
        for (int dev_idx = 0; dev_idx < NUM_DEVS; ++dev_idx) {
            expected[i] += host_mats[dev_idx][i];
        }
        if (i < 4)
            printf("%f ", expected[i]);
    }
    printf("... (%d elements)\n", nelem);

    /*
        Perform reduction
    */

    // Prepare kernel parameters
    const int chunk_nelem = (nelem + NUM_DEVS - 1) / NUM_DEVS;
    int **flags = (int**)malloc(NUM_DEVS * sizeof(int*));
    float **buffers = (float**)malloc(NUM_DEVS * sizeof(float*));
    for (int dev_idx = 0; dev_idx < NUM_DEVS; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaMalloc((void**)&flags[dev_idx], sizeof(int)));
        CUDACHECK(cudaMalloc((void**)&buffers[dev_idx], chunk_nelem * sizeof(float)));
    }
    
    // Launch kernels
    auto start = std::chrono::high_resolution_clock::now();
    for (int dev_idx = 0; dev_idx < NUM_DEVS; ++dev_idx) {
        int next_dev_idx = (dev_idx + 1) % NUM_DEVS;
        CUDACHECK(cudaSetDevice(dev_idx));
        allReduceFloat32SumRing<<<(nelem + 255) / 256, 256>>>(
            dev_idx, NUM_DEVS, 
            dev_mats[dev_idx], dev_mats[next_dev_idx],
            flags[dev_idx], buffers[dev_idx],
            flags[next_dev_idx], buffers[next_dev_idx],
            nelem, chunk_nelem,
            0xbeef
        );
    }
    for (int dev_idx = 0; dev_idx < NUM_DEVS; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Time for %d MiB ring all reduce: %f ms\n", SIZE / 1024 / 1024, elapsed.count() * 1e3);

    /*
        Bring back data
    */
    for (int dev_idx = 0; dev_idx < NUM_DEVS; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaMemcpy(host_mats[dev_idx], dev_mats[dev_idx], SIZE, cudaMemcpyDeviceToHost));
    }

    /*
        Verify the results
    */
    float TOL = 1e-5;
    for (int dev_idx = 0; dev_idx < NUM_DEVS; ++dev_idx) {
        printf("Device %d: ", dev_idx);
        for (int i = 0; i < nelem; ++i) {
            if (fabs(expected[i] - host_mats[dev_idx][i]) > TOL) {
                fprintf(stderr, "Mismatch at device %d, index %d: expected %f, got %f\n", dev_idx, i, expected[i], host_mats[dev_idx][i]);
                exit(1);
            }
        }
    }
    printf("\n");
    for (int dev_idx = 0; dev_idx < NUM_DEVS; ++dev_idx) {
        printf("Device %d: ", dev_idx);
        for (int i = 0; i < 4; ++i) {
            printf("%f ", host_mats[dev_idx][i]);
        }
        printf("... (%d elements)\n", nelem);
    }

    /*
        Cleanup and exit
    */

    // Free resources
    for (int dev_idx = 0; dev_idx < NUM_DEVS; ++dev_idx) {
        free(host_mats[dev_idx]);
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaFree(dev_mats[dev_idx]));
        CUDACHECK(cudaFree(flags[dev_idx]));
        CUDACHECK(cudaFree(buffers[dev_idx]));
    }

    free(flags);
    free(buffers);
    free(host_mats);
    free(dev_mats);
    free(expected);

    printf("Done\n");
    return 0;
}
