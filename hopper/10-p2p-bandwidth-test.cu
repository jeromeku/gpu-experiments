#include "multi-gpu.cuh"

using namespace std;

void host_to_device(unsigned int messages_count, size_t message_size) {

    unique_ptr<unsigned char[]> host_memory(new unsigned char[messages_count * message_size]);

    unsigned char *device_memory;
    CUDACHECK(cudaMalloc(&device_memory, messages_count * message_size));
    CUDACHECK(cudaDeviceSynchronize()); // Ensure that all previous operations are completed (for accurate timing; otherwise, the time taken to copy the data will be included in the timing)

    auto begin = chrono::high_resolution_clock::now();

    for (unsigned int message_index = 0; message_index < messages_count; message_index++) {
        CUDACHECK(cudaMemcpy(device_memory + message_size * message_index,
                             host_memory.get() + message_size * message_index,
                             message_size, cudaMemcpyHostToDevice));
    }

    auto end = chrono::high_resolution_clock::now();

    CUDACHECK(cudaFree(device_memory));

    auto duration = chrono::duration<double>(end - begin);

    cout << "----------------------------------------" << endl;
    cout << "Total size: " << messages_count * message_size / 1e3 << " KB" << endl;
    cout << "Message size: " << message_size / 1e3 << " KB" << endl;
    cout << "Messages count: " << messages_count << endl;
    cout << "Latency: " << duration.count() << " seconds" << endl;
    cout << "Bandwidth: " << (messages_count * message_size) / 1e9 / duration.count() << " GB/s" << endl;
    cout << "----------------------------------------" << endl;
}

int main(int argc, char **argv) {

    cudaSetDevice(0);

    // Host to device (actual limit goes far beyond!)
    // host_to_device(100, 1024);
    // host_to_device(100, 2048);
    // host_to_device(100, 4096);
    // host_to_device(100, 8192);
    // host_to_device(100, 16384);
    // host_to_device(100, 32768);
    // host_to_device(100, 65536);
    // host_to_device(100, 131072);
    // host_to_device(100, 262144);
    // host_to_device(100, 524288);
    // host_to_device(100, 1048576);
    // host_to_device(100, 2097152);
    // host_to_device(100, 4194304);
    // host_to_device(100, 8388608);
    // host_to_device(100, 16777216);
    // host_to_device(100, 33554432);
    // host_to_device(100, 67108864);
    // host_to_device(100, 134217728);

    /*
     * Now moving on to GPU to GPU transfer
     */

    /*
     * Note that GPU-CPU-GPU transfer has another downside in NUMA systems:
     * You must care about NUMA affinity. Yet it is possible to partition GPUs 
     * depending on their CPU affinity with NVML (nvmlDeviceSetCpuAffinity). 
     * 
     * However, if you are doing GPU-CPU-GPU transfers, you just can't avoid this
     * since different GPUs can be on different NUMA nodes.
     * To avoid this, the only way is to throw CPU out of the picture.
     * --> Peer-to-peer (P2P) transaction
     */

    /*
     * P2P setup
     */

    // Check that unified virtual addressing (UVA) is enabled (should be for all modern NVIDIA GPUs)
    cudaDeviceProp device_properties;
    CUDACHECK(cudaGetDeviceProperties(&device_properties, 0));
    cout << "UVA enabled: " << device_properties.unifiedAddressing << endl;
    if (!device_properties.unifiedAddressing) {
        return 1;
    }

    /*
     * Another problem of P2P, if using PCIe: each root PCIe port defines a separate
     * PCIe hierarchy. If two GPUs are connected to the different root PCIe ports,
     * P2P communication is NOT possible. In order to check, you can use:
     */

    int can_access_peer_0_1;
    int can_access_peer_1_0;
    CUDACHECK(cudaDeviceCanAccessPeer(&can_access_peer_0_1, 0, 1));
    CUDACHECK(cudaDeviceCanAccessPeer(&can_access_peer_1_0, 1, 0));
    cout << "Device 0 can access device 1: " << can_access_peer_0_1 << endl;
    cout << "Device 1 can access device 0: " << can_access_peer_1_0 << endl;

    /*
     * Also, PCIe doesn't distinguish between PCIe P2P from NVLink.
     * So the above code will return true if 2 GPUs are either:
     * 1. Connected via PCIe P2P, in the same PCIe root port
     * 2. Connected via NVLink
     */

    /*
     * The above code returning true != P2P will be used.
     * You must manually enable P2P access between GPUs.
     * *Without NVLink, up to 8 peer connections are supported.
     */

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaDeviceEnablePeerAccess(1, 0));
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaDeviceEnablePeerAccess(0, 0));
    // cudaSetDevice (0);
    // cudaFree (pointers[0]);
    // cudaSetDevice (1);
    // cudaFree (pointers[1]);

    /*
     * Note that above enables access to all memory allocated before/after the enable.
     * The problem with above code is that (pro: simple):
     *  - No checking if a memory location is intended for P2P (every p2p access is enabled)
     *  - O(D * log n) time for future malloc(). D = device count, n = number of allocations so far
     */

    /*
     * What we can do is using low-level CUDA driver API: cuMemCreate, cuMemAddressReserve, cuMemMap, and 
     * cuMemSetAccess functions for physical memory allocation, virtual address range reservation, 
     * memory mapping, and access control respectively.  https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/
     * The key is on cuMemSetAccess, which lets us specify which devices can read/write to the memory.
     * By doing this, we can P2P to a particular memory region.
     */



}
