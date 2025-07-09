#include "multi-gpu.cuh"

/*
    Lessons learned:
        - cudaMemcpy does not involve SMs so it's fast even when GPU is busy doing other stuff
        - 16B vector copy is faster than 1B copy (duh)
        - Using raw PTX instruction "usually" results in faster ld/st (but not always)
        - When GPU is busy on-SM load/store can be slower up to 2x (prob due to SM availability)
        - Using vectors = better, but using for loops for less threads = no effect

    Outputs: (all GPUs were very busy when running this)
             (values can be off by +- 10GB/s, take it with grain of salt)
    
PCIe bandwidth test (theoretical max: 63 GB/s uni; reads > writes):
----------------------------------------
Direction: Host to device
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.914812 seconds
Bandwidth: 2.18624 GB/s
----------------------------------------
----------------------------------------
Direction: Device to Host
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.211967 seconds
Bandwidth: 9.43543 GB/s
----------------------------------------
NVLink/NVSwitch bandwidth test (theoretical max: 450 GB/s uni):
----------------------------------------
Direction: Device to device (using runtime API)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.00521014 seconds
Bandwidth: 383.867 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using naive copy kernel)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.0141604 seconds
Bandwidth: 141.238 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using naive PTX copy kernel)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.0125173 seconds
Bandwidth: 159.779 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using 16B vector copy kernel)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.0113856 seconds
Bandwidth: 175.66 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using 16B vector PTX copy kernel)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.0110307 seconds
Bandwidth: 181.312 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using 16B vector PTX copy kernel + prefetch 256B)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.0124095 seconds
Bandwidth: 161.167 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using better copy kernel)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.0119276 seconds
Bandwidth: 167.679 GB/s
----------------------------------------


Another run:
PCIe bandwidth test (theoretical max: 63 GB/s uni; reads > writes):
----------------------------------------
Direction: Host to device
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.678572 seconds
Bandwidth: 2.94737 GB/s
----------------------------------------
----------------------------------------
Direction: Device to Host
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.153232 seconds
Bandwidth: 13.0521 GB/s
----------------------------------------
NVLink/NVSwitch bandwidth test (theoretical max: 450 GB/s uni):
----------------------------------------
Direction: Device to device (using runtime API)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.00523409 seconds
Bandwidth: 382.11 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using naive copy kernel)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.0121969 seconds
Bandwidth: 163.976 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using naive PTX copy kernel)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.0121357 seconds
Bandwidth: 164.804 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using 16B vector copy kernel)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.00973767 seconds
Bandwidth: 205.388 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using 16B vector PTX copy kernel)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.00981414 seconds
Bandwidth: 203.788 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using 16B vector PTX copy kernel + prefetch 256B)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.0110261 seconds
Bandwidth: 181.387 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using better copy kernel) <--- I think this one heavily depends on SM availability
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.00921391 seconds
Bandwidth: 217.063 GB/s
----------------------------------------

*/




/*
    On a free 8-H100 machine:

PCIe bandwidth test (theoretical max: 63 GB/s uni; reads > writes):
----------------------------------------
Direction: Host to device
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.588046 seconds
Bandwidth: 3.4011 GB/s
----------------------------------------
----------------------------------------
Direction: Device to Host
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.0915205 seconds
Bandwidth: 21.853 GB/s
----------------------------------------
NVLink/NVSwitch bandwidth test (theoretical max: 450 GB/s uni):
----------------------------------------
Direction: Device to device (using runtime API)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.00520899 seconds
Bandwidth: 383.952 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using naive copy kernel)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.00613709 seconds
Bandwidth: 325.887 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using naive PTX copy kernel)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.0058159 seconds
Bandwidth: 343.885 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using 16B vector copy kernel)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.00561719 seconds
Bandwidth: 356.05 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using 16B vector PTX copy kernel)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.00561868 seconds
Bandwidth: 355.956 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using 16B vector PTX copy kernel + prefetch 256B)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.00566315 seconds
Bandwidth: 353.16 GB/s
----------------------------------------
----------------------------------------
Direction: Device to device (using better copy kernel)
Total size: 2000 MB
Message size: 2e+06 KB
Messages count: 1
Latency: 0.00568602 seconds
Bandwidth: 351.74 GB/s
----------------------------------------


*/

using namespace std;

#define KB(x) ((x) * 1000)
#define MB(x) (KB(x) * 1000)
#define GB(x) (MB(x) * 1000)

#define DEV0 0
#define DEV1 1

#define WARPSIZE 32
#define STRIDE 32

// Measures PCIe bandwidth between host and device
void host_device_test(unsigned int total_size, unsigned int message_size, int verbose = 1) {

    if (total_size % message_size != 0) {
        cout << "Total size must be a multiple of message size" << endl;
        exit(1);
    }
    unsigned int messages_count = total_size / message_size;

    unique_ptr<unsigned char[]> host_memory(new unsigned char[total_size]);

    // Initialize
    for (unsigned int i = 0; i < total_size; i++) {
        host_memory[i] = rand() % 256;
    }

    unsigned char *device_memory;
    CUDACHECK(cudaSetDevice(DEV0));
    CUDACHECK(cudaMalloc(&device_memory, total_size));
    CUDACHECK(cudaDeviceSynchronize());

    auto begin = chrono::high_resolution_clock::now();

    for (unsigned int i = 0; i < messages_count; i++) {
        CUDACHECK(cudaMemcpy(device_memory + message_size * i, 
                             host_memory.get() + message_size * i,
                             message_size, cudaMemcpyHostToDevice));
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration<double>(end - begin);

    if (verbose) {
        cout << "----------------------------------------" << endl;
        cout << "Direction: Host to device" << endl;
        cout << "Total size: " << total_size / 1e6 << " MB" << endl;
        cout << "Message size: " << message_size / 1e3 << " KB" << endl;
        cout << "Messages count: " << messages_count << endl;
        cout << "Latency: " << duration.count() << " seconds" << endl;
        cout << "Bandwidth: " << (total_size / 1e9) / duration.count() << " GB/s" << endl;
        cout << "----------------------------------------" << endl;
    }

    begin = chrono::high_resolution_clock::now();

    for (unsigned int i = 0; i < messages_count; i++) {
        CUDACHECK(cudaMemcpy(host_memory.get() + message_size * i,
                             device_memory + message_size * i, 
                             message_size, cudaMemcpyDeviceToHost));
    }

    end = chrono::high_resolution_clock::now();
    duration = chrono::duration<double>(end - begin);

    if (verbose) {
        cout << "----------------------------------------" << endl;
        cout << "Direction: Device to Host" << endl;
        cout << "Total size: " << total_size / 1e6 << " MB" << endl;
        cout << "Message size: " << message_size / 1e3 << " KB" << endl;
        cout << "Messages count: " << messages_count << endl;
        cout << "Latency: " << duration.count() << " seconds" << endl;
        cout << "Bandwidth: " << (total_size / 1e9) / duration.count() << " GB/s" << endl;
        cout << "----------------------------------------" << endl;
    }

    CUDACHECK(cudaFree(device_memory));
}

__global__ void copyKernel1(unsigned char *dst, const unsigned char *src, const unsigned int size);
__global__ void copyKernel2(unsigned char *dst, const unsigned char *src, const unsigned int size);
__global__ void copyKernel3(uint4 *dst, const uint4 *src, unsigned int size);
__global__ void copyKernel4(uint4 *dst, const uint4 *src, unsigned int size);
__global__ void copyKernel5(uint4 *dst, const uint4 *src, unsigned int size);
__global__ void copyKernel6(uint4 *dst, const uint4 *src, unsigned int size);

// Measures NVLink/NVSwitch bandwidth between devices
void device_device_test(unsigned int total_size, unsigned int message_size, int verbose = 1) {

    if (total_size % message_size != 0) {
        cout << "Total size must be a multiple of message size" << endl;
        exit(1);
    }
    unsigned int messages_count = total_size / message_size;

    unique_ptr<unsigned char[]> host_memory(new unsigned char[total_size]);
    unique_ptr<unsigned char[]> host_memory2(new unsigned char[total_size]);

    // Initialize
    for (unsigned int i = 0; i < total_size; i++) {
        host_memory[i] = rand() % 256;
    }

    // Allocate memory on device 0
    unsigned char *device0_memory;
    CUDACHECK(cudaSetDevice(DEV0));
    CUDACHECK(cudaMalloc(&device0_memory, total_size));
    CUDACHECK(cudaMemcpy(device0_memory, host_memory.get(), total_size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());

    // Allocate memory on device 1
    unsigned char *device1_memory;
    CUDACHECK(cudaSetDevice(DEV1));
    CUDACHECK(cudaMalloc(&device1_memory, total_size)); // no memcpy needed
    CUDACHECK(cudaDeviceSynchronize());

    // Turn on peer access
    CUDACHECK(cudaSetDevice(DEV0));
    CUDACHECK(cudaDeviceEnablePeerAccess(1, 0));
    CUDACHECK(cudaDeviceSynchronize());

    auto begin = chrono::high_resolution_clock::now();

    // Copy from device 0 to device 1
    for (unsigned int i = 0; i < messages_count; i++) {
        CUDACHECK(cudaMemcpyPeer(device1_memory + message_size * i, DEV1,
                                 device0_memory + message_size * i, DEV0,
                                 message_size));
    }
    CUDACHECK(cudaDeviceSynchronize());

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration<double>(end - begin);

    if (verbose) {
        cout << "----------------------------------------" << endl;
        cout << "Direction: Device to device (using runtime API)" << endl;
        cout << "Total size: " << total_size / 1e6 << " MB" << endl;
        cout << "Message size: " << message_size / 1e3 << " KB" << endl;
        cout << "Messages count: " << messages_count << endl;
        cout << "Latency: " << duration.count() << " seconds" << endl;
        cout << "Bandwidth: " << (total_size / 1e9) / duration.count() << " GB/s" << endl;
        cout << "----------------------------------------" << endl;
    }

    // Copy Kernel 1
    for (unsigned int i = 0; i < total_size; i++) {
        host_memory[i] = rand() % 256;
    }
    CUDACHECK(cudaSetDevice(DEV0));
    CUDACHECK(cudaMemcpy(device0_memory, host_memory.get(), total_size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());

    begin = chrono::high_resolution_clock::now();

    for (unsigned int i = 0; i < messages_count; i++) {
        copyKernel1<<<(message_size + 255) / 256, 256>>>(device1_memory + message_size * i,
                                                        device0_memory + message_size * i,
                                                        message_size);
    }
    CUDACHECK(cudaDeviceSynchronize());

    end = chrono::high_resolution_clock::now();
    duration = chrono::duration<double>(end - begin);

    if (verbose) {
        cout << "----------------------------------------" << endl;
        cout << "Direction: Device to device (using naive copy kernel)" << endl;
        cout << "Total size: " << total_size / 1e6 << " MB" << endl;
        cout << "Message size: " << message_size / 1e3 << " KB" << endl;
        cout << "Messages count: " << messages_count << endl;
        cout << "Latency: " << duration.count() << " seconds" << endl;
        cout << "Bandwidth: " << (total_size / 1e9) / duration.count() << " GB/s" << endl;
        cout << "----------------------------------------" << endl;
    }

    // Copy Kernel 2
    for (unsigned int i = 0; i < total_size; i++) {
        host_memory[i] = rand() % 256;
    }
    CUDACHECK(cudaSetDevice(DEV0));
    CUDACHECK(cudaMemcpy(device0_memory, host_memory.get(), total_size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());

    begin = chrono::high_resolution_clock::now();

    for (unsigned int i = 0; i < messages_count; i++) {
        copyKernel2<<<(message_size + 255) / 256, 256>>>(device1_memory + message_size * i,
                                                        device0_memory + message_size * i,
                                                        message_size);
    }
    CUDACHECK(cudaDeviceSynchronize());

    end = chrono::high_resolution_clock::now();
    duration = chrono::duration<double>(end - begin);

    if (verbose) {
        cout << "----------------------------------------" << endl;
        cout << "Direction: Device to device (using naive PTX copy kernel)" << endl;
        cout << "Total size: " << total_size / 1e6 << " MB" << endl;
        cout << "Message size: " << message_size / 1e3 << " KB" << endl;
        cout << "Messages count: " << messages_count << endl;
        cout << "Latency: " << duration.count() << " seconds" << endl;
        cout << "Bandwidth: " << (total_size / 1e9) / duration.count() << " GB/s" << endl;
        cout << "----------------------------------------" << endl;
    }

    // check that values are correct
    CUDACHECK(cudaSetDevice(DEV1));
    CUDACHECK(cudaMemcpy(host_memory2.get(), device1_memory, total_size, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaDeviceSynchronize());
    for (unsigned int i = 0; i < total_size; i++) {
        if (host_memory[i] != host_memory2[i]) {
            cout << "Values don't match" << endl;
            exit(1);
        }
    }

    // Copy Kernel 3
    for (unsigned int i = 0; i < total_size; i++) {
        host_memory[i] = rand() % 256;
    }
    CUDACHECK(cudaSetDevice(DEV0));
    CUDACHECK(cudaMemcpy(device0_memory, host_memory.get(), total_size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());

    begin = chrono::high_resolution_clock::now();

    for (unsigned int i = 0; i < messages_count; i++) {
        copyKernel3<<<(message_size + 4095) / 4096, 256>>>((uint4 *)(device1_memory + message_size * i),
                                                           (uint4 *)(device0_memory + message_size * i),
                                                           message_size);
    }
    CUDACHECK(cudaDeviceSynchronize());

    end = chrono::high_resolution_clock::now();
    duration = chrono::duration<double>(end - begin);

    if (verbose) {
        cout << "----------------------------------------" << endl;
        cout << "Direction: Device to device (using 16B vector copy kernel)" << endl;
        cout << "Total size: " << total_size / 1e6 << " MB" << endl;
        cout << "Message size: " << message_size / 1e3 << " KB" << endl;
        cout << "Messages count: " << messages_count << endl;
        cout << "Latency: " << duration.count() << " seconds" << endl;
        cout << "Bandwidth: " << (total_size / 1e9) / duration.count() << " GB/s" << endl;
        cout << "----------------------------------------" << endl;
    }

    // Copy Kernel 4
    for (unsigned int i = 0; i < total_size; i++) {
        host_memory[i] = rand() % 256;
    }
    CUDACHECK(cudaSetDevice(DEV0));
    CUDACHECK(cudaMemcpy(device0_memory, host_memory.get(), total_size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());

    begin = chrono::high_resolution_clock::now();

    for (unsigned int i = 0; i < messages_count; i++) {
        copyKernel4<<<(message_size + 4095) / 4096, 256>>>((uint4 *)(device1_memory + message_size * i),
                                                           (uint4 *)(device0_memory + message_size * i),
                                                           message_size);
    }
    CUDACHECK(cudaDeviceSynchronize());

    end = chrono::high_resolution_clock::now();
    duration = chrono::duration<double>(end - begin);

    if (verbose) {
        cout << "----------------------------------------" << endl;
        cout << "Direction: Device to device (using 16B vector PTX copy kernel)" << endl;
        cout << "Total size: " << total_size / 1e6 << " MB" << endl;
        cout << "Message size: " << message_size / 1e3 << " KB" << endl;
        cout << "Messages count: " << messages_count << endl;
        cout << "Latency: " << duration.count() << " seconds" << endl;
        cout << "Bandwidth: " << (total_size / 1e9) / duration.count() << " GB/s" << endl;
        cout << "----------------------------------------" << endl;
    }

    // check that values are correct
    CUDACHECK(cudaSetDevice(DEV1));
    CUDACHECK(cudaMemcpy(host_memory2.get(), device1_memory, total_size, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaDeviceSynchronize());
    for (unsigned int i = 0; i < total_size; i++) {
        if (host_memory[i] != host_memory2[i]) {
            cout << "Values don't match" << endl;
            exit(1);
        }
    }

    // Copy Kernel 5
    for (unsigned int i = 0; i < total_size; i++) {
        host_memory[i] = rand() % 256;
    }
    CUDACHECK(cudaSetDevice(DEV0));
    CUDACHECK(cudaMemcpy(device0_memory, host_memory.get(), total_size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());

    begin = chrono::high_resolution_clock::now();

    for (unsigned int i = 0; i < messages_count; i++) {
        copyKernel5<<<(message_size + 4095) / 4096, 256>>>((uint4 *)(device1_memory + message_size * i),
                                                           (uint4 *)(device0_memory + message_size * i),
                                                           message_size);
    }
    CUDACHECK(cudaDeviceSynchronize());

    end = chrono::high_resolution_clock::now();
    duration = chrono::duration<double>(end - begin);

    if (verbose) {
        cout << "----------------------------------------" << endl;
        cout << "Direction: Device to device (using 16B vector PTX copy kernel + prefetch 256B)" << endl;
        cout << "Total size: " << total_size / 1e6 << " MB" << endl;
        cout << "Message size: " << message_size / 1e3 << " KB" << endl;
        cout << "Messages count: " << messages_count << endl;
        cout << "Latency: " << duration.count() << " seconds" << endl;
        cout << "Bandwidth: " << (total_size / 1e9) / duration.count() << " GB/s" << endl;
        cout << "----------------------------------------" << endl;
    }

    // check that values are correct
    CUDACHECK(cudaSetDevice(DEV1));
    CUDACHECK(cudaMemcpy(host_memory2.get(), device1_memory, total_size, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaDeviceSynchronize());
    for (unsigned int i = 0; i < total_size; i++) {
        if (host_memory[i] != host_memory2[i]) {
            cout << "Values don't match" << endl;
            exit(1);
        }
    }

    // Copy Kernel 6
    for (int i = 0; i < total_size; i++) {
        host_memory[i] = rand() % 256;
    }
    CUDACHECK(cudaSetDevice(DEV0));
    CUDACHECK(cudaMemcpy(device0_memory, host_memory.get(), total_size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaDeviceSynchronize());

    begin = chrono::high_resolution_clock::now();

    for (unsigned int i = 0; i < messages_count; i++) {
        copyKernel6<<<(message_size + (4096 * STRIDE - 1)) / (4096 * STRIDE), 256>>>((uint4 *)(device1_memory + message_size * i),
                                                           (uint4 *)(device0_memory + message_size * i),
                                                           message_size);
    }
    CUDACHECK(cudaDeviceSynchronize());

    end = chrono::high_resolution_clock::now();
    duration = chrono::duration<double>(end - begin);

    if (verbose) {
        cout << "----------------------------------------" << endl;
        cout << "Direction: Device to device (using better copy kernel)" << endl;
        cout << "Total size: " << total_size / 1e6 << " MB" << endl;
        cout << "Message size: " << message_size / 1e3 << " KB" << endl;
        cout << "Messages count: " << messages_count << endl;
        cout << "Latency: " << duration.count() << " seconds" << endl;
        cout << "Bandwidth: " << (total_size / 1e9) / duration.count() << " GB/s" << endl;
        cout << "----------------------------------------" << endl;
    }

    // check that values are correct
    CUDACHECK(cudaSetDevice(DEV1));
    CUDACHECK(cudaMemcpy(host_memory2.get(), device1_memory, total_size, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaDeviceSynchronize());
    for (unsigned int i = 0; i < total_size; i++) {
        if (host_memory[i] != host_memory2[i]) {
            cout << "Values don't match" << endl;
            exit(1);
        }
    }

    CUDACHECK(cudaFree(device0_memory));
    CUDACHECK(cudaFree(device1_memory));
}

int main(int argc, char **argv) {

    // Warmup call
    host_device_test(MB(1), KB(1), 0);

    /*
        <PCIe bandwidth test>
        Running `sudo lspci -vvv -d 10de:` gives theoretical bandwidth of PCIe on this system. Apparently, it is:
          - Speed 32GT/s, Width x16
          - Meaning PCIe 5.0
          - So 63 GB/s is the theoretical max uni-directional bandwidth for either host-to-device or device-to-host
        Also note that writes are slower than reads in PCIe.
    */
    cout << "PCIe bandwidth test (theoretical max: 63 GB/s uni; reads > writes):" << endl;
    // host_device_test(GB(1), KB(250));
    // host_device_test(GB(1), MB(1));
    // host_device_test(GB(1), MB(10));
    // host_device_test(GB(1), MB(100));
    // host_device_test(GB(1), MB(1000));
    // host_device_test(GB(2), KB(250));
    // host_device_test(GB(2), MB(1));
    // host_device_test(GB(2), MB(10));
    // host_device_test(GB(2), MB(100));
    // host_device_test(GB(2), MB(1000));
    host_device_test(GB(2), MB(2000));

    /*
        <NVLink/NVSwitch bandwidth test>
        Hopper uses NVLink 4.0, which has a theoretical max bandwidth of 450 GB/s uni-directional.
    */
    cout << "NVLink/NVSwitch bandwidth test (theoretical max: 450 GB/s uni):" << endl;
    // device_device_test(GB(1), KB(250));
    // device_device_test(GB(1), MB(1));
    // device_device_test(GB(1), MB(10));
    // device_device_test(GB(1), MB(100));
    // device_device_test(GB(1), MB(1000));
    // device_device_test(GB(2), KB(250));
    // device_device_test(GB(2), MB(1));
    // device_device_test(GB(2), MB(10));
    // device_device_test(GB(2), MB(100));
    // device_device_test(GB(2), MB(1000));
    device_device_test(GB(2), MB(2000));

    return 0;
}

__global__ void copyKernel1(unsigned char *dst, const unsigned char *src, const unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

__global__ void copyKernel2(unsigned char *dst, const unsigned char *src, const unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        volatile int value;
        asm volatile (
            "{ ld.weak.global.u8 %0, [%1];"
              "st.weak.global.u8 [%2], %0; }"
            : "=r"(value)
            : "l"(src + idx), "l"(dst + idx)
            : "memory"
        );
    }
}

__global__ void copyKernel3(uint4 *dst, const uint4 *src, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx * sizeof(uint4) < size) {
        dst[idx] = src[idx];
    }
}

__global__ void copyKernel4(uint4 *dst, const uint4 *src, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx * sizeof(uint4) < size) {
        volatile uint4 value;
        asm volatile (
            "{ ld.weak.global.v4.u32 {%0, %1, %2, %3}, [%4];"
              "st.weak.global.v4.u32 [%5], {%0, %1, %2, %3}; }"
            : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
            : "l"(src + idx), "l"(dst + idx)
            : "memory"
        );
    }
}

__global__ void copyKernel5(uint4 *dst, const uint4 *src, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx * sizeof(uint4) < size) {
        volatile uint4 value;
        asm volatile (
            "{ ld.weak.global.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
              "st.weak.global.v4.u32 [%5], {%0, %1, %2, %3}; }"
            : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
            : "l"(src + idx), "l"(dst + idx)
            : "memory"
        );
    }
}

__global__ void copyKernel6(uint4 *dst, const uint4 *src, unsigned int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpid = tid / WARPSIZE;
    unsigned int laneid = tid % WARPSIZE;
    unsigned int idx = warpid * WARPSIZE * STRIDE + laneid;

    #pragma unroll
    for (int i = 0; i < STRIDE; ++i) {
        if (idx + WARPSIZE * i < size)
            dst[idx + WARPSIZE * i] = src[idx + WARPSIZE * i];
        __syncthreads();
    }
}
