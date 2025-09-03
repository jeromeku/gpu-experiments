#ifndef MULTI_GPU_CUH
#define MULTI_GPU_CUH

// C
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <assert.h>

// C++
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <omp.h>
#include <thread>

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"
#include "cuda_fp8.h"

// Pybind11
#include <pybind11/pybind11.h>
namespace py = pybind11;

// CUDA driver API
#define CUCHECK(cmd) do {                                     \
    CUresult err = cmd;                                       \
    if (err != CUDA_SUCCESS) {                                \
        const char *errStr;                                   \
        cuGetErrorString(err, &errStr);                       \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, errStr);                      \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// CUDA runtime API
#define CUDACHECK(cmd) do {                                   \
    cudaError_t err = cmd;                                    \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

/*
    Example usage:

    benchmark("Sorting a large vector", [] {
        std::vector<int> v(1'000'000);
        std::iota(v.begin(), v.end(), 0);  // Fill with 0,1,2,...999999
        std::sort(v.begin(), v.end(), std::greater<>()); // Sort in descending order
    });
*/
template <typename Func>
void benchmark(const char* message, Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << message << ": " << elapsed.count() << " ms" << std::endl;
}

/*
    Example usage:

    TensorIPC tensor_ipc(local_rank, local_world_size);

    std::vector<TensorIPCEntry> out_entries(tensor_ipc.local_world_size_);
    tensor_ipc.all_gather_ptrs(out_entries.data(), ptr);

    // Access the actual pointer
    void *other_ptr = out_entries[0].raw_ptr;

    Also, export to Python:

    PYBIND11_MODULE(_C, m){
        pybind11::class_<TensorIPC>(m, "TensorIPC")
            .def(pybind11::init<int, int>());
    }
*/
#include <cerrno>
#include <fcntl.h>
#include <iostream>
#include <semaphore.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

struct TensorIPCEntry {
    bool is_imported;
    void *raw_ptr; // if imported, MUST be freed with cudaIpcCloseMemHandle (not cudaFree), which is why we have a custom struct

    __host__ inline void free_imported_ptr() {
        // Close the import pointer (MUST be done first before cudaFree on the source)
        if (is_imported && raw_ptr)
            CUDACHECK(cudaIpcCloseMemHandle(raw_ptr));
    }

    __host__ inline void import_from_handle(cudaIpcMemHandle_t import_handle) {
        if (is_imported && raw_ptr) {
            std::cerr << "WARNING: Importing a new pointer while already importing one. Freeing the old one." << std::endl;
            free_imported_ptr();
        }

        // Import the IPC handle. This implicitly & lazily does cudaDeviceEnablePeerAccess
        CUDACHECK(cudaIpcOpenMemHandle(&raw_ptr, import_handle, cudaIpcMemLazyEnablePeerAccess)); // this is the only flag supported
        is_imported = true;
    }

    __host__ inline TensorIPCEntry() : is_imported(false), raw_ptr(nullptr) {}
    __host__ inline TensorIPCEntry(void *raw_ptr) : is_imported(false), raw_ptr(raw_ptr) {}
    __host__ inline TensorIPCEntry(cudaIpcMemHandle_t import_handle) : is_imported(true) {
        import_from_handle(import_handle);
    }

    __host__ inline ~TensorIPCEntry() {
        free_imported_ptr();
        // It is recommended to call TensorIPC::sync() after this
    }
};

struct TensorIPC {
    __host__ inline TensorIPC(int local_rank, int local_world_size)
        : local_rank_(local_rank), 
          local_world_size_(local_world_size) {
        if (local_rank_ >= local_world_size_)
            throw std::runtime_error("Local rank is greater than local world size");
        if (local_world_size_ > MAX_LOCAL_WORLD_SIZE_)
            throw std::runtime_error("Local world size is greater than MAX_LOCAL_WORLD_SIZE");
        if (!is_device_id_valid())
            throw std::runtime_error("Invalid device ID. Must be equal to local rank.");
        if (!is_tensor_ipc_supported())
            throw std::runtime_error("Tensor IPC is not supported on this device");

        // Create or open an existing shared memory
        int shm_id = shm_open(SHM_KEY_, O_CREAT | O_RDWR, 0600);
        if (shm_id == -1)
            throw std::runtime_error("Failed to create shared memory");

        // Create named semaphores
        sem_counter_ = sem_open(SEM_COUNTER_KEY_, O_CREAT, 0600, 1);
        sem_enter_ = sem_open(SEM_ENTER_KEY_, O_CREAT, 0600, 0);
        sem_exit_ = sem_open(SEM_EXIT_KEY_, O_CREAT, 0600, 0);
        sem_ready_ = sem_open(SEM_READY_KEY_, O_CREAT, 0600, 0);
        if (sem_counter_ == SEM_FAILED || sem_enter_ == SEM_FAILED || sem_exit_ == SEM_FAILED || sem_ready_ == SEM_FAILED)
            throw std::runtime_error("Failed to create semaphores");

        if (local_rank_ == 0) {
            // Allocate a page-aligned block
            if (ftruncate(shm_id, SHM_SIZE_) == -1)
                throw std::runtime_error("Failed to allocate shared memory");

            // Map shared memory
            void *p = mmap(
                nullptr, SHM_SIZE_, PROT_READ | PROT_WRITE, 
                MAP_SHARED, shm_id, 0
            );
            if (p == MAP_FAILED)
                throw std::runtime_error("Failed to map shared memory");
            close(shm_id);
            shm_ = (TensorIPCShm *)p;
        
            // Initialize shared memory
            shm_->counter = 0;

            // Wake up other processes
            for (int i = 0; i < local_world_size_ - 1; i++)
                xsem_post(sem_ready_);
        } else {
            // Wait until initialized
            xsem_wait(sem_ready_);

            // Map shared memory
            void* p = mmap(
                nullptr, SHM_SIZE_, PROT_READ | PROT_WRITE, 
                MAP_SHARED, shm_id, 0
            );
            if (p == MAP_FAILED)
                throw std::runtime_error("Failed to map shared memory");
            close(shm_id);
            shm_ = (TensorIPCShm *)p;
        }

        // Ensure all processes reach here
        sync();

        // Unlink immediately
        if (local_rank_ == 0) {
            shm_unlink(SHM_KEY_);
            sem_unlink(SEM_COUNTER_KEY_);
            sem_unlink(SEM_ENTER_KEY_);
            sem_unlink(SEM_EXIT_KEY_);
            sem_unlink(SEM_READY_KEY_);
        }

        // Clean up
        sem_close(sem_ready_);
        sem_ready_ = nullptr;

        // Ensure all processes reach here
        sync();
    }

    __host__ inline ~TensorIPC() {
        if (sem_counter_) sem_close(sem_counter_);
        if (sem_enter_) sem_close(sem_enter_);
        if (sem_exit_) sem_close(sem_exit_);
        if (shm_) munmap(shm_, SHM_SIZE_);
    }

    __host__ inline void sync(int num_ranks = -1) {
        if (num_ranks == -1)
            num_ranks = local_world_size_;
        else if (num_ranks < 0 || num_ranks > local_world_size_)
            throw std::runtime_error("Invalid number of ranks");
    
        // Phase 1: arrive
        xsem_wait(sem_counter_);
        if (++shm_->counter == num_ranks) {
            for (int i = 0; i < num_ranks; i++)
                xsem_post(sem_enter_);
        }
        xsem_post(sem_counter_);
        xsem_wait(sem_enter_);
      
        // Phase 2: depart
        xsem_wait(sem_counter_);
        if (--shm_->counter == 0) {
            for (int i = 0; i < num_ranks; i++)
                xsem_post(sem_exit_);
        }
        xsem_post(sem_counter_);
        xsem_wait(sem_exit_);
    }

    __host__ inline void all_gather_ptrs(TensorIPCEntry *out_entries, void *src_ptr) {
        if (!src_ptr || !out_entries)
            throw std::runtime_error("Source and destination entries must be non-null");
    
        // Export IPC handle
        cudaIpcMemHandle_t export_handle;
        CUDACHECK(cudaIpcGetMemHandle(&export_handle, src_ptr));
    
        // Share IPC handle
        sync(); // ensure all processes are ready
        shm_->handle[local_rank_] = export_handle;
        sync();
    
        // Import IPC handle
        for (int i = 0; i < local_world_size_; i++) {
            if (i == local_rank_) {
                out_entries[i].is_imported = false;
                out_entries[i].raw_ptr = src_ptr;
            } else {
                out_entries[i].import_from_handle(shm_->handle[i]);
            }
        }
    }

    __host__ static inline void xsem_wait(sem_t* s) {
        while (sem_wait(s) == -1) {
            if (errno == EINTR)
                continue;
            throw std::runtime_error("Failed to wait on semaphore");
        }
    }
    
    __host__ static inline void xsem_post(sem_t* s) {
        if (sem_post(s) == -1)
            throw std::runtime_error("Failed to post on semaphore");
    }

    __host__ inline bool is_device_id_valid() const {
        int device_id;
        CUDACHECK(cudaGetDevice(&device_id));
        return device_id >= 0 && device_id < local_world_size_ && device_id == local_rank_;
    }

    __host__ inline bool is_tensor_ipc_supported() const {
        // Check if IPC is supported
        int ipc_supported;
        CUDACHECK(cudaDeviceGetAttribute(&ipc_supported, cudaDevAttrIpcEventSupport, local_rank_));
        return ipc_supported;
    }

    static inline constexpr int MAX_LOCAL_WORLD_SIZE_ = 72;
    static inline constexpr int PAGE_SIZE_ = 4096;

    // TODO: make these unique per process group
    static inline constexpr const char* SHM_KEY_ = "/tensor_shm";
    static inline constexpr const char* SEM_COUNTER_KEY_ = "/tensor_sem_counter";
    static inline constexpr const char* SEM_ENTER_KEY_ = "/tensor_sem_enter";
    static inline constexpr const char* SEM_EXIT_KEY_ = "/tensor_sem_exit";
    static inline constexpr const char* SEM_READY_KEY_ = "/tensor_sem_ready";

    struct TensorIPCShm {
        int counter;
        cudaIpcMemHandle_t handle[MAX_LOCAL_WORLD_SIZE_];
        __host__ inline TensorIPCShm() : counter(0) {}
    };
    static inline constexpr int SHM_SIZE_ = (sizeof(TensorIPCShm) + PAGE_SIZE_ - 1) / PAGE_SIZE_ * PAGE_SIZE_;

    int local_rank_;
    int local_world_size_;
    TensorIPCShm *shm_;
    sem_t *sem_counter_;
    sem_t *sem_enter_;
    sem_t *sem_exit_;
    sem_t *sem_ready_;
};

#endif // MULTI_GPU_CUH
