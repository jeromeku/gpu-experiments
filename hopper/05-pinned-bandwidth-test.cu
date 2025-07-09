#include "multi-gpu.cuh"

// Shows that pinned memory is faster than pageable memory
// Also shows example of using cudaEventElapsedTime to measure bandwidth and time
// From NVIDIA: You should not over-allocate pinned memory. Doing so can reduce overall system performance because it reduces the amount of physical memory available to the operating system and other programs
// How much is too much is difficult to tell in advance, so as with all optimizations, test your applications and the systems they run on for optimal performance parameters

void profileCopies(float        *h_a, 
                   float        *h_b, 
                   float        *d, 
                   unsigned int  n,
                   char         *desc)
{
  printf("\n%s transfers\n", desc);

  unsigned int size = n * sizeof(float);

  // events for timing
  cudaEvent_t startEvent, stopEvent; 

  CUDACHECK( cudaEventCreate(&startEvent) );
  CUDACHECK( cudaEventCreate(&stopEvent) );

  CUDACHECK( cudaEventRecord(startEvent, 0) );
  CUDACHECK( cudaMemcpy(d, h_a, size, cudaMemcpyHostToDevice) );
  CUDACHECK( cudaEventRecord(stopEvent, 0) );
  CUDACHECK( cudaEventSynchronize(stopEvent) );

  float time; // milliseconds
  CUDACHECK( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("  Host to Device bandwidth (GiB/s): %f\n", size / (1024 * 1024 * 1024 * 1e0) / (time / 1e3));

  CUDACHECK( cudaEventRecord(startEvent, 0) );
  CUDACHECK( cudaMemcpy(h_b, d, size, cudaMemcpyDeviceToHost) );
  CUDACHECK( cudaEventRecord(stopEvent, 0) );
  CUDACHECK( cudaEventSynchronize(stopEvent) );

  CUDACHECK( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("  Device to Host bandwidth (GiB/s): %f\n", size / (1024 * 1024 * 1024 * 1e0) / (time / 1e3));

  for (int i = 0; i < n; ++i) {
    if (h_a[i] != h_b[i]) {
      printf("*** %s transfers failed ***\n", desc);
      break;
    }
  }

  // clean up events
  CUDACHECK( cudaEventDestroy(startEvent) );
  CUDACHECK( cudaEventDestroy(stopEvent) );
}




int main()
{
  int dev_id = 0; // we use only one GPU
  unsigned int n_elem = 4 * 1024 * 1024;
  const unsigned int size = n_elem * sizeof(float);

  // host arrays
  float *h_a_pageable, *h_b_pageable;   
  float *h_a_pinned, *h_b_pinned;

  // device array
  float *d_a;

  // allocate and initialize
  h_a_pageable = (float*)malloc(size);                    // host pageable
  h_b_pageable = (float*)malloc(size);                    // host pageable
  CUDACHECK( cudaMallocHost((void**)&h_a_pinned, size) ); // host pinned
  CUDACHECK( cudaMallocHost((void**)&h_b_pinned, size) ); // host pinned
  CUDACHECK( cudaMalloc((void**)&d_a, size) );            // device

  // A = [0, 1, 2, 3, ...]
  for (int i = 0; i < n_elem; ++i) h_a_pageable[i] = i;      
  memcpy(h_a_pinned, h_a_pageable, size);

  // B = [0, 0, 0, 0, ...]
  memset(h_b_pageable, 0, size);
  memset(h_b_pinned, 0, size);

  // output device info and transfer size
  cudaDeviceProp prop;
  CUDACHECK( cudaGetDeviceProperties(&prop, dev_id) );

  printf("\nDevice: %s\n", prop.name);
  printf("Transfer size (MiB): %d\n", size / (1024 * 1024));

  // perform copies and report bandwidth
  profileCopies(h_a_pageable, h_b_pageable, d_a, n_elem, "Pageable");
  profileCopies(h_a_pinned, h_b_pinned, d_a, n_elem, "Pinned");

  // cleanup
  cudaFree(d_a);
  cudaFreeHost(h_a_pinned);
  cudaFreeHost(h_b_pinned);
  free(h_a_pageable);
  free(h_b_pageable);

  printf("Done\n");

  return 0;
}
