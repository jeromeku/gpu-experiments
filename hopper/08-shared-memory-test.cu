// A thread’s execution can only proceed past a __syncthreads() after all threads in its block have executed the __syncthreads()

#include "multi-gpu.cuh"

__global__ void staticReverse(int *d, int n)
{
  // Static shared memory
  // note how global memory access can be coalesced thanks to shared memory
  // The only potential problem is bank conflicts
  __shared__ int s[64];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

__global__ void dynamicReverse(int *d, int n)
{
  // Dynamic shared memory
  // Amount of shared memory is known at runtime, specified by the third parameter of <<<..., ..., size >>>
  extern __shared__ int ds[]; // when declared like this, size is implicitly determined from the third parameter of <<<...>>>
                             // If you need multiple shared dynamic memory, you should divide the above array
  int t = threadIdx.x;
  int tr = n-t-1;
  ds[t] = d[t];
  __syncthreads();
  d[t] = ds[tr];
}

/*
  Bank conflicts
  - If multiple threads access addresses that map to the same bank, accesses are serialized
  - One exception is when different threads access the same address -> broadcast
  - Bank bandwidth is 32 bits per bank per cycle
  - Successive 32-bit words are assigned to successive banks
  - 32 banks. Bank size is configurable
*/

int main(void)
{
  const int n = 64;
  int a[n], r[n], d[n];
  
  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
  }

  int *d_d;
  cudaMalloc(&d_d, n * sizeof(int)); 
  
  // run version with static shared memory
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  staticReverse<<<1,n>>>(d_d, n);
  cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
  
  // run dynamic shared memory version
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  dynamicReverse<<<1,n,n*sizeof(int)>>>(d_d, n);
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
}
