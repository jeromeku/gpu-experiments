#define REPEAT2(x) x; x;
#define REPEAT4(x) REPEAT2(x) REPEAT2(x)
#define REPEAT8(x) REPEAT4(x) REPEAT4(x)
#define REPEAT16(x) REPEAT8(x) REPEAT8(x)
#define REPEAT32(x) REPEAT16(x) REPEAT16(x)
#define REPEAT64(x) REPEAT32(x) REPEAT32(x)

struct node { node *ptr; };

static __device__ __inline__ uint32_t __clk ()
{
  uint32_t mclk;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(mclk) :: "memory");
  return mclk;
}

__global__ void pointer_chase (unsigned int n, node * __restrict__ head, int32_t *result)
{
  node *ptr = head;

  const int32_t begin = __clk ();

  REPEAT64(ptr = ptr->ptr)

  const int32_t end = __clk ();

  if (result)
    result[0] = (end - begin) / 64;

  head[n] = *ptr;
}