

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])


// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}
// Dot product
// grid(N/128),block(128)
// a:N*1 b:N*1 y=sum(elementwise_mul(a,b))
template<const int NUM_THREADS=128>
__global__ void dot(float* a,float* b,float* y,int N)
{
    int tid = threadIdx.x;
    int idx = tid + blockIdx.x * NUM_THREADS;
    constexpr int NUM_THREADS = (NUM_THREADS+WARP_SIZE-1)/WARP_SIZE;
    __shared__ float reduce_sum[NUM_THREADS];

    float prod = (idx<N)? a[idx]*b[idx] : 0.0f;

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    prod = warp_reduce_sum<WARP_SIZE>(prod);
  // warp leaders store the data to shared memory.
    if (lane == 0) reduce_smem[warp] = prod;
    __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
    prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) prod = warp_reduce_sum<NUM_WARPS>(prod);
    if (tid == 0) atomicAdd(y, prod);
}

// Dot product + vec4
// grid(N/128),block(128)
// a:N*1 b:N*1 y=sum(elementwise_mul(a,b))
template<const int NUM_THREADS=128/4>
__global__ void dot_vec4(float* a,float* b,float* y,int N)
{
    int tid = threadIdx.x;
    int idx = tid + blockIdx.x * NUM_THREADS;
    constexpr int NUM_WARPS = (NUM_THREADS+WARP_SIZE-1)/WARP_SIZE;
    __shared__ float reduce_sum[NUM_THREADS];

    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float prod = (idx<N)? (reg_a.x*reg_b.x+reg_a.y*reg_b.y_+reg_a.z*reg_b.z+reg_a.w*reg_b.w) : 0.0f;

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    prod = warp_reduce_sum<WARP_SIZE>(prod); 

    if(lane == 0) {
        reduce_sum[warp]=prod;
    }
    __syncthreads();
    // the first warp compute the final sum.  
    prod = (lane< NUM_WARPS) ? reduce_sum[lane]:0.0f;
    if(warp == 0) prod = warp_reduce_sum<NUM_WARPS>(prod);
    if(tid==0) {
        atomicAdd(y,prod);
    }
}