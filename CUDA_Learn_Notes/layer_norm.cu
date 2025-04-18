
// #define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

template <unsigned int WARP_SIZE = 32>
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
// Block All Reduce Sum
template<const int NUM_THREADS = 128>
__global__ void block_all_reduce_sum(float* a, float* y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    // 先将数据读入寄存器
    float sum = (idx < N) ? a[idx] : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    // warp内归约
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    // 每个warp的第0个线程将归约结果写入共享内存
    if (lane == 0) reduce_smem[warp] = sum;
    __syncthreads();
    // 第一个warp负责读取共享内存中的数据并归约
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(y, sum);
}
// Layer Norm: x: NxK(K=128<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template<const int NUM_THREADS=128>
__global__ void layer_norm(float* x,float* y,float g,float b,int N,int K)
{
    int tid = threadIdx.x;
    int idx = tid + blockIdx.x * NUM_THREADS;
    int bid = blockIdx.x;
    const float epsilon = 1e-5f;

    __shared__ float s_mean; // shared within block
    __shared__ float s_variance; // shared within block
    float value = (idx < N) ? x[idx] : 0.0f;
    float sum = block_reduce_sum<NUM_THREADS>(value);

    if(tid == 0) s_mean = sum / (float)K; // 均值
    __syncthreads();

    float variance = (value-s_mean) * (value-s_mean);
    variance = block_reduce_sum<NUM_THREADS>(variance);
    if(tid == 0) s_variance = rsqrtf(variance/(float)K+epsilon);
    __syncthreads();

    if(idx<N*K) y[idx] = ((value-s_mean)*s_variance)*g+b;

}

// LayerNorm + vec4
template<const int NUM_THREADS=128/4>
__global__ void layer_norm_vec4(float* x,float* y,float g,float b,int N,int K)
{
  int tid = threadIdx.x;
  int idx = tid + blockIdx.x * NUM_THREADS;
  int bid = blockIdx.x;
  const float epsilon = 1e-5f;
  __shared__ float s_mean; // shared within block
  __shared__ float s_variance; // shared within block

  float4 reg_x = FLOAT4(x[idx]);
  float value = (idx < N*K) ? (reg_x.x+reg_x.y+reg_x.z+reg_x.w) : 0.0f;
  float sum = block_reduce_sum_f32<NUM_THREADS>(value);
  if(tid == 0) s_mean = sum / (float)K;
  __syncthreads();

  float4 reg_x_hat;
  reg_x_hat.x = reg_x.x - s_mean;
  reg_x_hat.y = reg_x.y - s_mean;
  reg_x_hat.z = reg_x.z - s_mean;
  reg_x_hat.w = reg_x.w - s_mean;
  float variance = reg_x_hat.x*reg_x_hat.x + reg_x_hat.y*reg_x_hat.y + reg_x_hat.z*reg_x_hat.z + reg_x_hat.w*reg_x_hat.w;
  variance = block_reduce_sum_f32<NUM_THREADS>(variance);
  if(tid == 0) s_variance = rsqrtf(variance.(float)K+epsilon);
  __syncthreads();

  float4 reg_y ;
  reg_y.x = reg_x_hat.x * s_variance * + b;
  reg_y.y = reg_x_hat.y * s_variance * + b;
  reg_y.z = reg_x_hat.z * s_variance * + b;
  reg_y.w = reg_x_hat.w * s_variance * + b;

  if(idx < N*K) FLOAT4(y[idx]) = reg_y;
}


