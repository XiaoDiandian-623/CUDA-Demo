#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// -------------------------------------- FP32 -------------------------------------- 
// DS required for Online Softmax
struct __align__(8) MD { float m; float d; }; 
// Warp Reduce for Online Softmax
template<const int kWarpSize = WARP_SIZE >
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
  unsigned int mask = 0xffffffff;
  #pragma unroll
  for(int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
    MD other;
    other.m = __shfl_xor_sync(mask, value.m, stride);
    other.d = __shfl_xor_sync(mask, value.d, stride);

    bool value_bigger = (value.m > other.m);
    MD bigger_m = value_bigger ? value : other;
    MD smaller_m = value_bigger ? other : value;
    
    value.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    value.m = bigger_m.m;
  }
  return value;
}

// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Warp Reduce Max
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// grid 1D block 1D, grid(N/256), block(256)
template<const int NUM_THREADS=256>
__device__ float block_reduce_sum_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  float value = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  value = warp_reduce_sum_f32<NUM_WARPS>(value);  
  // WRAN: need to broadcast value to all threads within warp
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

template<const int NUM_THREADS=256>
__device__ float block_reduce_max_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  float value = warp_reduce_max_f32<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
  value = warp_reduce_max_f32<NUM_WARPS>(value);
  // WRAN: need to broadcast value to all threads within warp
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

// Softmax x: N, y: N
// grid(N/256), block(K=256)
template<const int NUM_THREADS = 256>
__global__ void softmax_f32_kernel(float* x, float* y, float* total, int N) {
  
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid; 
  
  float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, exp_sum);
  __threadfence(); // grid level memory fence
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  // printf("N: %d, idx: %d, bid: %d, tid: %d, exp_val: %f, exp_sum: %f, total: %f\n", 
  //         N,     idx, blockIdx.x,  tid,     exp_val,     exp_sum,     *total);
  if (idx < N) y[idx] = exp_val / (*total); 
}

// Softmax Vec4 x: N, y: N
// grid(N/256), block(256/4)
template<const int NUM_THREADS = 256/4>
__global__ void softmax_f32x4_kernel(float* x, float* y, float* total, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 4; 
  
  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_exp;
  reg_exp.x = (idx + 0 < N) ? expf(reg_x.x) : 0.0f;
  reg_exp.y = (idx + 1 < N) ? expf(reg_x.y) : 0.0f;
  reg_exp.z = (idx + 2 < N) ? expf(reg_x.z) : 0.0f;
  reg_exp.w = (idx + 3 < N) ? expf(reg_x.w) : 0.0f;
  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, exp_sum);
  __threadfence(); // grid level memory fence
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx + 3 < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / (*total);
    reg_y.y = reg_exp.y / (*total);
    reg_y.z = reg_exp.z / (*total);
    reg_y.w = reg_exp.w / (*total);
    FLOAT4(y[idx]) = reg_y; 
  }
}

// NOTE: softmax per-token
// Softmax x: (S,h), y: (S,h)
// grid(S*h/h), block(h), assume h<=1024
// one token per thread block, only support 64<=h<=1024 and 2^n
// HEAD_SIZE/KV_LEN=NUM_THREADS
template<const int NUM_THREADS = 256>
__global__ void softmax_f32_per_token_kernel(float* x, float* y, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid; 
  
  float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  // printf("N: %d, idx: %d, tid: %d, exp_val: %f, exp_sum: %f\n", 
  //         N, idx, tid, exp_val, exp_sum);
  if (idx < N) y[idx] = exp_val / exp_sum;
}

template<const int NUM_THREADS = 256/4>
__global__ void softmax_f32x4_per_token_kernel(float* x, float* y, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 4;

  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_exp;
  reg_exp.x = (idx + 0 < N) ? expf(reg_x.x) : 0.0f;
  reg_exp.y = (idx + 1 < N) ? expf(reg_x.y) : 0.0f;
  reg_exp.z = (idx + 2 < N) ? expf(reg_x.z) : 0.0f;
  reg_exp.w = (idx + 3 < N) ? expf(reg_x.w) : 0.0f;

  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx + 3 < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / (exp_sum);
    reg_y.y = reg_exp.y / (exp_sum);
    reg_y.z = reg_exp.z / (exp_sum);
    reg_y.w = reg_exp.w / (exp_sum);
    FLOAT4(y[idx]) = reg_y; 
  }
}

// safe_softmax per token
template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f32_per_token_kernel(float* x, float* y, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid; 
  
  float val = (idx < N) ? x[idx] : -FLT_MAX;
  float max_val = block_reduce_max_f32<NUM_THREADS>(val); // block max
  float exp_val = (idx < N) ? expf(x[idx] - max_val) : 0.0f;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx < N) y[idx] = exp_val / exp_sum; 
}

template<const int NUM_THREADS = 256/4>
__global__ void safe_softmax_f32x4_per_token_kernel(float* x, float* y, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 4;

  float4 reg_x = FLOAT4(x[idx]);
  reg_x.x = (idx + 0 < N) ? reg_x.x : -FLT_MAX;
  reg_x.y = (idx + 1 < N) ? reg_x.y : -FLT_MAX;
  reg_x.z = (idx + 2 < N) ? reg_x.z : -FLT_MAX;
  reg_x.w = (idx + 3 < N) ? reg_x.w : -FLT_MAX;
  float val =      reg_x.x;
  val = fmaxf(val, reg_x.y);
  val = fmaxf(val, reg_x.z);
  val = fmaxf(val, reg_x.w);
  float max_val = block_reduce_max_f32<NUM_THREADS>(val); // block max

  float4 reg_exp;
  reg_exp.x = (idx + 0 < N) ? expf(reg_x.x - max_val) : 0.0f;
  reg_exp.y = (idx + 1 < N) ? expf(reg_x.y - max_val) : 0.0f;
  reg_exp.z = (idx + 2 < N) ? expf(reg_x.z - max_val) : 0.0f;
  reg_exp.w = (idx + 3 < N) ? expf(reg_x.w - max_val) : 0.0f;

  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx + 3 < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / (exp_sum);
    reg_y.y = reg_exp.y / (exp_sum);
    reg_y.z = reg_exp.z / (exp_sum);
    reg_y.w = reg_exp.w / (exp_sum);
    FLOAT4(y[idx]) = reg_y; 
  }
}

template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f16_f32_per_token_kernel(half* x, half* y, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid; 
  
  float val = (idx < N) ? __half2float(x[idx]) : -FLT_MAX;
  float max_val = block_reduce_max_f32<NUM_THREADS>(val); // block max
  float exp_val = (idx < N) ? expf(val - max_val) : 0.0f;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx < N) y[idx] = __float2half_rn(exp_val / exp_sum); 
}

template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x2_f32_per_token_kernel(half* x, half* y, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 2; 
  
  float2 reg_x = __half22float2(HALF2(x[idx]));
  float max_val = -FLT_MAX;
  max_val = ((idx + 0) < N) ? fmaxf(reg_x.x, max_val): -FLT_MAX;
  max_val = ((idx + 1) < N) ? fmaxf(reg_x.y, max_val): -FLT_MAX;
  max_val = block_reduce_max_f32<NUM_THREADS>(max_val); // block max

  float2 reg_exp;
  reg_exp.x = ((idx + 0) < N) ? expf(reg_x.x - max_val) : 0.0f;
  reg_exp.y = ((idx + 1) < N) ? expf(reg_x.y - max_val) : 0.0f;

  float exp_val = reg_exp.x + reg_exp.y;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum

  float2 reg_y;
  reg_y.x = reg_exp.x / (exp_sum);
  reg_y.y = reg_exp.y / (exp_sum);

  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if ((idx + 1) < N) HALF2(y[idx]) = __float22half2_rn(reg_y); 
}

template<const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x8_pack_f32_per_token_kernel(half* x, half* y, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 8; 
  // temporary register(memory), .local space in ptx, addressable
  half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits
  
  float max_val = -FLT_MAX;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    max_val = fmaxf(__half2float(pack_x[i]), max_val);
  }
  max_val = block_reduce_max_f32<NUM_THREADS>(max_val); // block max

  float exp_sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float exp_val = expf(__half2float(pack_x[i]) - max_val);
    exp_sum += (((idx + i) < N) ? exp_val : 0.0f);
  }
  exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_sum); // block sum

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    // e^x_i/sum(e^x_0,...,e^x_n-1) 
    float exp_val = expf(__half2float(pack_x[i]) - max_val);
    pack_y[i] = __float2half_rn(exp_val / exp_sum);
  }
  // reinterpret as float4 and store 128 bits in 1 memory issue.
  if ((idx + 7) < N) { LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]); }
  // TODO: support non 8-multiple K here
}

template<const int NUM_THREADS = 256 >
__global__ void online_safe_softmax_f32_per_token_kernel(const float* x, float* y, int N) {
  // reference: https://arxiv.org/pdf/1805.02867 (Online normalizer calculation for softmax)
  int local_tid = threadIdx.x;
  int global_tid = blockIdx.x * NUM_THREADS + threadIdx.x;
  const int WAPR_NUM = NUM_THREADS / WARP_SIZE;
  int warp_id = local_tid / WARP_SIZE;
  int lane_id = local_tid % WARP_SIZE;
  MD val;
  val.m = global_tid < N ? x[global_tid] : -FLT_MAX;
  val.d = global_tid < N ? 1.0f : 0.0f;

  __shared__ MD shared[WAPR_NUM]; 
  MD res = warp_reduce_md_op<WARP_SIZE>(val);

  if (lane_id == 0) shared[warp_id] = res; 
  __syncthreads();

  if (local_tid < WARP_SIZE) {
    MD block_res = shared[local_tid];
    block_res = warp_reduce_md_op<WAPR_NUM>(block_res); 
    if (local_tid == 0) {
      shared[0] = block_res; 
    }
  }
  __syncthreads();

  MD final_res = shared[0];
  float d_total_inverse = __fdividef(1.0f, final_res.d);
  if (global_tid < N) {
    y[global_tid] = __expf(x[global_tid] - final_res.m) * d_total_inverse;
  }
}

template <const int NUM_THREADS = 256 / 4>
__global__ void online_safe_softmax_f32x4_pack_per_token_kernel(float *x, float *y, int N)
{
    // reference: https://arxiv.org/pdf/1805.02867 (Online normalizer calculation for softmax)
    int local_tid = threadIdx.x;
    int global_tid = (blockIdx.x * NUM_THREADS + local_tid) * 4;

    const int WAPR_NUM = NUM_THREADS / WARP_SIZE;
    int warp_id = local_tid / WARP_SIZE;
    int lane_id = local_tid % WARP_SIZE;
    // compare local max value
    float4 val = FLOAT4((x)[global_tid]);
    float local_m = fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w));
    float local_d = __expf(val.x - local_m) + __expf(val.y - local_m) + __expf(val.z - local_m) + __expf(val.w - local_m);

    
    MD local_md = {local_m, local_d};
    MD res = warp_reduce_md_op<WARP_SIZE>(local_md);
    __shared__ MD shared[WAPR_NUM];

    if (lane_id == 0) shared[warp_id] = res;
    __syncthreads();
    // do block reduce
    if (local_tid < WARP_SIZE)
    {
        MD block_res = shared[local_tid];
        block_res = warp_reduce_md_op<WAPR_NUM>(block_res);
        if (local_tid == 0) shared[0] = block_res;
    }
    __syncthreads();
    // write back
    MD final_res = shared[0];
    float d_total_inverse = __fdividef(1.0f, final_res.d);
    if (global_tid < N)
    {
        float4 reg_y;
        reg_y.x = __expf(val.x - final_res.m) * d_total_inverse;
        reg_y.y = __expf(val.y - final_res.m) * d_total_inverse;
        reg_y.z = __expf(val.z - final_res.m) * d_total_inverse;
        reg_y.w = __expf(val.w - final_res.m) * d_total_inverse;
        FLOAT4((y)[global_tid]) = reg_y;
    }
}


// 验证函数
void cpu_softmax(float* x, float* y, int N) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; ++i) 
        if (x[i] > max_val) max_val = x[i];
    
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) 
        sum += expf(x[i] - max_val);
    
    for (int i = 0; i < N; ++i) 
        y[i] = expf(x[i] - max_val) / sum;
}

bool validate(float* ref, float* test, int N, float eps=1e-4) {
    for (int i = 0; i < N; ++i) {
        if (fabs(ref[i] - test[i]) > eps) {
            printf("Mismatch at %d: ref=%.5f test=%.5f\n", i, ref[i], test[i]);
            return false;
        }
    }
    return true;
}

// 时间测量宏
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("%s in %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define TIME_START(name) cudaEvent_t name##_start, name##_stop; \
                         CHECK_CUDA(cudaEventCreate(&name##_start)); \
                         CHECK_CUDA(cudaEventCreate(&name##_stop)); \
                         CHECK_CUDA(cudaEventRecord(name##_start))

#define TIME_END(name) CHECK_CUDA(cudaEventRecord(name##_stop)); \
                       CHECK_CUDA(cudaEventSynchronize(name##_stop)); \
                       float name##_ms; \
                       CHECK_CUDA(cudaEventElapsedTime(&name##_ms, name##_start, name##_stop)); \
                       printf("%s: %.3f ms\n", #name, name##_ms); \
                       CHECK_CUDA(cudaEventDestroy(name##_start)); \
                       CHECK_CUDA(cudaEventDestroy(name##_stop))

void test_softmax(int N, int repeat=100) {
    float *h_x = new float[N];
    float *h_ref = new float[N];
    float *h_test = new float[N];
    half *h_x_half = new half[N];
    
    for (int i = 0; i < N; ++i) {
        h_x[i] = (float)rand() / RAND_MAX;  // [0,1)
        h_x_half[i] = __float2half_rn(h_x[i]);
    }
    
    // CPU参考结果
    cpu_softmax(h_x, h_ref, N);
    
    // 设备内存
    float *d_x, *d_y, *d_total;
    half *d_x_half, *d_y_half;
    CHECK_CUDA(cudaMalloc(&d_x, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_total, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x_half, N*sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_y_half, N*sizeof(half)));
    
    // 拷贝数据
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x_half, h_x_half, N*sizeof(half), cudaMemcpyHostToDevice));

    // 测试各个内核
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    // 1. Naive FP32
    TIME_START(f32);
    for (int i = 0; i < repeat; ++i) {
        CHECK_CUDA(cudaMemset(d_total, 0, sizeof(float)));
        softmax_f32_kernel<<<grid, block>>>(d_x, d_y, d_total, N);
    }
    TIME_END(f32);
    CHECK_CUDA(cudaMemcpy(h_test, d_y, N*sizeof(float), cudaMemcpyDeviceToHost));
    if (!validate(h_ref, h_test, N)) printf("FP32 Validation Failed!\n");
    
    // 2. Vectorized FP32x4
    dim3 block4(256/4);
    dim3 grid4((N + block4.x*4 - 1) / (block4.x*4));
    TIME_START(f32x4);
    for (int i = 0; i < repeat; ++i) {
        CHECK_CUDA(cudaMemset(d_total, 0, sizeof(float)));
        softmax_f32x4_kernel<<<grid4, block4>>>(d_x, d_y, d_total, N);
    }
    TIME_END(f32x4);
    CHECK_CUDA(cudaMemcpy(h_test, d_y, N*sizeof(float), cudaMemcpyDeviceToHost));
    if (!validate(h_ref, h_test, N)) printf("FP32x4 Validation Failed!\n");
    
    // 3. Safe FP32 per-token
    TIME_START(safe_f32);
    for (int i = 0; i < repeat; ++i)
        safe_softmax_f32_per_token_kernel<<<1, N>>>(d_x, d_y, N);
    TIME_END(safe_f32);
    CHECK_CUDA(cudaMemcpy(h_test, d_y, N*sizeof(float), cudaMemcpyDeviceToHost));
    if (!validate(h_ref, h_test, N)) printf("Safe FP32 Validation Failed!\n");
    
    // 4. Online FP32
    TIME_START(online_f32);
    for (int i = 0; i < repeat; ++i)
        online_safe_softmax_f32_per_token_kernel<<<1, N>>>(d_x, d_y, N);
    TIME_END(online_f32);
    CHECK_CUDA(cudaMemcpy(h_test, d_y, N*sizeof(float), cudaMemcpyDeviceToHost));
    if (!validate(h_ref, h_test, N)) printf("Online FP32 Validation Failed!\n");
    
    // 5. FP16 Kernel
    TIME_START(safe_f16);
    for (int i = 0; i < repeat; ++i)
        safe_softmax_f16_f32_per_token_kernel<<<1, N>>>(d_x_half, d_y_half, N);
    TIME_END(safe_f16);
    CHECK_CUDA(cudaMemcpy(h_test, d_y_half, N*sizeof(half), cudaMemcpyDeviceToHost));
    // 转换FP16到FP32比较
    std::vector<float> h_test_fp32(N);
    for (int i = 0; i < N; ++i) 
        h_test_fp32[i] = __half2float(h_test[i]);
    if (!validate(h_ref, h_test_fp32.data(), N)) printf("FP16 Validation Failed!\n");

    // 清理
    delete[] h_x;
    delete[] h_ref;
    delete[] h_test;
    delete[] h_x_half;
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_total));
    CHECK_CUDA(cudaFree(d_x_half));
    CHECK_CUDA(cudaFree(d_y_half));
}

int main() {
    int sizes[] = {256, 1024, 4096, 16384};
    for (int size : sizes) {
        printf("\nTesting size: %d\n", size);
        test_softmax(size);
    }
    return 0;
}