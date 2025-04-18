#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include <cmath>

#define WarpSize 32

bool CheckResult(float *out, float* groudtruth, int N){
    for (int i = 0; i < N; i++){
      if(i == 0){
        printf("1st comparsion: %f and %f \n" , out[i], groudtruth[i] );
      }
      if (out[i] != groudtruth[i]) {
          return false;
      }
    }
    return true;
}
// softmax��ʽ
// e^(xi - max(xi)) / sigma(e^(xi - max(xi)))
void softmaxCPU(float* input, float* result, int rows, int cols){
  for (int j = 0; j < rows; j++)
  {
    float total = 0;
    float MAX = 0;
    for(int i = 0; i < cols; i++)
    {
      MAX = max(input[j * cols + i], MAX);
    }
    for(int i = 0; i < cols; i++)
    {
      total += exp(input[j * cols + i] - MAX);
    }
    for(int i = 0; i < cols; i++)
    {
      result[j * cols + i] = exp(input[j * cols + i] - MAX) / total;
    }
  }
}
// �����������ͣ�����������ΪVecSize
// ��: ����float����������ΪVectorType<float, 4>
template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
  T val[VecSize];
};

template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const 
    { return a + b; }
};

template<typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const 
    { return max(a, b); }
};

//ģ��ʵ���� warp_reduce : ReductionOp(max or sum )
template<template<typename> class ReductionOp, typename T, int warp_width = WarpSize>
__inline__ __device__ T WarpReduce(T val) {
  for (int mask = warp_width / 2; mask > 0; mask /= 2) {
    val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template<typename T>
__inline__ __device__ T Exp(T x);
// ������exp fp32
template<>
__inline__ __device__ float Exp<float>(float x) {
  //return __expf(x);//fast math
  return exp(x);
}

template<typename T>
__inline__ __device__ T Inf();
template<>
__inline__ __device__ float Inf<float>() {
  return 10000000000;
}

template<typename T>
__inline__ __device__ T Div(T a, T b);
template<>
__inline__ __device__ float Div<float>(float a, float b) {
  //return __fdividef(a, b);//fast math
  return a / b;
}

// ������������ݵĲ�������src���������ص�row�е�col�е����ݵ�dst
template<int VecSize>
__device__ void load(const float* src, float* dst, int row, const int row_size, const int col) {
  using VecType = VectorType<float, VecSize>;
  const int offset = (row * row_size + col) / VecSize;
  *reinterpret_cast<VecType*>(dst) = *(reinterpret_cast<VecType*>(const_cast<float*>(src)) + offset);
}

// ������������ݵĲ�������src������д��row�е�col�е����ݵ�dst
template<int VecSize>
__device__ void store(float* dst, float* src, int row, const int row_size, const int col) {
  using VecType = VectorType<float, VecSize>;
  const int offset = (row * row_size + col) / VecSize;
  *(reinterpret_cast<VecType*>(dst) + offset) = *reinterpret_cast<VecType*>(src);
}

// 1, 1024/32,32, 1
template<int pack_size, int cols_per_thread,
         int warp_width, int rows_per_thread>
__global__ void WarpSoftmax(const float* src, float* dst, const int rows, const int cols) {
  constexpr int num_packs = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * warp_width);
  float buf[rows_per_thread][cols_per_thread];
  //��ǰwarp������warp�е�id�ţ���Ϊÿ�б�ʾһ��warp������ֻ������кţ���global warp id
  const int global_warp_id = blockIdx.y * blockDim.y + threadIdx.y;
  const int num_global_warp = gridDim.y * blockDim.y; // 125 * 8 = 1000, ��src.rows()ƥ��
  const int lane_id = threadIdx.x;
  const int step = num_global_warp * rows_per_thread; // 1000 һ��warp���ܴ������
  // ���뵽��ǰ�����������block��������ֵ����Χ
  for (int row = global_warp_id * rows_per_thread; row < rows; row += step) {
    float thread_max[rows_per_thread];
    // ϸ���Ȼ������뵽ÿ���߳��������������Χ---������һ�е����� ���ֵ
    for (int row_id = 0; row_id < rows_per_thread; ++row_id) {
      thread_max[row_id] = -Inf<float>();
      float* row_buf = buf[row_id];
      // ��ϸ����һ�㣬���뵽ÿ���߳��������һ�еĶ��������Χ ---�������������Χ�ڵ�����
      for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
       
        const int pack_offset = pack_id * pack_size; // ÿ����������ʼƫ��
        const int col = (pack_id * warp_width + lane_id) * pack_size; // ��ǰ�������ڵ���ʼ�к�
        if (col < cols) {
          // ������ʼ�кţ���ȡ��ǰ������row_buf�Ĵ���
          load<pack_size>(src, row_buf + pack_offset, row + row_id, cols, col);
          // ���pack  local��thread local�����ֵ
          for (int i = 0; i < pack_size; ++i) {
            thread_max[row_id] = max(thread_max[row_id], row_buf[pack_offset + i]);
          }
        } else {
          for (int i = 0; i < pack_size; ++i) { row_buf[pack_offset + i] = -Inf<float>(); }  //��ʼ�кų���������������Ϊ�������softmaxֵ��Ӱ��
        }
      }
    }
   
    float warp_max[rows_per_thread]; // ����rows_per_thread���Ĵ������浱ǰ�̼߳�����е����ֵ
    for (int row_id = 0; row_id < rows_per_thread; ++row_id) { // reduce�����̼߳�������ֵ���ó������߳��е����ֵ����һ�е����ֵ
      warp_max[row_id] = WarpReduce<MaxOp, float, warp_width>(thread_max[row_id]);
    }

    // ����rows_per_thread���Ĵ������浱ǰ�̼߳�����е��ܺͣ���softmax��ĸ
    float thread_sum[rows_per_thread];
    for (int row_id = 0; row_id < rows_per_thread; ++row_id) {
      thread_sum[row_id] = 0;
      float* row_buf = buf[row_id];
      // ��ǰ�߳�ӵ�е�row_bufֵ���ܺͣ�softmax��ĸ��partial value
      for (int i = 0; i < cols_per_thread; ++i) {
        row_buf[i] = Exp(row_buf[i] - warp_max[row_id]);
        thread_sum[row_id] += row_buf[i];
      }
    }
    float warp_sum[rows_per_thread];
    // softmax��ĸ��final value
    for (int row_id = 0; row_id < rows_per_thread; ++row_id) {
      warp_sum[row_id] = WarpReduce<SumOp, float, warp_width>(thread_sum[row_id]);
    }

    for (int row_id = 0; row_id < rows_per_thread; ++row_id) {
      float* row_buf = buf[row_id];
      // ���ӳ���ĸ�õ�sfotmax���ս��
      for (int i = 0; i < cols_per_thread; ++i) {
        row_buf[i] = Div(row_buf[i], warp_sum[row_id]);
      }
      // ������������ȥ�������ս��д���Դ�
      for (int i = 0; i < num_packs; ++i) {
        const int col = (i * warp_width + lane_id) * pack_size;
        if (col < cols) {
          store<pack_size>(dst, row_buf + i * pack_size, row + row_id, cols, col);
        }
      }
    }
  }
}

int main(){
    float milliseconds = 0;
    const int N = 1000 * 1024; // 1000�� ÿ��1024������
    float *src = (float *)malloc(N * sizeof(float));
    float *d_src;
    cudaMalloc((void **)&d_src, N * sizeof(float));

    //int gridSize = ;//2d block, blockx=32,blocky=num warps in a block,griddimy=block nums
    //int blockSize = 256;
    float *dst = (float*)malloc(N * sizeof(float));
    float *d_dst;
    cudaMalloc((void **)&d_dst, N * sizeof(float));
    float *groudtruth = (float *)malloc(N * sizeof(float));

    for(int i = 0; i < N; i++){
        src[i] = 1;
    }

    softmaxCPU(src, groudtruth, 1000, 1024);

    cudaMemcpy(d_src, src, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(1, 125);//y��125��block,
    dim3 Block(32, 8);//x��32��threads���һ��warp����һ��,y��8��threads,8*125=1000��
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    WarpSoftmax<1, 1024 / 32, 32, 1><<<Grid, Block>>>(d_src, d_dst, 1000, 1024);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(dst, groudtruth, N);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i=0;i<10;i++){
            printf("%lf ",dst[i]);
        }
        printf("\n");
    }
    printf("WarpSoftmax latency = %f ms\n", milliseconds);

    cudaFree(d_src);
    cudaFree(d_dst);
    free(src);
    free(dst);
    free(groudtruth);
}