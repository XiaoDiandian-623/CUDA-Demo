#include <stdio.h>
#include <cmath>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"

#define warpSize 32

void softmaxCPU(float* input,float* output,int rows,int cols) 
{
    for(int j=0;j<rows;j++) {
        float MAX = 0;
        float totalSum = 0;
        for(int i=0;i<cols;i++) {
            MAX = max(MAX,input[j*cols+i]);
        }
        for(int i=0;i<cols;i++) {
            totalSum += exp(input[j*cols+i]-MAX);
        }
        for(int i=0;i<cols;i++) {
            output[j*cols+i] = exp(input[j*cols+i]-MAX) / totalSum;
        }
    }
}

//定义向量类型
template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
  T val[VecSize];
};
template<typename T>
struct SumOp{
__device__ __forceinline__ T operator()(const T& a,const T& b) const {
    return a+b;
}
};
template<typename T>
struct MaxOp{
__device__ __forceinline__ T operator()(const T& a,const T& b) const {
    return max(a,b);
}
};
template<template<typename> class ReductionOp,typename T,int warp_width=warpSize>
__inline__ __device__ T WarpReduce(T val) {
    for(int mask=warpSize/2;mask>0;mask>>1) {
        val = ReductionOp<T>(val,__shfl_xor_sync(0xffffffff,val,mask));
    }
}

template<typename T>
__inline__ __device__ T Exp(T x);
template<>
__inline__ __device__ float Exp<float>(float x) {
    return __expf(x);
    //return exp(x);
}

template<typename T>
__inline__ __device__ T Div(T a,T b);
template<>
__inline__ __device__ float Div<float>(float a,float b) {
    return __fdividef(a,b);
    //return a/ b;
}

template<typename T>
__inline__ __device__ T inf();
template<>
__inline__ __device__ float inf<float>()
{
    return 10000000000;
}

//向量化load，从src向量化加载第row行第col列的数据到dst
template<int VecSize>
__device__ void load(const float* src,const float* dst,int row,const int row_size,const int col) {
    using VecType = VectorType<float,VecSize>;
    const int offset = (row * row_size + col) / VecSize;
    *reinterpret_cast<VecType*>dst = *(reinterpret_cast<VecType*>(const_cast<float*>(src)) + offset);
}
//向量化store, 从src向量化写第row行第col列的数据到dst
 template<int VecSize>
 __device__ void store(float* dst,float* src,int row,const int row_size,const int col) {
    using VecType = VectorType<float,VecSize>;
    const int offset = (row*row_size + col) / VecSize;
    *(reinterpret_cast<VecType*>(dst)+offset) = *(reinterpret_cast<VecType*)(src);
 }

 template<int pack_size,int cols_per_thread,int warp_width,int rows_per_thread>
 __global__ void WarpSummax(const float* src,float *dst,const int rows,const int cols)
 {
    constexpr int num_packs = cols_per_thread / pack_size;
    assert(cols <= cols_per_thread*warp_width);
    float buf[rows_per_thread][cols_per_thread];

    const int global_warp_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int num_global_warp = gridDim.y * blockDim.y;
    const int lane_id = threadIdx.x;
    const int step = num_global_warp * rows_per_thread;

    for(int row=global_warp_id*rows_per_thread;row<rows;row+=step) {
        float thread_max[rows_per_thread];

        for(int row_id=0;row_id<rows_per_thread;++row_id) {
            thread_max[row_id] = -inf<float>();
            float* row_buf = buf[row_id];

            for(int pack_id=0;pack_id<num_packs;++pack_id) {
                const int pack_offset = pack_id * pack_size;
                const int col = (pack_id*warp_width+lane_id) * pack_size;
                if(col < cols) {
                    load<pack_size>(src,row_buf+pack_offset,row+row_id,cols,col);
                    for(int i=0;i<pack_size;++i) {
                        thread_max[row_id] = max(thread_max[row_id],row_buf[pack_offset+i]);
                    }
                } else {
                    for(int i=0;i<pack_size;++i) {
                        row_buf[i+pack_offset] = -inf<float>();
                    }
                }
            }

            float warp_max[rows_per_thread];
            for(int row_id=0;row_id<rows_per_thread;row_id++) {
                warp_max[row_id] = WarpReduce<MaxOp,float,warpSize>(thread_max[row_id]);
            }
            
            float thread_sum[rows_per_thread];
            for(int row_id=0;row_id<rows_per_thread;++row_id) {
                thread_sum[row_id] = 0;
                float* row_buf = buf[row_id];

                for(int i=0;i<cols_per_thread;i++) {
                    row_buf[i] = Exp(row_buf[i]-warp_max[row_id]);
                    thread_sum[row_id] += row_buf[i];
                }
                
            }

            float warp_sum[rows_per_thread];
            for(int row_id = 0;row_id<rows_per_thread;row_id++) {
                warp_sum[row_id] = WarpReduce<SumOp,float,warpSize>(thread_sum[row_id]);
            }

            for(int row_id=0;row_id<rows_per_thread;++row_id) {
                float* row_buf buf[row_id];
                for(int i=0;i<cols_per_thread;i++) {
                    row_buf[i] = Div(row_buf[i],warp_sum[row_id]);
                }
                for(int i=0;i<num_packs;i++) {
                    const int col = (i*warp_width+lane_id)*pack_size;
                    if(col<cols) {
                        store<pack_size>(dst,row_buf+i*pack_size,row+row_id,cols,col);
                    }
                }
            }
        }
    }
 }

 int main()
 {
    float millisecond=0;
 }