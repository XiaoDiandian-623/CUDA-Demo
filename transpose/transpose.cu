#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <math.h>
#include <iostream>

#define Ceil(a, b) (((a) + (b) - 1) / (b))
#define FETCH_CFLOAT4(p) (reinterpret_cast<const float4*>(&(p))[0])
#define FETCH_FLOAT4(p) (reinterpret_cast<float4*>(&(p))[0])

// row major
// question: memory bound
// 把原矩阵中的(y,x)位置元素放到转置后的(x,y)位置
__global__ void mat_transpose_kernel_v0(const float* idata,float* odata,int M,int N )
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if(y<M && x<N) {
        odata[x*M+y] = idata[y*N+x]; 
    }
}

void mat_transpose_v0(const float* idata, float* odata, int M, int N) {
    constexpr int BLOCK_SZ = 16;
    dim3 block(BLOCK_SZ, BLOCK_SZ);
    dim3 grid((N + BLOCK_SZ - 1) / BLOCK_SZ, (M + BLOCK_SZ - 1) / BLOCK_SZ);
    mat_transpose_kernel_v0<<<grid, block>>>(idata, odata, M, N);
}
// 引入shared mem 
template<int BLOCK_SZ>
__global__ void mat_transpose_kernel_v1(const float* idata,float* odata,int M,int N)
{
    const int bx = blockIdx.x,by = blockIdx.y;
    const int tx = threadIdx.x,ty = threadIdx.y;

    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ];

    int x = bx * BLOCK_SZ + tx;
    int y = by * BLOCK_SZ + ty;

    if(y<M && x<N) {
        sdata[ty][tx] = idata[y*N+x];
    }
    __syncthreads();

    // 索引重计算
    x = by * BLOCK_SZ + tx;
    y = bx * BLOCK_SZ + ty;
    if(y<N && x<M) {
        odata[y*M+x] = sdata[tx][ty];
    }

}
void mat_transpose_v1(const float* idata, float* odata, int M, int N) {
    constexpr int BLOCK_SZ = 16;
    dim3 block(BLOCK_SZ, BLOCK_SZ);
    dim3 grid(Ceil(N, BLOCK_SZ), Ceil(M, BLOCK_SZ));
    mat_transpose_kernel_v1<BLOCK_SZ><<<grid, block>>>(idata, odata, M, N);
}

// 解决shared mem中的bank conflict-----padding
template<int BLOCK_SZ>
__global__ void mat_transpose_kernel_v2(const float* idata,float* odata,int M,int N)
{
    const int bx = blockIdx.x,by = blockIdx.y;
    const int tx = threadIdx.x,ty = threadIdx.y;

    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ+1];

    int x = bx * BLOCK_SZ + tx;
    int y = by * BLOCK_SZ + ty;
    if(y < M && x < N) {
        sdata[ty][tx] = idata[y*N+x];
    }
    __syncthreads();

    x = by * BLOCK_SZ + tx;
    y = bx * BLOCK_SZ + ty;
    if(y<N && x<M) {
        odata[y*M+x] = sdata[tx][ty];
    }
}
void mat_transpose_v2(const float* idata, float* odata, int M, int N) {
    constexpr int BLOCK_SZ = 16;
    dim3 block(BLOCK_SZ, BLOCK_SZ);
    dim3 grid(Ceil(N, BLOCK_SZ), Ceil(M, BLOCK_SZ));
    mat_transpose_kernel_v2<BLOCK_SZ><<<grid, block>>>(idata, odata, M, N);
}


template<int BLOCK_SZ,int NUM_PER_THREAD>
__global__ void mat_transpose_kernel_v3(const float* idata,float* odata,int M,int N)
{
    const int bx = blockIdx.x,by = blockIdx.y;
    const int tx = threadIdx.x,ty = threadIdx.y;

    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ+1];

    int x = bx * BLOCK_SZ + tx;
    int y = by * BLOCK_SZ + ty;

    constexpr int ROW_STRIDE = BLOCK_SZ / NUM_PER_THREAD;

    if(x<N) {
        #pragma unroll
        for(int y_off=0;y_off<BLOCK_SZ;y_off+=ROW_STRIDE) {
            if(y + y_off < M) {
                sdata[ty+y_off][tx] = idata[(y+y_off)*N+x];
            }
        }
    }
    __syncthreads();

    x = by * BLOCK_SZ + tx;
    y = bx * BLOCK_SZ + ty;
    if(x<M) {
        for(int y_off=0;y_off<BLOCK_SZ;y_off+=ROW_STRIDE) {
            if(y + y_off < N) {
                odata[(y+y_off)*M+x] = sdata[tx][ty+y_off];
            }            
        }
    }
}
void mat_transpose_v3(const float* idata,float* odata,int M,int N) {
    constexpr int BLOCK_SZ = 32;
    constexpr int NUM_PER_THREAD = 4;
    dim3 block(BLOCK_SZ,BLOCK_SZ/NUM_PER_THREAD);
    dim3 grid(Ceil(N,BLOCK_SZ),Ceil(M,BLOCK_SZ));
    mat_transpose_kernel_v3<BLOCK_SZ,NUM_PER_THREAD><<<grid,block>>>(idata,odata,M,N);
}


template <int BLOCK_SZ>
__global__ void mat_transpose_kernel_v3_5(const float* idata, float* odata, int M, int N) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ];

    int x = bx * BLOCK_SZ + tx * 4; // 列维度让每个线程负责连续的多个元素
    int y = by * BLOCK_SZ + ty;

    if (x < N && y < M) {
        FETCH_FLOAT4(sdata[ty][tx * 4]) = FETCH_CFLOAT4(idata[y * N + x]);
    }
    __syncthreads();

    x = by * BLOCK_SZ + tx * 4;
    y = bx * BLOCK_SZ + ty;
    float tmp[4];
    if (x < M && y < N) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            tmp[i] = sdata[tx * 4 + i][ty];
        }
        FETCH_FLOAT4(odata[y * M + x]) = FETCH_FLOAT4(tmp);
    }
}

void mat_transpose_v3_5(const float* idata, float* odata, int M, int N) {
    constexpr int BLOCK_SZ = 32;
    dim3 block(BLOCK_SZ / 4, BLOCK_SZ);
    dim3 grid(Ceil(N, BLOCK_SZ), Ceil(M, BLOCK_SZ));
    mat_transpose_kernel_v3_5<BLOCK_SZ><<<grid, block>>>(idata, odata, M, N);
}


// 减少条件分支
template <int BLOCK_SZ, int NUM_PER_THREAD>
__global__ void mat_transpose_kernel_v4(const float* idata, float* odata, int M, int N) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ+1];

    int x = bx * BLOCK_SZ + tx;
    int y = by * BLOCK_SZ + ty;

    constexpr int ROW_STRIDE = BLOCK_SZ / NUM_PER_THREAD;

    if (x < N) { // 如果该线程块的数据分片并不在矩阵边缘, 那么每次迭代时便无需再进行越界检查
        if (y + BLOCK_SZ <= M) {
            #pragma unroll
            for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
                sdata[ty + y_off][tx] = idata[(y + y_off) * N + x]; 
            }
        } else {
            for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
                if (ty + y_off < M) {
                    sdata[ty + y_off][tx] = idata[(y + y_off) * N + x];
                }
            }
        }

    }
    __syncthreads();

    x = by * BLOCK_SZ + tx;
    y = bx * BLOCK_SZ + ty;
    if (x < M) {
        if (y + BLOCK_SZ <= N) {
            #pragma unroll
            for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
                odata[(y + y_off) * M + x] = sdata[tx][ty + y_off];
            }
        } else {
            for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
                if (y + y_off < N) {
                    odata[(y + y_off) * M + x] = sdata[tx][ty + y_off];
                }
            }
        }
    }
}

void mat_transpose_v4(const float* idata, float* odata, int M, int N) {
    constexpr int BLOCK_SZ = 32;
    constexpr int NUM_PER_THREAD = 4;
    dim3 block(BLOCK_SZ, BLOCK_SZ/NUM_PER_THREAD);
    dim3 grid(Ceil(N, BLOCK_SZ), Ceil(M, BLOCK_SZ));
    mat_transpose_kernel_v4<BLOCK_SZ, NUM_PER_THREAD><<<grid, block>>>(idata, odata, M, N);
}



int main() {
    const int M = 32;  // 行数
    const int N = 32;  // 列数
    const int size = M * N;
    const int bytes = size * sizeof(float);


    float *h_idata = new float[size];
    float *h_odata = new float[size];


    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_idata[i * N + j] = static_cast<float>(i * N + j);
        }
    }


    float *d_idata = nullptr, *d_odata = nullptr;
    cudaMalloc(&d_idata, bytes);
    cudaMalloc(&d_odata, bytes);

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

    mat_transpose_v1(d_idata, d_odata, M, N);

    cudaMemcpy(h_odata, d_odata, bytes, cudaMemcpyDeviceToHost);

    
    std::cout << "Input Matrix:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_idata[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "\nTransposed Matrix:" << std::endl;
    // 注意：转置后矩阵尺寸为 N×M
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << h_odata[i * M + j] << "\t";
        }
        std::cout << std::endl;
    }

    // 释放设备和主机内存
    cudaFree(d_idata);
    cudaFree(d_odata);
    delete[] h_idata;
    delete[] h_odata;

    return 0;
}