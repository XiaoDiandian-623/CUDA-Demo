// sgemm naive, sgemm + block-tile + k-tile + vec4
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include <iostream>
#include <cmath>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])



// 验证CUDA函数调用是否成功
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

// CPU版本矩阵乘法 (用于验证结果)
void sgemm_cpu(float *a, float *b, float *c, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

// 生成随机浮点数矩阵
void init_matrix(float *mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX; // [0, 1)
    }
}

// 比较两个矩阵的结果是否一致
bool verify_result(float *c_cpu, float *c_gpu, int M, int N) {
    const float eps = 1e-3;
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(c_cpu[i] - c_gpu[i]) > eps) {
            printf("Mismatch at index %d: CPU=%.5f, GPU=%.5f\n", 
                   i, c_cpu[i], c_gpu[i]);
            return false;
        }
    }
    return true;
}




// SGEMM: Block Tile + K Tile, with smem
// Block Tile (BM, BN) + K Tile (BK=32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
// a: MxK, b: KxN, c: MxN, compute: c = a * b, all row major  
__global__ void sgemm(float* a, float* b, float* c, int M, int N, int K) {
  // [1] Block Tile: 32x32的block处理c上一块32x32的元素计算
  // [2]     K Tile: 使用共享内存，并将K分块为BK大小的块
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;
  __shared__ float s_a[BM][BK], s_b[BK][BN]; 

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx; // tid within the block
  // load values to shared memory, 32x32 threads working together 
  // to fetch data along the row direction of a and b both for s_a 
  // and s_b 32x32x4x2=8KB, we use 32x32 threads within block to 
  // load 32x32 elements from global memory to shared memory, namely, 
  // each thread will load 1 element.
  int load_smem_a_m = tid / 32; // 0~31, tid / 32, tid / BM, threadIdx.y
  int load_smem_a_k = tid % 32; // 0~31, tid % 32, tid % BK, threadIdx.x
  int load_smem_b_k = tid / 32; // 0~31, tid / 32, tid / BK, threadIdx.y
  int load_smem_b_n = tid % 32; // 0~31, tid % 32, tid % BN, threadIdx.x
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  // if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;
  
  float sum = 0.f;
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      int comp_smem_a_m = load_smem_a_m;
      int comp_smem_b_n = load_smem_b_n;
      sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
    }
    __syncthreads();
  }
  int store_gmem_c_m = load_gmem_a_m;
  int store_gmem_c_n = load_gmem_b_n;
  int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
  c[store_gmem_c_addr] = sum;
}

// SGEMM: Block Tile + Thread Tile + K Tile + Vec4, with smem
// BK:TILE_K=8 BM=BN=128
// TM=TN=8 增加计算密度 BM/TM=16 BN/TN=16
// dim3 blockDim(BN/TN, BM/TM);
// dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)
__global__ void sgemm_thread_tile_vec4(
  float* a, float* b, float* c, int M, int N, int K) {
  // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
  // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
  // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次，
  //                  每次计算TM*TN个元素各自的部分乘累加
  // [4]   Vectorize: 减少load和store指令，使用float4
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8; 
  constexpr int TM = 8;
  constexpr int TN = 8;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx; // tid within the block
  __shared__ float s_a[BM][BK], s_b[BK][BN]; // 2*128*8*4=8KB
  
  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行8个数据，每个线程读取4个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // tid/2 (128/8)*(128/8)=256 threads per block, tid/2->[0,128), BM=128 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4;  // (tid%2 == 0) ? 0 : 4, col of s_a 0,4
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=8 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读4个数据，需要32个线程；总共8行，需要32x8=256个线程
  int load_smem_b_k = tid / 32; // tid/32, row of s_b 256/32=8 行 0~7
  int load_smem_b_n = (tid % 32) * 4;  // (tid % 32) * 4, col of s_b 0,4,...,124
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  
  float r_c[TM][TN] = {0.0}; // 8x8
  // 2. 先对K进行分块，每块BK大小
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    // 加载数据到共享内存smem s_a BM*BK 128*8 vectorize float4
    int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]);
    // 加载数据到共享内存smem s_b BK*BN 8*128 vectorize float4
    int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 
    FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]); 
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; k++) {
      // 3. 每个线程负责计算BM*BN(12x128)中的TM*TN(8x8)个元素
      #pragma unroll
      for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
          // k from 0~7，0 ~ BK, ty and tx range from 0 to 15, 16x8=128
          int comp_smem_a_m = ty * TM + m;  // 128*8 128/TM(8)=16 M方向 16线程
          int comp_smem_b_n = tx * TN + n;  // 8*128 128/TN(8)=16 N方向 16线程
          r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    int store_gmem_c_m = by * BM + ty * TM + m;
    #pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int store_gmem_c_n = bx * BN + tx * TN + n;
      int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
      FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
    }
  }
}



int main() {
    // 设置矩阵维度 (测试时可调整)
    const int M = 256;
    const int N = 256;
    const int K = 256;

    // 分配主机内存
    float *a_cpu = new float[M * K];
    float *b_cpu = new float[K * N];
    float *c_cpu = new float[M * N];
    float *c_gpu_naive = new float[M * N];
    float *c_gpu_opt = new float[M * N];

    // 初始化输入矩阵
    init_matrix(a_cpu, M * K);
    init_matrix(b_cpu, K * N);
    memset(c_cpu, 0, M * N * sizeof(float));
    memset(c_gpu_naive, 0, M * N * sizeof(float));
    memset(c_gpu_opt, 0, M * N * sizeof(float));

    // CPU计算参考结果
    sgemm_cpu(a_cpu, b_cpu, c_cpu, M, N, K);

    // 分配设备内存
    float *a_gpu, *b_gpu, *c_gpu;
    CHECK_CUDA(cudaMalloc(&a_gpu, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b_gpu, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&c_gpu, M * N * sizeof(float)));

    // 拷贝数据到设备
    CHECK_CUDA(cudaMemcpy(a_gpu, a_cpu, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(b_gpu, b_cpu, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // 测试第一个内核：sgemm（Naive版本）
    {
        CHECK_CUDA(cudaMemset(c_gpu, 0, M * N * sizeof(float)));
        dim3 block(32, 32);  // BM=32, BN=32
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        sgemm<<<grid, block>>>(a_gpu, b_gpu, c_gpu, M, N, K);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpy(c_gpu_naive, c_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaDeviceSynchronize());
        printf("Naive kernel verification: %s\n", 
               verify_result(c_cpu, c_gpu_naive, M, N) ? "PASS" : "FAIL");
    }

    // 测试第二个内核：sgemm_thread_tile_vec4（优化版本）
    {
        CHECK_CUDA(cudaMemset(c_gpu, 0, M * N * sizeof(float)));
        dim3 block_threads(16, 16); // BN=128/TN=8 => 16, BM=128/TM=8 => 16
        dim3 grid_threads((N + 127) / 128, (M + 127) / 128);
        sgemm_thread_tile_vec4<<<grid_threads, block_threads>>>(a_gpu, b_gpu, c_gpu, M, N, K);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpy(c_gpu_opt, c_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaDeviceSynchronize());
        printf("Optimized kernel verification: %s\n", 
               verify_result(c_cpu, c_gpu_opt, M, N) ? "PASS" : "FAIL");
    }

    // 释放内存
    delete[] a_cpu;
    delete[] b_cpu;
    delete[] c_cpu;
    delete[] c_gpu_naive;
    delete[] c_gpu_opt;
    CHECK_CUDA(cudaFree(a_gpu));
    CHECK_CUDA(cudaFree(b_gpu));
    CHECK_CUDA(cudaFree(c_gpu));

    return 0;
}