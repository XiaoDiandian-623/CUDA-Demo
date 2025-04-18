#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cfloat>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
//-----------------------------------------------------
// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

//-----------------------------------------------------
// SGEMV: Warp SGEMV K32
// 假设K为32的倍数，每个warp负责一行
// grid(M/4), block(32,4)  blockDim.x=32, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = A*x
__global__ void sgemv_k32(float* a, float* x, float* y, int M, int K) {
  int tx = threadIdx.x;         // 0 ~ 31
  int ty = threadIdx.y;         // 0 ~ 3
  int bx = blockIdx.x;          // 0 ~ M/4 - 1
  int lane = tx % WARP_SIZE;    // 0 ~ 31
  int m = bx * blockDim.y + ty; // 行号
  if (m < M) {
    float sum = 0.0f;
    int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      int k = w * WARP_SIZE + lane;
      if (k < K)
        sum += a[m * K + k] * x[k];
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    if (lane == 0) y[m] = sum;
  }
}

//-----------------------------------------------------
// SGEMV: Warp SGEMV K128 + Vec4
// 假设K为128的倍数，利用float4一次加载4个float
// grid(M/4), block(32,4)  blockDim.x=32, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = A*x
__global__ void sgemv_k128(float* a, float* x, float* y, int M, int K) {
  int tx = threadIdx.x;         // 0 ~ 31
  int ty = threadIdx.y;         // 0 ~ 3
  int bx = blockIdx.x;          // 0 ~ M/4 - 1
  int lane = tx % WARP_SIZE;    // 0 ~ 31
  int m = bx * blockDim.y + ty; // 行号
  if (m < M) {
    float sum = 0.0f;
    // 每个线程每次处理4个元素，整个warp处理4*WARP_SIZE个元素
    int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      int k = (w * WARP_SIZE + lane) * 4;
      if (k < K) { // 注意：由于K是128的倍数，通常不会越界
        float4 reg_x = FLOAT4(x[k]);
        float4 reg_a = FLOAT4(a[m * K + k]);
        sum += (reg_a.x * reg_x.x + reg_a.y * reg_x.y +
                reg_a.z * reg_x.z + reg_a.w * reg_x.w);
      }
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    if(lane == 0) y[m] = sum;
  }
}

//-----------------------------------------------------
// SGEMV: Warp SGEMV K16
// 假设K为16 (< 32), 每个warp负责2行，每行16个元素
// NUM_THREADS = 128, NUM_WARPS = NUM_THREADS / WARP_SIZE, ROW_PER_WARP = 2
// grid(M/NUM_ROWS), block(32,NUM_WARPS)  NUM_ROWS = NUM_WARPS * ROW_PER_WARP
// a: MxK, x: Kx1, y: Mx1, compute: y = A*x
template<const int ROW_PER_WARP = 2> 
__global__ void sgemv_k16(float* A, float* x, float* y, int M, int K) {
  // K_WARP_SIZE: 每个warp中每行处理元素数
  constexpr int K_WARP_SIZE = (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP; // 例如: (32+2-1)/2 = 16
  int tx = threadIdx.x;       // 0 ~ 31
  int ty = threadIdx.y;       // 0 ~ NUM_WARPS - 1
  int bx = blockIdx.x;        // 0 ~ M/NUM_ROWS - 1, NUM_ROWS = NUM_WARPS*ROW_PER_WARP
  int lane = tx;              // 0 ~ 31
  int k = lane % K_WARP_SIZE; // 0 ~ 15
  // 计算全局行号：每个线程组处理 ROW_PER_WARP 行，每个warp对应 ROW_PER_WARP 行
  int m = (blockDim.y * bx + ty) * ROW_PER_WARP + lane / K_WARP_SIZE;
  if (m < M && k < K) {
    float sum = A[m * K + k] * x[k];
    sum = warp_reduce_sum<K_WARP_SIZE>(sum);
    // 注意这里使用 k==0 作为归约完成的标志
    if(k == 0) y[m] = sum;
  }
}

//-----------------------------------------------------
// CPU版SGEMV (简单实现)
void sgemv_cpu(const float* A, const float* x, float* y, int M, int K) {
  for (int m = 0; m < M; m++) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      sum += A[m*K + k] * x[k];
    }
    y[m] = sum;
  }
}

//-----------------------------------------------------
// 计算误差：返回最大绝对误差
float compute_max_error(const float* ref, const float* res, int M) {
  float max_err = 0.0f;
  for (int i = 0; i < M; i++) {
    float err = fabs(ref[i] - res[i]);
    if(err > max_err) max_err = err;
  }
  return max_err;
}

//-----------------------------------------------------
// 辅助函数：检查CUDA错误
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << "@" << __LINE__      \
                      << ": " << cudaGetErrorString(err) << std::endl;        \
            exit(1);                                                          \
        }                                                                     \
    } while (0)


//-----------------------------------------------------
// 主函数：分别测试三种内核，并输出GPU执行时间、带宽等信息
int main() {
  //--------------- SGEMV K32 测试 ---------------
  {
    std::cout << "===== SGEMV K32 Test =====" << std::endl;
    const int M = 1024;
    const int K = 128;  // K为32的倍数
    size_t sizeA = M * K * sizeof(float);
    size_t sizeX = K * sizeof(float);
    size_t sizeY = M * sizeof(float);

    // 分配主机内存
    float* h_A = (float*)malloc(sizeA);
    float* h_x = (float*)malloc(sizeX);
    float* h_y_cpu = (float*)malloc(sizeY);
    float* h_y_gpu = (float*)malloc(sizeY);

    // 初始化矩阵和向量
    for (int i = 0; i < M*K; i++) {
      h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K; i++) {
      h_x[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // CPU计算
    auto cpu_start = std::chrono::high_resolution_clock::now();
    sgemv_cpu(h_A, h_x, h_y_cpu, M, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // 分配设备内存
    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_x, sizeX));
    CUDA_CHECK(cudaMalloc((void**)&d_y, sizeY));

    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeX, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, sizeY));

    // 定义block和grid：grid(M/4), block(32,4)
    dim3 blockDim(32, 4);
    dim3 gridDim((M + blockDim.y - 1) / blockDim.y);

    // 使用CUDA Event计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    sgemv_k32<<<gridDim, blockDim>>>(d_A, d_x, d_y, M, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpuTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));  // 毫秒

    // 将结果拷贝回主机
    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, sizeY, cudaMemcpyDeviceToHost));

    // 计算最大误差
    float max_err = compute_max_error(h_y_cpu, h_y_gpu, M);
    // 计算数据传输量：读 A, 读 x, 写 y
    size_t bytes = sizeA + sizeX + sizeY;
    double bandwidth = bytes / (gpuTime / 1000.0) / 1e9; // GB/s

    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU time: " << gpuTime << " ms" << std::endl;
    std::cout << "Max error: " << max_err << std::endl;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // 清理
    free(h_A); free(h_x); free(h_y_cpu); free(h_y_gpu);
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }

  //--------------- SGEMV K128 测试 ---------------
  {
    std::cout << "\n===== SGEMV K128 Test =====" << std::endl;
    const int M = 1024;
    const int K = 128;  // K为128的倍数
    size_t sizeA = M * K * sizeof(float);
    size_t sizeX = K * sizeof(float);
    size_t sizeY = M * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_x = (float*)malloc(sizeX);
    float* h_y_cpu = (float*)malloc(sizeY);
    float* h_y_gpu = (float*)malloc(sizeY);

    for (int i = 0; i < M*K; i++) {
      h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K; i++) {
      h_x[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    sgemv_cpu(h_A, h_x, h_y_cpu, M, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_x, sizeX));
    CUDA_CHECK(cudaMalloc((void**)&d_y, sizeY));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeX, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, sizeY));

    // grid(M/4), block(32,4)
    dim3 blockDim(32, 4);
    dim3 gridDim((M + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    sgemv_k128<<<gridDim, blockDim>>>(d_A, d_x, d_y, M, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpuTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, sizeY, cudaMemcpyDeviceToHost));

    float max_err = compute_max_error(h_y_cpu, h_y_gpu, M);
    size_t bytes = sizeA + sizeX + sizeY;
    double bandwidth = bytes / (gpuTime / 1000.0) / 1e9;

    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU time: " << gpuTime << " ms" << std::endl;
    std::cout << "Max error: " << max_err << std::endl;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    free(h_A); free(h_x); free(h_y_cpu); free(h_y_gpu);
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }

  //--------------- SGEMV K16 测试 ---------------
  {
    std::cout << "\n===== SGEMV K16 Test =====" << std::endl;
    const int M = 1024;
    const int K = 16;  // 固定K=16
    size_t sizeA = M * K * sizeof(float);
    size_t sizeX = K * sizeof(float);
    size_t sizeY = M * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_x = (float*)malloc(sizeX);
    float* h_y_cpu = (float*)malloc(sizeY);
    float* h_y_gpu = (float*)malloc(sizeY);

    for (int i = 0; i < M*K; i++) {
      h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K; i++) {
      h_x[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    sgemv_cpu(h_A, h_x, h_y_cpu, M, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_x, sizeX));
    CUDA_CHECK(cudaMalloc((void**)&d_y, sizeY));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeX, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, sizeY));

    // 对于 sgemv_k16: block(32, NUM_WARPS) 其中NUM_WARPS = blockDim.y, 每个warp负责2行，因此每个block处理 NUM_WARPS*2 行
    // 这里取 blockDim.y = 4 => 每个block处理 8 行, grid = M/8
    dim3 blockDim(32, 4);
    dim3 gridDim((M + (blockDim.y * 2) - 1) / (blockDim.y * 2));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    sgemv_k16<2><<<gridDim, blockDim>>>(d_A, d_x, d_y, M, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpuTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, sizeY, cudaMemcpyDeviceToHost));

    float max_err = compute_max_error(h_y_cpu, h_y_gpu, M);
    size_t bytes = sizeA + sizeX + sizeY;
    double bandwidth = bytes / (gpuTime / 1000.0) / 1e9;

    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU time: " << gpuTime << " ms" << std::endl;
    std::cout << "Max error: " << max_err << std::endl;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    free(h_A); free(h_x); free(h_y_cpu); free(h_y_gpu);
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }

  return 0;
}
