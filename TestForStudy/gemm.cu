#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define OFFSET(row, col, stride) ((row) * (stride) + (col))

// CUDA kernel：计算矩阵乘法 C = A * B，其中 A 的维度为 M×K，B 为 K×N
__global__ void basic_gemm(
    float *A, float *B, float *C,
    const int M, const int N, const int K) {

    int _x = blockIdx.x * blockDim.x + threadIdx.x;
    int _y = blockIdx.y * blockDim.y + threadIdx.y;
    if (_x < M && _y < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[OFFSET(_x, k, K)] * B[OFFSET(k, _y, N)];
        }
        C[OFFSET(_x, _y, N)] = sum;
    }
}

// Host端初始化矩阵数据
void initialize_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    // 设置随机数种子
    srand(time(NULL));

    // 矩阵大小
    const int M = 4096;
    const int K = 1024;
    const int N = 4096;
    const int ITER = 100;

    // 在主机端分配内存
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // 初始化矩阵 A 与 B
    initialize_matrix(h_A, M, K);
    initialize_matrix(h_B, K, N);

    // 在 GPU 上分配内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // 将主机数据拷贝到设备内存
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 定义线程块大小（32x32）
    dim3 blockDim(32, 32, 1);
    // 计算网格大小，确保覆盖所有元素
    int grid_x = (M + blockDim.x - 1) / blockDim.x;
    int grid_y = (N + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(grid_x, grid_y, 1);

    // 创建 CUDA 事件用于计时
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int i = 0; i < ITER; i++) {
        basic_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec = 0.0f;
    cudaEventElapsedTime(&msec, start, end);

    // 计算工作量：每次迭代进行 M*N*K 次乘加操作，每次乘加 2 次 FLOP
    long workload = (long)M * N * K * 2 * ITER;
    double avg_Tflops = ((double)workload / 1e12) / ((double)msec / 1e3);
    printf("Average Performance: %6.4lf Tflops\n", avg_Tflops);

    // 可选：将计算结果从设备内存拷贝回主机进行验证
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 清理资源
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}
