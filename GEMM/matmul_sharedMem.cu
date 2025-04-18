#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void sgemm_naive_cpu(float *A, float *B, float *C, int M, int N, int K)
{
    for (int x = 0; x < M; x++)
    {
        for (int y = 0; y < N; y++)
        {
            float sum = 0.0f;
            for (int i = 0; i < K; i++)
            {
                sum += A[x * K + i] * B[i * N + y];
            }
            C[x * N + y] = sum;
        }
    }
}

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    const uint c_row = blockIdx.x;
    const uint c_col = blockIdx.y;

    __shared__ float A_shared[BLOCKSIZE * BLOCKSIZE];
    __shared__ float B_shared[BLOCKSIZE * BLOCKSIZE];

    // 每个线程在矩阵块的位置
    const uint thread_row = threadIdx.x / BLOCKSIZE;
    const uint thread_col = threadIdx.x % BLOCKSIZE;

    // advance pointers to the starting positions
    A += c_row * BLOCKSIZE * K;
    B += c_col * BLOCKSIZE;
    C += c_row * BLOCKSIZE * N + c_col * BLOCKSIZE;

    float tmp = 0.0f;
    for (int i = 0; i < K; i += BLOCKSIZE)
    {
        // global Mem -> shared memory
        A_shared[thread_row * BLOCKSIZE + thread_col] = A[thread_row * K + thread_col];
        B_shared[thread_row * BLOCKSIZE + thread_col] = B[thread_row * N + thread_col];
        __syncthreads();

        // 一个块内的部分和
        for (int j = 0; j < BLOCKSIZE; j++)
        {
            tmp += A_shared[thread_row * BLOCKSIZE + j] * B_shared[j * BLOCKSIZE + thread_col];
        }
        __syncthreads();

        // 推进到下一个块
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
    }

    C[thread_row * N + thread_col] = tmp;
}

void run_sgemm_shared_memory(float *A, float *B, float *C, int m, int n, int k)
{
    const int BLOCKSIZE = 32;
    dim3 block_size(BLOCKSIZE * BLOCKSIZE);
    dim3 grid_size(CEIL_DIV(m, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));
    sgemm_shared_mem_kernel<BLOCKSIZE><<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void randomize_matrix(float *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = rand() % 100;
    }
}

int main()
{
    int m = 256;
    int n = 256;
    int k = 256;

    float *A, *B, *C, *C_ref;
    float *d_A, *d_B, *d_C, *d_C_ref;

    A = new float[m * k];
    B = new float[k * n];
    C = new float[m * n];
    // save reference result
    C_ref = new float[m * n];

    randomize_matrix(A, m * k);
    randomize_matrix(B, k * n);

    cudaEvent_t start,stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void **)&d_A, m * k * sizeof(float));
    cudaMalloc((void **)&d_B, k * n * sizeof(float));
    cudaMalloc((void **)&d_C, m * n * sizeof(float));
    cudaMalloc((void **)&d_C_ref, m * n * sizeof(float));
    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ref, C_ref, m * n * sizeof(float), cudaMemcpyHostToDevice);

    run_sgemm_shared_memory(d_A, d_B, d_C, m, n, k);
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << elapsedTime << " ms" << std::endl;

    sgemm_naive_cpu(A, B, C_ref, m, n, k);


    for (int i = 0; i < m * n; i++)
    {
        if (C[i] != C_ref[i])
        {
            printf("Error: mismatch at index %d, expected %f, got %f\n", i, C_ref[i], C[i]);
            return 1;
        }
    }

    printf("Success!\n");
    return 0;
}