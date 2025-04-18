#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdlib>

#define CEIL_DIV(M,N) (((M)+(N)-2) / (N))
#define OFFSET(row,col,ld) ((row)*(ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4>(&pointer)[0])

void sgemm_naive_cpu(float *A,float *B,float *C,int M,int N,int K)
{
    for(int x=0;x<M;x++) {
        for(int y=0;y<N;y++) {
            float sum = 0.0f;
            for(int i=0;i<K;i++) {
                sum += A[x*K+i] * B[i*N+y];
            }
            C[x*N+y] = sum;
        }
    }
}

template<const int BM,
        const int BN,
        const int BK,
        const int TM,
        const int TN>
__global__ void __launch_bounds__((BM*BN)/(TM*TN),1) sgemm_vectorize_kernel(float *A,float *B,float *C,int M,int N,int K)
{
    const uint c_row = blockIdx.y;
    const uint c_col = blockIdx.x;

    const uint block_row_thread = BN / TN;
    const uint block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread;

    const uint thread_col = (threadIdx.x % block_row_thread) * TN;
    const uint thread_row = (threadIdx.x / block_row_thread) * TM;

    // 每行4个字节作为一个内存块，当前线程负责第inner_row_a行的第inner_col_a个内存块的搬运
    const uint inner_row_a = threadIdx.x / (BK / 4);
    const uint inner_col_a = threadIdx.x % (BK / 4) * 4;
    //每行4个字节作为一个内存块 当前线程负责inner_row_b个内存块的搬运
    const uint inner_row_b = threadIdx.x / (BN / 4);
    const uint inner_col_b = threadIdx.x % (BN / 4) * 4;

    //每个线程搬运4个浮点数 完成搬运至smem_a 需要所有线程搬运 ldg_a_num轮
    const int ldg_a_num = BK * BM / thread_num / 4;
    const int ldg_b_num = BK * BN / thread_num / 4;

    //一共BM行 搬运ldg_a_num轮 每轮搬运stride_a行
    const int stride_a = BM / ldg_a_num;
    const int stride_b = BN / ldg_b_num;

    __shared__ float smem_a[BM*BK];
    __shared__ float smem_b[BK*BN];

    //计算当前线程负责计算的矩阵的起始位置
    A += c_row * BM * K;
    B += c_col * BN;
    C += c_row * BM * N + c_col * BN;

    float thread_results[TM * TN] = {0.0};
    float ldg_reg_a[4 * ldg_a_num] = {0.};
    float reg_a[TM] = {0.0};
    float reg_b[TN] = {0.0};

    //outer-most loop over block tiles
    for(uint bk_idx=0;bk_idx<K;bk_idx+=BK) {
        for(int i=0;i<BM;i+=stride_a) {
            int ldg_index = i / stride_a * 4;
            FETCH_FLOAT4(ldg_reg_a[ldg_index]) = FETCH_FLOAT4(A[OFFSET(i+inner_row_a,inner_col_a,K)]);
            // smem_a 转置存，其中 ldg_reg_a 做中间缓存，目的是读取时可以按FLOAT4读取
            smem_a[OFFSET(inner_col_a, i + inner_row_a, BM)] = ldg_reg_a[ldg_index];
            smem_a[OFFSET(inner_col_a + 1, i + inner_row_a, BM)] = ldg_reg_a[ldg_index + 1];
            smem_a[OFFSET(inner_col_a + 2, i + inner_row_a, BM)] = ldg_reg_a[ldg_index + 2];
            smem_a[OFFSET(inner_col_a + 3, i + inner_row_a, BM)] = ldg_reg_a[ldg_index + 3];
        }

        for(int i=0;i<BK;i+=stride_b) {
            FETCH_FLOAT4(smem_b[OFFSET(inner_row_b+i,inner_col_b,BN)]) = 
                FETCH_FLOAT4(B[OFFSET(inner_row_b+i,inner_col_b,N)]);
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        // calculate per-thread results
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
        {
            for (int m = 0; m < TM; m += 4)
            {
                FETCH_FLOAT4(reg_a[m]) = FETCH_FLOAT4(smem_a[OFFSET(dot_idx, thread_row + m, BM)]);
            }
            for (int n = 0; n < TN; n += 4)
            {
                FETCH_FLOAT4(reg_b[n]) = FETCH_FLOAT4(smem_b[OFFSET(dot_idx, thread_col + n, BN)]);
            }

            for (int m = 0; m < TM; m++)
            {
                for (int n = 0; n < TN; n++)
                {
                    thread_results[m * TN + n] += reg_a[m] * reg_b[n];
                }
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; m++)
        {
          for (int n = 0; n < TN; n += 4)
            {
               FETCH_FLOAT4(C[OFFSET(thread_row + m, thread_col + n, N)]) = FETCH_FLOAT4(thread_results[m * TN + n]);
            }
        }
}

void run_sgemm_vectorize(float *A,float *B,float *C,int m,int n,int k)
{
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    const uint BM = 64;
    const uint BN = 64;

    dim3 gridDim(CEIL_DIV(n,BN),CEIL_DIV(m,BM));
    dim3 blockDim((BM*BN) / (TM*TN));
    sgemm_vectorize_kernel<BM,BN,BK,TM,TN>
        <<<gridDim,blockDim>>> (A,B,C,m,n,k);
}
