#include<bits/stdc++.h>
#include<cuda.h>
#include"cuda_runtime.h"

#define WarpSize 32
//latency = 1.4252416ms

//warp层面的加
template<int blockSize>
__device__ float WarpShuffle(float sum) 
{
    if(blockSize >= 32) sum += __shfl_down_sync(0xffffffff,sum,16);
    if(blockSize >= 16) sum += __shfl_down_sync(0xffffffff,sum,8);
    if(blockSize >= 8) sum += __shfl_down_sync(0xffffffff,sum,4);
    if(blockSize >= 4) sum += __shfl_down_sync(0xffffffff,sum,2);
    if(blockSize >= 2) sum += __shfl_down_sync(0xffffffff,sum,1); 
    return sum;
}

template<int blockSize>
__global__ void reduce_warp_level(float* d_in,float* d_out,unsigned int n)
{
    float sum = 0;

    unsigned tid = threadIdx.x;
    unsigned gtid = blockIdx.x * blockSize + threadIdx.x;
    unsigned total_thread_num = blockSize * gridDim.x;

    for(int i=gtid;i<n;i+=total_thread_num) {
       sum += d_in[i];
    }

    //partial sum for each warp
    __shared__ float WarpSums[blockSize/WarpSize];
    const int laneId = tid % WarpSize;//当前id在某一个warp内的序号
    const int warpId = tid / WarpSize;//当前的warpid在所有warp内的序号
    sum = WarpShuffle<blockSize>(sum);
    if(laneId == 0) {
        WarpSums[warpId] = sum;
    }
    __syncthreads();

    sum = (tid < blockSize/WarpSize) ? WarpSums[laneId] : 0;

    if(warpId == 0) {
        sum = WarpShuffle<blockSize>(sum);
    }
    if(tid == 0) {
        d_out[blockIdx.x] = sum;
    }
}

bool CheckResult(float* out,float groudtruth,int n)
{
    float res = 0;
    for(int i=0;i<n;i++) {
        res += out[i];
    }
    if(res != groudtruth) {
        return false;
    }
    return true;

}

int main()
{
    float milliseconds = 0;

    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,0);

    const int blockSize = 256;
    int GridSize = std::min((N + blockSize - 1) / blockSize, deviceProp.maxGridSize[0]);
    //int GridSize = 100000;
    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    float *out = (float*)malloc((GridSize) * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(float));
    double msecPerMatrixMul[2] = {0,0};
    double gigaFlops[2] = {0,0};
    double flopPerMatrixMul = 2.0 * N;
    
    for(int i = 0; i < N; i++){
        a[i] = 2.0f;
    }

    float groudtruth = N * 2.0f*2.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_warp_level<blockSize><<<Grid,Block>>>(d_a, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
   // printf("allcated %d blocks, data counts are %d \n", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    // if(is_right) {
    //     printf("the ans is right\n");
    // } else {
    //     printf("the ans is wrong\n");
    //     for(int i = 0; i < GridSize;i++){
    //         printf("resPerBlock : %lf ",out[i]);
    //     }
    //     printf("\n");
    //     printf("groudtruth is: %f \n", groudtruth);
    // }
    msecPerMatrixMul[0] = milliseconds;
    gigaFlops[0] = (flopPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf("Gemm performance = %.2f GFlop/s,Time= %.3f,Size = %.0f Ops,\n",gigaFlops[0],msecPerMatrixMul[0],flopPerMatrixMul);
    printf("reduce_warp_level latency = %f ms\n", milliseconds);
    
    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
