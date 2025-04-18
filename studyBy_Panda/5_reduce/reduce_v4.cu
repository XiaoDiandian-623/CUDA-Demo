#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

//v4: ���һ��warp���ò���__syncthreads

__device__ void WarpSharedMemReduce(volatile float* smem, int tid){
    // CUDA����֤���е�shared memory������������д����֮ǰ��ɣ���˴��ھ�����ϵ�����ܵ��½������
    // ����smem[tid] += smem[tid + 16] => smem[3] += smem[16], smem[16] += smem[32]
    // ��ʱL9��smem[16]�Ķ���д����˭��ǰ˭�ں����ǲ�ȷ���ģ�������Volta�ܹ����������м�Ĵ���(L11)���syncwarp��֤��д����
    float x = smem[tid];
    if (blockDim.x >= 64) {
      x += smem[tid + 32]; __syncwarp();
      smem[tid] = x; __syncwarp();
    }
    x += smem[tid + 16]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 8]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 4]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 2]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 1]; __syncwarp();
    smem[tid] = x; __syncwarp();
}

template<int blockSize>
__global__ void reduce_v4(float *d_in,float *d_out){
    __shared__ float smem[blockSize];

    int tid = threadIdx.x;
    // ��ָ��ǰ�߳�������block��Χ�ڵ�ȫ��id, *2����ǰblockҪ����2*blocksize������
    // ep. blocksize = 2, blockIdx.x = 1, when tid = 0, gtid = 4, gtid + blockSize = 6; when tid = 1, gtid = 5, gtid + blockSize = 7
    // ep. blocksize = 2, blockIdx.x = 0, when tid = 0, gtid = 0, gtid + blockSize = 2; when tid = 1, gtid = 1, gtid + blockSize = 3
    // so, we can understand L38, one thread handle data located in tid and tid + blockSize 
    int i = blockIdx.x * (blockSize * 2) + threadIdx.x;

    smem[tid] = d_in[i] + d_in[i + blockSize];
    __syncthreads();

    // ����v3�Ľ��������һ��warp�������reduce���������һ��sync threads
    // ��ʱһ��block��d_in������ݵ�reduce sum���������idΪ0���߳�����
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }


    if (tid < 32) { //չ������warp
        WarpSharedMemReduce(smem, tid);
    }
    
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}

bool CheckResult(float *out, float groudtruth, int n){
    float res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
    }

    if (res != groudtruth) {
        return false;
    }
    return true;
}

int main(){
    float milliseconds = 0;

    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    double msecPerMatrixMul[2] = {0,0};
    double gigaFlops[2] = {0,0};
    double flopPerMatrixMul = 2.0 * N;
    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    float *out = (float*)malloc((GridSize) * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(float));

    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
    }

    float groudtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize / 2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v4<blockSize / 2><<<Grid,Block>>>(d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d \n", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < GridSize;i++){
            printf("resPerBlock : %lf ",out[i]);
        }
        printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    msecPerMatrixMul[0] = milliseconds;
    gigaFlops[0] = (flopPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf("Gemm performance = %.2f GFlop/s,Time= %.3f,Size = %.0f Ops,\n",gigaFlops[0],msecPerMatrixMul[0],flopPerMatrixMul);
    printf("reduce_v4 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
