#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "cooperative_groups.h"
//#define THREAD_PER_BLOCK 256
// ����warp��shared���ϵ�gpu�������г�Ч��������turing���GPU��nvcc�������Ż��˺ܶ࣬���Ե���Ч��������
// cpu
int filter(int *dst, int *src, int n) {
  int nres = 0;
  for (int i = 0; i < n; i++)
    if (src[i] > 0)
      dst[nres++] = src[i];
  // return the number of elements copied
  return nres;
}

//GPU
// ������Ϊ256000000ʱ��latency=0.257824 ms
// naive kernel
// __global__ void filter_k(int *dst, int *nres, int *src, int n) {
//  int i = threadIdx.x + blockIdx.x * blockDim.x;
//  // �������ݴ���0�ģ�������+1�����Ѹ���д������Դ��Լ�����ֵΪ�����ĵ�ַ
//  if(i < n && src[i] > 0)
//    dst[atomicAdd(nres, 1)] = src[i];
// }

// //������Ϊ256000000ʱ��latency=0.191712 ms
// //block level, use block level atomics based on shared memory
__global__ 
void filter_shared_k(int *dst, int *nres, const int* src, int n) {
  // ����������Ϊshared memory����Ϊ���������̶߳�����ʵ�
  __shared__ int l_n;
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_thread_num = blockDim.x * gridDim.x;

  for (int i = gtid; i < n; i += total_thread_num) {
    // use first thread to zero the counter
    // ��ʼ��ֻ��1���߳�������
    if (threadIdx.x == 0)
      l_n = 0;
    __syncthreads();

    // ÿ��block�ڲ�������0������(l_n)��ÿ������0��thread offset(pos)������1 2 4���̶߳�����0����ô����4���߳���˵����pos = 3
    int d, pos;

    if(i < n && src[i] > 0) {
        //pos: src[thread]>0��thread�ڵ�ǰblock��index
        pos = atomicAdd(&l_n, 1);
    }
    __syncthreads();

    // ÿ��blockѡ��tid=0��Ϊleader
    // leader��ÿ��block�Ĵ���0������l_n�ۼӵ�ȫ�ּ�����(nres),������block�ľֲ���������һ�� reduce�ۺ�
    if(threadIdx.x == 0)
      l_n = atomicAdd(nres, l_n);
    __syncthreads();

    //write & store
    if(i < n && d > 0) {
    // 1. pos: src[thread]>0��thread�ڵ�ǰblock��index
    // 2. l_n: �ڵ�ǰblock��ǰ�漸��block������src>0�ĸ���
    // 3. pos + l_n����ǰthread��ȫ��offset
      pos += l_n; 
      dst[pos] = d;
    }
    __syncthreads();
  }
}

//������Ϊ256000000ʱ��latency=0.219104ms
//warp level, use warp-aggregated atomics
// __device__ int atomicAggInc(int *ctr) {
//   unsigned int active = __activemask();
//   int leader = 0;
//   int change = __popc(active);//warp mask��Ϊ1������
//   int lane_mask_lt;
//   //���
//   asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
//   unsigned int rank = __popc(active & lane_mask_lt); // �ȵ�ǰ�߳�idС��ֵΪ1��mask֮��
//   int warp_res;
//   if(rank == 0)//leader thread of every warp
//     warp_res = atomicAdd(ctr, change);//compute global offset of warp
//   warp_res = __shfl_sync(active, warp_res, leader);//broadcast to every thread
//   return warp_res + rank; // global offset + local offset = final offset����L86��ʾ��src[i]�����յ�����λ��
// }

// __global__ void filter_warp_k(int *dst, int *nres, const int *src, int n) {
//   int i = threadIdx.x + blockIdx.x * blockDim.x;
//   if(i >= n)
//     return;
//   if(src[i] > 0)
//     // ����L70ֻ�Ǽ��㵱ǰthread�������ݵ�ȫ��offset
//     dst[atomicAggInc(nres)] = src[i];
// }

bool CheckResult(int *out, int groudtruth, int n){
    if (*out != groudtruth) {
        return false;
    }
    return true;
}

int main(){
    float milliseconds = 0;
    int N = 2560000;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);

    int *src_h = (int *)malloc(N * sizeof(int));
    int *dst_h = (int *)malloc(N * sizeof(int));
    int *nres_h = (int *)malloc(1 * sizeof(int));
    int *dst, *nres;
    int *src;
    cudaMalloc((void **)&src, N * sizeof(int));
    cudaMalloc((void **)&dst, N * sizeof(int));
    cudaMalloc((void **)&nres, 1 * sizeof(int));

    for(int i = 0; i < N; i++){
        src_h[i] = 1;
    }

    int groudtruth = 0;
    for(int j = 0; j < N; j++){
        if (src_h[j] > 0) {
            groudtruth += 1;
        }
    }


    cudaMemcpy(src, src_h, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    //filter_k<<<Grid,Block>>>(dst,nres,src,N);
    filter_shared_k<<<Grid,Block>>>(dst, nres,  src, N);
    //filter_warp_k<<<Grid, Block>>>(dst, nres, src, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(nres_h, nres, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(nres_h, groudtruth, N);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        printf("%d ",*nres_h);
        printf("\n");
    }
    printf("filter_k latency = %f ms\n", milliseconds);    

    cudaFree(src);
    cudaFree(dst);
    cudaFree(nres);
    free(src_h);
    free(dst_h);
    free(nres_h);
}