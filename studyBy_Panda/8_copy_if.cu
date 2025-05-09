#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "cooperative_groups.h"
//#define THREAD_PER_BLOCK 256
// 这种warp和shared在老的gpu上面会很有成效，但是在turing后的GPU，nvcc编译器优化了很多，所以导致效果不明显
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
// 数据量为256000000时，latency=0.257824 ms
// naive kernel
// __global__ void filter_k(int *dst, int *nres, int *src, int n) {
//  int i = threadIdx.x + blockIdx.x * blockDim.x;
//  // 输入数据大于0的，计数器+1，并把该数写到输出显存以计数器值为索引的地址
//  if(i < n && src[i] > 0)
//    dst[atomicAdd(nres, 1)] = src[i];
// }

// //数据量为256000000时，latency=0.191712 ms
// //block level, use block level atomics based on shared memory
__global__ 
void filter_shared_k(int *dst, int *nres, const int* src, int n) {
  // 计数器声明为shared memory，因为这是所有线程都会访问的
  __shared__ int l_n;
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_thread_num = blockDim.x * gridDim.x;

  for (int i = gtid; i < n; i += total_thread_num) {
    // use first thread to zero the counter
    // 初始化只需1个线程来操作
    if (threadIdx.x == 0)
      l_n = 0;
    __syncthreads();

    // 每个block内部，大于0的数量(l_n)和每个大于0的thread offset(pos)，比如1 2 4号线程都大于0，那么对于4号线程来说它的pos = 3
    int d, pos;

    if(i < n && src[i] > 0) {
        //pos: src[thread]>0的thread在当前block的index
        pos = atomicAdd(&l_n, 1);
    }
    __syncthreads();

    // 每个block选出tid=0作为leader
    // leader把每个block的大于0的数量l_n累加到全局计数器(nres),即所有block的局部计数器做一个 reduce聚合
    if(threadIdx.x == 0)
      l_n = atomicAdd(nres, l_n);
    __syncthreads();

    //write & store
    if(i < n && d > 0) {
    // 1. pos: src[thread]>0的thread在当前block的index
    // 2. l_n: 在当前block的前面几个block的所有src>0的个数
    // 3. pos + l_n：当前thread的全局offset
      pos += l_n; 
      dst[pos] = d;
    }
    __syncthreads();
  }
}

//数据量为256000000时，latency=0.219104ms
//warp level, use warp-aggregated atomics
// __device__ int atomicAggInc(int *ctr) {
//   unsigned int active = __activemask();
//   int leader = 0;
//   int change = __popc(active);//warp mask中为1的数量
//   int lane_mask_lt;
//   //汇编
//   asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
//   unsigned int rank = __popc(active & lane_mask_lt); // 比当前线程id小且值为1的mask之和
//   int warp_res;
//   if(rank == 0)//leader thread of every warp
//     warp_res = atomicAdd(ctr, change);//compute global offset of warp
//   warp_res = __shfl_sync(active, warp_res, leader);//broadcast to every thread
//   return warp_res + rank; // global offset + local offset = final offset，即L86表示的src[i]的最终的索引位置
// }

// __global__ void filter_warp_k(int *dst, int *nres, const int *src, int n) {
//   int i = threadIdx.x + blockIdx.x * blockDim.x;
//   if(i >= n)
//     return;
//   if(src[i] > 0)
//     // 以上L70只是计算当前thread负责数据的全局offset
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