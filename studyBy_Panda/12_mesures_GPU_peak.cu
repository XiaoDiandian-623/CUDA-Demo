#include<stdio.h>
#include<cuda.h>
#include<cuda_fp16.h>
#include"cuda_runtime.h"
//4.133518(TFLOPS) 

#define LOOP_TIMES 1000
__global__ void FP32LOPS(int* start,int* stop,float* x,float* y,float* result) 
{
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    float d1 = x[gtid];
    float d2 = y[gtid];
    float res = 0;
    int start_time = 0;
    //PTX÷∏¡Ó
    asm volatile("mov.u32 %0,%%clock;" : "=r"(start_time)::"memory");
    for(int i=0;i<LOOP_TIMES;i++) {
        res = d1*d2 + res;
        res = d1*d2 + res;
        res = d1*d2 + res;
        res = d1*d2 + res;
    }
    asm volatile("bar.sync 0;"); //sync all threads

    int stop_time = 0;
    asm volatile("mov.u32 %0,%%clock;" : "=r"(stop_time)::"memory");

    start[gtid] = start_time;
    stop[gtid] = stop_time;
    result[gtid] = res;
}


int main() 
{
    int N = 1024;
    float* x = (float*)malloc(sizeof(float)*N);
    float* y = (float*)malloc(N*sizeof(float));
    float* d_x;
    float* d_y;
    cudaMalloc((void**)&d_x,N*sizeof(float));
    cudaMalloc((void**)&d_y,N*sizeof(float));

    for(int i=0;i<N;i++) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i);
    }
    cudaMemcpy(d_x,x,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,y,N*sizeof(float),cudaMemcpyHostToDevice);

    float* d_result;
    int* startClock = (int*)malloc(N*sizeof(int));
    int* stopClock = (int*)malloc(N*sizeof(int));
    int* d_startClock;
    int* d_stopClock;
    cudaMalloc((void**)&d_startClock,N*sizeof(int));
    cudaMalloc((void**)&d_stopClock,N*sizeof(int));
    cudaMalloc((void**)&d_result,N*sizeof(float));

    FP32LOPS<<<1,1024>>>(d_startClock,d_stopClock,d_x,d_y,d_result);
    cudaMemcpy(startClock,d_startClock,N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(stopClock,d_stopClock,N*sizeof(int),cudaMemcpyDeviceToHost);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props,0);

    int ThreadsPerSM = props.maxThreadsPerMultiProcessor;
    float FLOPS = (LOOP_TIMES *4*2*1024) / (static_cast<float>(stopClock[0]-startClock[0]));
    printf("GPU max clock rate : %0.2f GHz\n",props.clockRate*1e-6f);
    printf("SM count is %d",props.multiProcessorCount);
    printf("actual NVIDIA T4 GPU peak FLOPS is %f(TFLOPS) \n",FLOPS*props.clockRate*1e-9*props.multiProcessorCount);

    free(x);
    free(y);
    free(startClock);
    free(stopClock);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    cudaFree(d_startClock);
    cudaFree(d_stopClock);

}