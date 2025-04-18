#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//CUDA RunTime API
#include <cuda_runtime.h>

#define THREAD_NUM 256

#define MATRIX_SIZE 1000

const int blocks_num = (MATRIX_SIZE*MATRIX_SIZE + THREAD_NUM - 1) / THREAD_NUM;

//打印设备信息
void printDeviceProp(const cudaDeviceProp &prop)
{
printf("Device Name : %s.\n", prop.name);
printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
printf("regsPerBlock : %d.\n", prop.regsPerBlock);
printf("warpSize : %d.\n", prop.warpSize);
printf("memPitch : %d.\n", prop.memPitch);
printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
printf("totalConstMem : %d.\n", prop.totalConstMem);
printf("major.minor : %d.%d.\n", prop.major, prop.minor);
printf("clockRate : %d.\n", prop.clockRate);
printf("textureAlignment : %d.\n", prop.textureAlignment);
printf("deviceOverlap : %d.\n", prop.deviceOverlap);
printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

//CUDA 初始化
bool InitCUDA()
{
    int count;

    //取得支持Cuda的装置的数目
    cudaGetDeviceCount(&count);

    if (count == 0) 
    {
        fprintf(stderr, "There is no device.\n");

        return false;
    }

    int i;

    for (i = 0; i < count; i++) 
    {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    //打印设备信息
    printDeviceProp(prop);

        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) 
        {
            if (prop.major >= 1) 
            {
            break;
            }
        }
    }

    if (i == count) 
    {
    fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
    return false;
    }

    cudaSetDevice(i);

    return true;

}

//生成随机矩阵
void matgen(float* a, int n)
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {

            a[i * n + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);

        }
    }
}

// __global__ 函数 并行计算矩阵乘法
__global__ static void matMultCUDA(const float* a, const float* b, float* c, int n, clock_t* time)
{

    //表示目前的 thread 是第几个 thread（由 0 开始计算）
    const int tid = threadIdx.x;

    //表示目前的 thread 属于第几个 block（由 0 开始计算）
    const int bid = blockIdx.x;

    //从 bid 和 tid 计算出这个 thread 应该计算的 row 和 column
    const int idx = bid * THREAD_NUM + tid;
    const int row = idx / n;
    const int column = idx % n;

    int i;

    //记录运算开始的时间
    clock_t start;

    //只在 thread 0（即 threadIdx.x = 0 的时候）进行记录，每个 block 都会记录开始时间及结束时间
    if (tid == 0) time[bid] = clock();

    //计算矩阵乘法
    if (row < n && column < n)
    {
        float t = 0;

        for (i = 0; i < n; i++)
        {
            t += a[row * n + i] * b[i * n + column];
        }
        c[row * n + column] = t;
    }

    //计算时间,记录结果，只在 thread 0（即 threadIdx.x = 0 的时候）进行，每个 block 都会记录开始时间及结束时间
    if (tid == 0)
    {
        time[bid + blocks_num] = clock();
    }
}





int main()
{

    //CUDA 初始化
    if (!InitCUDA()) return 0; 

    //定义矩阵
    float *a, *b, *c, *d;

    int n = MATRIX_SIZE;

    //分配内存
    a = (float*)malloc(sizeof(float)* n * n); 
    b = (float*)malloc(sizeof(float)* n * n); 
    c = (float*)malloc(sizeof(float)* n * n); 
    d = (float*)malloc(sizeof(float)* n * n);

    //设置随机数种子
    srand(0);

    //随机生成矩阵
    matgen(a, n);
    matgen(b, n);

    /*把数据复制到显卡内存中*/
    float *cuda_a, *cuda_b, *cuda_c;

    clock_t* time;

    //cudaMalloc 取得一块显卡内存 
    cudaMalloc((void**)&cuda_a, sizeof(float)* n * n);
    cudaMalloc((void**)&cuda_b, sizeof(float)* n * n);
    cudaMalloc((void**)&cuda_c, sizeof(float)* n * n);
    cudaMalloc((void**)&time, sizeof(clock_t)* blocks_num * 2);


    //cudaMemcpy 将产生的矩阵复制到显卡内存中
    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
    cudaMemcpy(cuda_a, a, sizeof(float)* n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(float)* n * n, cudaMemcpyHostToDevice);

    // 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
    matMultCUDA << < blocks_num, THREAD_NUM, 0 >> >(cuda_a , cuda_b , cuda_c , n , time);

    /*把结果从显示芯片复制回主内存*/

    clock_t time_use[blocks_num * 2];

    //cudaMemcpy 将结果从显存中复制回内存
    cudaMemcpy(c, cuda_c, sizeof(float)* n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_use, time, sizeof(clock_t)* blocks_num * 2, cudaMemcpyDeviceToHost);

    //Free
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);
    cudaFree(time);

    //把每个 block 最早的开始时间，和最晚的结束时间相减，取得总运行时间
    clock_t min_start, max_end;

    min_start = time_use[0];

    max_end = time_use[blocks_num];

    for (int i = 1; i < blocks_num; i++) 
    {
        if (min_start > time_use[i]) min_start = time_use[i];

        if (max_end < time_use[i + blocks_num]) max_end = time_use[i + blocks_num];
    }

    //核函数运行时间
    clock_t final_time = max_end - min_start;



    //CPU矩阵乘法，存入矩阵d
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        { 
            double t = 0;

            for (int k = 0; k < n; k++)
            { 

                t += a[i * n + k] * b[k * n + j]; 

            } 

            d[i * n + j] = t; 

        } 
    }

    //验证正确性与精确性

    float max_err = 0;

    float average_err = 0; 


    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            if (d[i * n + j] != 0)
            { 
                //fabs求浮点数x的绝对值
                float err = fabs((c[i * n + j] - d[i * n + j]) / d[i * n + j]);

                if (max_err < err) max_err = err; 

                average_err += err; 
            } 
        } 
    }

    printf("Max error: %g Average error: %g\n",max_err, average_err / (n * n));


    printf("gputime: %d\n", final_time);



return 0;

}

