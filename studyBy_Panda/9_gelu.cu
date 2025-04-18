//fp16 实现gelu算子
#include<bits/stdc++.h>
#include<cuda.h>
#include<cuda_fp16.h>
#include"cuda_runtime.h"

template<typename T,int Size>
struct alignas(sizeof(T) * Size) AlignedVector{
    T val[Size];
    __host__ __device__ inline const T& operator[](int i) const {return val[i];}
    __host__ __device__ inline T& operator[](int i) {return val[i]; }
};

__device__ float TanhApprox(float x)
{

    return tanhf(x);
}

//gelu公式：x / 2 * (1 + tan(0.7978845608028654 * (x + 0.044714998453855515 * x^3)))
template<typename T>
struct GeluFunctor {
  static constexpr T alpha = static_cast<T>(0.7978845608028654);
  static constexpr T beta = static_cast<T>(0.044714998453855515);

  __device__ GeluFunctor() {};

  __device__ T operator()(T x) const {
    const T half = static_cast<T>(0.5);
    const T one = static_cast<T>(1);
    const T tanh_in = alpha * (x + beta * x * x * x);
    return half * x * (one + tanh(tanh_in));
  }
};

template<>
struct GeluFunctor<half> {
 
  static constexpr float alpha = GeluFunctor<float>::alpha;
  static constexpr float beta = GeluFunctor<float>::beta;
  GeluFunctor<float> float_functor;

  __device__ GeluFunctor() {};

  __device__ half operator()(const half x) const {
   
    return static_cast<half>(float_functor(static_cast<float>(x)));
  }
  
};

template<int VecSize>
__global__ void FP16GeluCUDAKernel(const __half* x,__half* y,int n) {

    //计算每一个线程所需处理的元素地址
    int offset = static_cast<int>(threadIdx.x+blockIdx.x*blockDim.x)*VecSize;
    int stride = static_cast<int>(blockDim.x*gridDim.x)*VecSize;//步长
    GeluFunctor<half>gelu_fwd;
    __half y_reg[VecSize];

    for(;offset<64;offset+=stride) {
       
        using ArrT = AlignedVector<__half, VecSize>;
        const ArrT* in_arr = reinterpret_cast<const ArrT*>(x + offset);
        const __half* in = reinterpret_cast<const __half*>(in_arr);
   
        if(VecSize == 1) {
            y_reg[0] = gelu_fwd(in[0]);
        }else {
            for(int i=0;i<VecSize;i++) {
                y_reg[i] = gelu_fwd(in[i]);
            }
        }
        *reinterpret_cast<ArrT*>(y+offset) = *reinterpret_cast<ArrT*>(y_reg);
    }

}

int main()
{
    int n=1000;
    __half* x = new __half[n];
    __half* y = new __half[n];

    for(int i=0;i<n;i++) {
        x[i] = (__half)(i);
    }
    __half* d_x,*d_y;
    cudaMalloc((void**)&d_x,n*sizeof(__half));
    cudaMalloc((void**)&d_y,n*sizeof(__half));
    cudaMemcpy(d_x,x,sizeof(__half)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,y,sizeof(__half)*n,cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,0);

    //满足内存对齐要求
    auto is_aligned = [](const void* p,int alignment) {
        return reinterpret_cast<uintptr_t>(p) % alignment == 0;
    };

  
    constexpr auto kAlignment = alignof(AlignedVector<__half,0>);
    if(n%8==0 && is_aligned(x,kAlignment) && is_aligned(y,kAlignment)) {
        int thread = std::min<int>(512,deviceProp.maxThreadsPerBlock);

        int block = (n + thread - 1)/ thread;
        block = std::min<int>(block,deviceProp.maxGridSize[0]);

        FP16GeluCUDAKernel<1><<<block,thread>>>(d_x,d_y,n);
        cudaMemcpy(y,d_y,sizeof(__half)*n,cudaMemcpyDeviceToHost);
    }
    printf("pass\n");
    delete x;
    x = nullptr;
    delete y;
    y = nullptr;
    cudaFree(d_x);
    cudaFree(d_y);
}