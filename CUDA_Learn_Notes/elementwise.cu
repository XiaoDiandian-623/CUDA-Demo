
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
// Elementwise Add
// grid(N/128) block(128)
// a:N*1 b:N*1 c:N*1 c=elementwise_add(a,b)
__global__ 
void element_add(float* a, float* b,float* c,int N)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// Elementwise Add + vec4
// grid(N/128) block(128/4)
// a:N*1 b:N*1 c:N*1 c=elementwise_add(a,b)
__global__ 
void element_add_vec4(float* a,float* b,float* c,int N)
{
    int idx = 4 * threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < N) {
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FLOAT4(c[idx]) = reg_c;
    }   
}

// 上面的实现扩展性很差

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;
/**
* @brief Get the Num of Blocks
*
* @param n, num of elements
* @param num_blocks
* @return cudaError_t
*/
inline cudaError_t GetNumBlocks(int64_t n, int *num_blocks)
{
    int dev; // which device is currently being used.
    {
        cudaError_t err = cudaGetDevice(&dev);
        if (err != cudaSuccess)
        { return err; }
    }
    int sm_count; // Number of multiprocessors on the device，即流多处理器的数量
    {
        cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev); // information about the device.
        if (err != cudaSuccess)
        { return err; }
    }
    int tpm; // Maximum resident threads per multiprocessor，即每个流多处理器的最大驻留线程数
    {
        cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        if (err != cudaSuccess)
        { return err; }
    }
    *num_blocks = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize, sm_count * tpm / kBlockSize * kNumWaves));
    return cudaSuccess;
}

template <typename T, int pack_size>
struct GetPackType
{
    using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};

template <typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

/**
* @brief 判断类型T的内存长度是否符合预期
*
* @tparam T
* @tparam pack_size
*/
template <typename T, int pack_size>
union Pack
{
    static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size, "Memory size does not meet expectations");
    __device__ Pack()
    {
        // do nothing
    }
    PackType<T, pack_size> storage; // 占位用的
    T elem[pack_size];
};

template <typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed
{
    __device__ Packed()
    {
        // do nothing
    }
    union
    {
        T elem[pack_size]; // 这里联合体只有一个成员，应该是为了方便后期扩展
    };
};

constexpr int kMaxPackBytes = 128 / 8; //  CUDA 最多支持 128 个 bit 的访问粒度
constexpr int kMaxPackSize = 8;        // half 类型占 2 个字节，也就是 16 个 bit，所以最大可以 Pack 的数量为 128 / 16 = 8

constexpr int Min(int a, int b) 
{ 
    return a < b ? a : b; 
}

template <typename T>
constexpr int PackSize()
{
    return Min(kMaxPackBytes / sizeof(T), kMaxPackSize);
}

template <typename T, typename U, typename... Args>
constexpr int PackSize()
{
    return Min(PackSize<T>(), PackSize<U, Args...>());
}

template <int MaxPackSize>
constexpr int PackSize()
{
	return  MaxPackSize;
}

template <int MaxPackSize, typename T, typename... Args>
constexpr int PackSize()
{
	return Min(kMaxPackBytes / sizeof(T), PackSize<MaxPackSize, Args...>());
}
/**
* @brief 对一个 pack 内的元素做循环，对 elem 数组中的每个元素调用 functor ，得到输出结果并返回
* OneFlow 给 ApplyPack 函数特化了一个版本，通过调用 functor 的 apply2 函数，来调用 half2 相关特殊指令
* 
* @tparam pack_size 
* @tparam FunctorT 
* @tparam R 
* @tparam IN 
* @param functor 
* @param in 
* @return __device__ 
*/
template <int pack_size, typename FunctorT, typename R, typename... IN>
__device__ typename std::enable_if<HasApply2<FunctorT>::value == true && pack_size % 2 == 0, Packed<R, pack_size>>::type
ApplyPack(const FunctorT &functor, const Packed<IN, pack_size>... in)
{
    Packed<R, pack_size> ret;
    #pragma unroll
    for (int j = 0; j < pack_size; j += 2)
    {
        functor.Apply2(ret.elem + j, (in.elem + j)...);
    }
    return ret;
}

/**
* @brief 该模板的作用是在编译期判断类型 T 是否具有成员方法 Apply2
*    
* @tparam T
*/
template <typename T>
class HasApply2
{
    // 定义两个数据类型：one(占一个字节) 和 two(占2个字节)，用于成员方法的返回值类型
    typedef char one;
    struct two
    {
        char x[2];
    };
    // 声明两个test成员函数模板，利用了函数重载以及其调用顺序，这里两个重载test方法无需定义
    template <typename C>
    static one test(decltype(&C::Apply2));  // decltype用来获取参数的类型，表明这个函数的参数类型是decltype(&C::Apply2)，这里省略了形参所以不好理解
    template <typename C>
    static two test(...);   // 省略了参数，表示可变长参数

    public:
    /**
    * @brief 这里联合体只有一个成员，后续方便扩展
    *        value为true表示T中有成员函数Apply2，为false表示没有
    *        当T中有成员函数Apply2，test<T>(0)调用的是"static one test(decltype(&C::Apply2));"所以其返回值类型为one；否则
    * 将调用"static two test(...);"返回值为two
    */
    enum
    {
        value = sizeof(test<T>(0)) == sizeof(char) 
    };
};

template <typename FactoryT, typename R, typename... IN>
struct GenericLauncher
{
    static cudaError_t Launch(FactoryT factory, int64_t n, R *r, const IN *...in,
    cudaStream_t stream)
    {
        constexpr int max_pack_size = PackSize<R, IN...>();
        if (IsAlignedForPack<max_pack_size, R, IN...>(r, in...))
        {
        	return LaunchKernel<max_pack_size, FactoryT, R, IN...>(factory, n, r, in..., stream);
        }
        else
        {
        	return LaunchKernel<1, FactoryT, R, IN...>(factory, n, r, in..., stream);
        }
    }
};

template <size_t pack_size>
bool IsAlignedForPack()
{
	return true;
}

template <size_t pack_size, typename T, typename... Args>
/**
* @brief 判断类型 T 在 pack 后是否内存对齐
* 
* @param ptr 
* @param others 
* @return true 
* @return false 
*/
bool IsAlignedForPack(const T *ptr, const Args *...others)
{
    // 判断ptr地址是否能够整除 pack 后的 T 的大小，reinterpret_cast<uintptr_t>(ptr)将指针转换为一个 unsigned __int64
    return reinterpret_cast<uintptr_t>(ptr) % sizeof(Pack<T, pack_size>) == 0 && IsAlignedForPack<pack_size, Args...>(others...);
}

template <size_t pack_size, typename FactoryT, typename R, typename... IN>
/**
* @brief 启动核函数
* 
* @param factory 
* @param n 元素个数
* @param r 
* @param in 
* @param stream 
* @return cudaError_t 
*/
cudaError_t LaunchKernel(FactoryT factory, int64_t n, R *r, const IN *...in, cudaStream_t stream)
{
    const int64_t n_pack = n / pack_size; // 根据元素个数和pack_size，计算pack数目，比如1026 / 4 = 256。
    const int64_t tail_offset = n_pack * pack_size; // 如果存在不被整除的情况，我们计算使用pack的偏移量：256*4
    const int64_t n_tail = n - tail_offset; // 元素数目-偏移量 = 剩下的元素个数-> 1026-1024 = 2
    int num_blocks;
    {
    	cudaError_t err = GetNumBlocks(n_pack, &num_blocks);
        if (err != cudaSuccess)
        {
        	return err;
    	}
	}
	ApplyGeneric<pack_size, FactoryT, R, IN...><<<num_blocks, kBlockSize, 0, stream>>>(
factory, n_pack, reinterpret_cast<Packed<R, pack_size> *>(r),
(reinterpret_cast<const Packed<IN, pack_size> *>(in))..., n_tail, r + tail_offset,
(in + tail_offset)...);
	return cudaPeekAtLastError();
}

/**
* @brief 
*    __launch_bounds__(kBlockSize):为编译器指定每个线程块的最大线程数，优化寄存器使用
* @tparam pack_size 
* @tparam FactoryT 
* @tparam R 
* @tparam IN 
*/
template <int pack_size, typename FactoryT, typename R, typename... IN>
/**
* @brief Construct a new Apply Generic object
* 
* @param factory 
* @param n_pack num of packed data
* @param pack_r 
* @param pack_in 
* @param n_tail num of elements which are unpacked
* @param tail_r 
* @param tail_in 
*/
__global__ void __launch_bounds__(kBlockSize)
ApplyGeneric(FactoryT factory, int64_t n_pack, Packed<R, pack_size> *pack_r,
const Packed<IN, pack_size> *...pack_in, int64_t n_tail, R *tail_r,
const IN *...tail_in)
{
    auto functor = factory();
    const int global_tid = blockIdx.x * kBlockSize + threadIdx.x;
    for (int64_t i = global_tid; i < n_pack; i += blockDim.x * gridDim.x)
    {
    	pack_r[i] = ApplyPack<pack_size, decltype(functor), R, IN...>(functor, (pack_in[i])...);
    }
    if (global_tid < n_tail)
    {
    	tail_r[global_tid] = functor((tail_in[global_tid])...);
    }
}
template <int pack_size, typename FunctorT, typename R, typename... IN>
__device__ typename std::enable_if<HasApply2<FunctorT>::value == false || pack_size % 2 != 0,
Packed<R, pack_size>>::type
ApplyPack(const FunctorT &functor, const Packed<IN, pack_size>... in)
{
	Packed<R, pack_size> ret;
    #pragma unroll
    for (int j = 0; j < pack_size; ++j)
    {
    	ret.elem[j] = functor((in.elem[j])...);
    }
    return ret;
}
template<typename T>
struct SigmoidFunctor
{
    __device__ __host__ __forceinline__ T operator()(T x) const
    {
        return T(1.0) / (T(1.0) + expf(-x));
    }
};
cudaStream_t stream;
CHECK(cudaStreamCreate(&stream));
CHECK(Unary(SigmoidFunctor<float>(), N, d_out, d_in, stream));
CHECK(cudaStreamSynchronize(stream));
CHECK(cudaStreamDestroy(stream));