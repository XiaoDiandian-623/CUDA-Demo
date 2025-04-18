// note: To make softmax better
#include <stdio.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include "cuda_runtime.h"


// naive softmax
template<int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void naive_softmax(
    const float * __restrict__ x,
    float * __restrict__ y,
    int V
) 
{
    int tid = threadIdx.x;
    int vector_id = blockIdx.x;

    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<float,THREADBLOCK_SIZE> BloceReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float d_total_inverse;

    float d_partial = 0.0f;
    for(int elem_id = tid;elem_id<V;elem_id+=THREADBLOCK_SIZE)
        d_partial += __expf(x[elem_id]);
    
    float d = BlockReduce(temp_storage).Sum(d_partial);
    if(tid == 0) 
        d_total_inverse = __fdividef(1.0f,d);
    __syncthreads();

    for(int elem_id = tid;elem_id < V;elem_id += THREADBLOCK_SIZE) 
        y[elem_id] = __expf(x[elem_id]) * d_total_inverse;
}

// safe softmax 三遍扫描：求最大值 计算归一化因子 计算最终输出
__device__ __forceinline__ float max_op(float a,float b) {
    return fmax(a,b);
}
template<int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void safe_softmax(
    const float* __restrict__ x,
    float& __restrict__ y,
    int V
)
{
    int tid = threadIdx.x;
    int vector_id = blockIdx.x;
    // 定位到当前block的指针
    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<float,THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float m_total;
    __shared__ float d_total_inverse;

    float m_partial = -FLT_MAX;
    for(int elem_id=tid;elem_id<V;elem_id+=THREADBLOCK_SIZE) 
        m_partial = max_op(m_partial,x[elem_id]);
    
    float m = BlockReduce(temp_storage).Reduce(m_partial,max_op);
    if(tid==0) 
        m_total = m;
    __syncthreads();

    float d_partial = 0.0f;
    for(int elem_id=tid;elem_id<V;elem_id+=THREADBLOCK_SIZE) 
        d_partial += __expf(x[elem_id]-m_total);
    
    float d = BlockReduce(temp_storage).Sum(d_partial);
    if(tid==0)
        d_total_inverse = __fdivide(1.0f,d);
    __syncthreads();

    for(int elem_id = tid;elem_id < V;elem_id += THREADBLOCK_SIZE) {
        y[elem_id] = __expf(x[elem_id]-m_total) * d_total_inverse;
    }
}

// online softmax 将前两遍扫描合并为一遍扫描 通过构造并规约MD对象同时计算最大值和累加归一化项
struct __align__(8) MD 
{
    float m;  // 用来保存局部“最大值”
    float d;  // 用来保存归一化因子，即归一化时需要的累加项（累加 exp( x – m ) ）
};
// “合并”两个 MD 结构，得到同时代表这两部分数据的最大值和归一化因子。其核心思路
__device__ __forceinline__ MD reduce_md_op(MD a,MD b) {
    bool a_bigger = (a.m > b.m);
    MD bigger_m = a_bigger ? a : b;
    MD smaller_m = a_bigger ? b : a;
    MD res; 
    // dj = exp^(xj-mj)+dj-1 * exp^(mj-1 - mj)
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m-bigger_m.m);
    res.m = bigger_m.m;
    return res;
}
template<int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) 
__global__ void online_softmax(
    const float * __restrict__ x,
    float * __restrict__ y,
    int V
) 
{
    int tid = threadIdx.x;
    int vector_id = blockIdx.x;

    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<MD,THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ MD md_total;
    
    // 每个线程计算局部MD值
    MD md_partial;
    md_partial.m = -FLT_MAX;
    md_partial.d = 0.0f;
    for(int elem_id = tid;elem_id < V;elem_id += THREADBLOCK_SIZE) {
        MD new_elem;
        new_elem.m = x[elem_id]; // 一次遍历数据即可完成d,m的动态更新 相比原始softmax 有效降低访存压力 潜在增加了缓存的利用率
        new_elem.d = 1.0f;
        md_partial = reduce_md_op(md_partial,new_elem);
    }

    MD md = BlockReduce(temp_storage).Reduce(md_partial,reduce_md_op);
    if(tid == 0) {
        md_total = md;
    }
    __syncthreads();

    float d_total_inverse = __fdividef(1.0f,md_total.d);
    for(int elem_id=tid;elem_id<V;elem_id+=THREADBLOCK_SIZE) {
        y[elem_id] = __expf(x[elem_id]-md_total.m) * d_total_inverse;
    }

}

// safe softmax fused Top-k
template<int MAX_K>
struct TopK
{
    int p[MAX_K]; // topk索引
    float u[MAX_K]; // topk向量

    __device__ __forceinline__ void insert(float elem,int elem_id) {
        if(elem > u[MAX_K]) {
            u[MAX_K] = elem;
            p[MAX_K] = elem_id;
        }
        for(int k=MAX_K-2;k>=0;--k) { //冒泡排序的思想 将新的elem插入到合适位置
            if(u[k+1]>u[k]) {
                float u2 = u[k];
                int p2 = p[k];
                u[k] = u[k+1];
                u[k+1] = u2;
                p[k] = p[k+1];
                p[k+1] = p2;
            }
        }
    }
};

template<int MAX_K>
__device__ __forceinline__ TopK<MAX_K> reduce_topk_op(const TopK<MAX_K>& a,const TopK<MAX_K>& b) {
    // 合并为一个topk
    TopK<MAX_K> res = a;
    for(int i=0;i<MAX_K;i++) {
        res.insert(b.u[i],b.p[i]);
    }
    return res;
}

template<int MAX_K,int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void topk(
    const float * __restrict__ y,
    int * __restrict__ z,
    float * __restrict__ v,
    int V,
    int K
) 
{
    int tid = threadIdx.x;
    int vector_id = blockIdx.x;

    y += vector_id * V;
    
    typedef cub::BlockReduce<TopK<MAX_K>,THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopK<MAX_K> partial;
    for(int i=0;i<MAX_K;i++) {
        partial.p[i] = 1;
    }
    for(int i=0;i<MAX_K;i++) {
        partial.u[i] = -FLT_MAX;
    }
    for(int elem_id=tid;elem_id<V;elem_id+=THREADBLOCK_SIZE) {
        float elem = y[elem_id];
        partial.insert(elem,elem_id);
    }

    TopK<MAX_K> total = BlockReduce(temp_storage).Reduce(partial,reduce_topk_op<MAX_K>);
    if(tid == 0) {
        z += vector_id * K; // 乘以K是通过配置连续的内存空间来防止输出向量的值被覆盖
        v += vector_id * K;

        for(int i=0;i<MAX_K;++i) {
            if(i<K) {
                z[i] = total.p[i];
                v[i] = total.u[i];
            }
        }
    }
}

template<int MAX_K>
struct TopKD{
    float d;
    TopK<MAX_K> topk;
};
template<int MAX_K>
__device__ __forceinline__ TopKD<MAX_K> reduce_topk_d_op(const TopKD<MAX_K>& a,const TopKD<MAX_K>& b) {
    TopKD<MAX_K> res;
    res.d = a.d * b.d;
    res.topk = reduce_topk_op(a.topk,b.topk);
    return res;
}
template<int MAX_K,int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void safe_softmax_topk(
    const float * __restrict__ x,
    int * __restrict__ z,
    float * __restrict__ v,
    int V,
    int K
)
{
    int tid = threadIdx.x;
    int vector_id = blockIdx.x;
    x += vector_id * V;

    typedef cub::BlockReduce<float,THREADBLOCK_SIZE> maxValBlockReduce;
    typedef cub::BlockReduce<TopKD<MAX_K>,THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename maxValBlockReduce::TempStorage max_val_temp_storage;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float m_total;

    float m_partial = -FLT_MAX;
    for(int elem_id = tid;elem_id < V;elem_id += THREADBLOCK_SIZE) 
        m_partial = max_op(m_partial,x[elem_id]);
    float m = maxValBlockReduce(max_val_temp_storage).Reduce(m_partial,max_op);
    if(tid == 0) 
        m_total = m;
    __syncthreads();

    TopKD<MAX_K> partial;
    for(int i=0;i<MAX_K;++i) {
        partial.topk.p[i] = -1;
    }
    for(int i=0;i<MAX_L;++i) {
        partial.topk.u[i] = -FLT_MAX;
    }
    partial.d = 0.0f;
    for(int elem_id = tid;elem_id<V;elem_id+=THREADBLOCK_SIZE) {
        float elem = x[elem_id];
        partial.d += __expf(elem-m_total);
        partial.topk.insert(elem,elem_id); // 计算softmax分子指数和更新向量的topk
    }

    TopKD<MAX_K> total = BlockReduce(temp_storage).Reduce(partial,reduce_topk_d_op<MAX_K>);

    if(tid == 0) {
        z += vector_id * K;
        v += vector_id * K;

        float d_total_inverse = __fdividef(1.0f,total.d);
        for(int i=0;i<MAX_K;++i) {
            float val = __expf(total.topk.u[i]-m_total) * d_total_inverse;
            if(i < K) {
                z[i] = total.topk.p[i];
                v[i] = val;
            }
        }
    }

}



// online fused topk 每个输入元素只进行一次内存访问
struct __align__(8) MD 
{
    float m;  // 用来保存局部“最大值”
    float d;  // 用来保存归一化因子，即归一化时需要的累加项（累加 exp( x – m ) ）
};
// “合并”两个 MD 结构，得到同时代表这两部分数据的最大值和归一化因子。其核心思路
__device__ __forceinline__ MD reduce_md_op(MD a,MD b) {
    bool a_bigger = (a.m > b.m);
    MD bigger_m = a_bigger ? a : b;
    MD smaller_m = a_bigger ? b : a;
    MD res; 
    // dj = exp^(xj-mj)+dj-1 * exp^(mj-1 - mj)
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m-bigger_m.m);
    res.m = bigger_m.m;
    return res;
}
template<int MAX_K>
struct TopK
{
    int p[MAX_K]; // topk索引
    float u[MAX_K]; // topk向量

    __device__ __forceinline__ void insert(float elem,int elem_id) {
        if(elem > u[MAX_K]) {
            u[MAX_K] = elem;
            p[MAX_K] = elem_id;
        }
        for(int k=MAX_K-2;k>=0;--k) { //冒泡排序的思想 将新的elem插入到合适位置
            if(u[k+1]>u[k]) {
                float u2 = u[k];
                int p2 = p[k];
                u[k] = u[k+1];
                u[k+1] = u2;
                p[k] = p[k+1];
                p[k+1] = p2;
            }
        }
    }
};
template<int MAX_K>
__device__ __forceinline__ TopK<MAX_K> reduce_topk_op(const TopK<MAX_K>& a,const TopK<MAX_K>& b) {
    // 合并为一个topk
    TopK<MAX_K> res = a;
    for(int i=0;i<MAX_K;i++) {
        res.insert(b.u[i],b.p[i]);
    }
    return res;
}

template<int MAX_K>
struct TopKMD
{   // 将向量最大值和指数和结果合并在结构体中
    MD md;
    TopK<MAX_K> topk;
};
template<int MAX_K>
__device__ __forceinline__ TopKMD<MAX_K> reduce_topk_md_op(const TopKMD<MAX_K> & a,const TopKMD<MAX_K> & b)
{   // 合并结构体和topk
    TopKMD<MAX_K> res;
    res.md = reduce_md_op(a.md,b.md);
    res.topk = reduce_topk_op(a.topk,b.topk);
    return res;
}

template<int MAX_K,int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void online_softmax_topk(
    const float * __restrict__ x,
    int * __restrict__ z,
    float * __restrict__ v,
    int V,
    int K
)
{
    int tid = threadIdx.x;
    int vector_id = blockIdx.x;

    // reposition y to data for the current vector
    x += vector_id * V;

    typedef cub::BlockReduce<TopKMD<MAX_K>,THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    // 每个线程部分结果的初始化
    TopKMD<MAX_K> partial;
    for(int i=0;i<MAX_K;i++)
        partial.topk.p[i] = -1;
    for(int i=0;i<MAX_K;i++) 
        partial.topk.u[i] = -FLT_MAX;
    partial.md.m = -FLT_MAX;
    partial.md.d = 0.0f;
    for(int elem_id=tid;elem_id<V;elem_id+=THREADBLOCK_SIZE) {
        float elem = x[elem_id];
        MD new_elem{elem,1.0f};
        partial.md = reduce_md_op(partial.md,new_elem);
        partial.topk.insert(elem,elem_id);
    }

    TopKMD<MAX_K> total = BlockReduce(temp_storage).Reduce(partial,reduce_topk_md_op<MAX_K>);
    if(tid == 0) {
        z += vector_id * K;
        v += vector_id * K;

        float d_total_inverse = __fdividef(1.0f,total.md.d);
        for(int i=0;i<MAX_K;i++) {
            float val = __expf(total.topk.u[i]-total.md.m) * d_total_inverse;
            if(i < K) {
                z[i] = total.topk.p[i];
                v[i] = val;
            }
        }
    }
}


