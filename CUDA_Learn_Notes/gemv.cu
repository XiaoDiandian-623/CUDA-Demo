
// 矩阵向量乘 A[M,N] x[N,1]
#define CEIL(a,b) ((a+b-1)/(b))
__global__ void gemv_v1(float* A,float* x,float* y,int M,int K)
{
    int laneId = threadIdx.x % warpSize;
    int row = blockIdx.x; // 0~M-1
    if(row >= M) return;

    float res = 0.0f;
    int kIteration = CEIL(K,warpSize); // 每个线程需要负责计算的数据个数

    for(int i=0;i<kIteration;i++) {
        int col = i * warpSize + laneId;
        res += (col<K)?A[row*K+col]*x[col]:0.0f;
    }

    for(int offset=warpSize >> 1;offset>0;offset >>= 1) {
        res += __shfl_down_sync(0xFFFFFFFF,res,offset);
    }
    if(laneId==0) y[row]=res;
}

template<unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if(WarpSize >= 32) sum += __shfl_down_sync(0xffffffff,sum,16);
    if(WarpSize >= 16) sum += __shfl_down_sync(0xffffffff,sum,8);
    if(WarpSize >= 8) sum += __shfl_down_sync(0xffffffff,sum,4);
    if(WarpSize >= 4) sum += __shfl_down_sync(0xffffffff,sum,2);
    if(WarpSize >= 2) sum += __shfl_down_sync(0xffffffff,sum,1);
    return sum;
}   

// N=32
__global__ void sgemv_v0(
    float* __restrict__ A,
    float* __restrict__ x,
    float* __restrict__ y,
    const int M,const int N
)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int warp_size = 32;
    int laneId = tx % warp_size;
    int current_row = blockDim.y * bx + ty;

    if(current_row < M) {
        float res = 0;
        // int kIteration = N / warp_size;
        // if(kIteration==0) kIteration = 1;
        int kIteration = (N + warp_size - 1) / warp_size;  // 向上取整
        #pragma unroll
        for(int i=0;i<kIteration;i++) {
            int current_col = i*warp_size + laneId;
            if(current_col < N) {
                res += A[current_row*N+current_col] * x[current_col];
            }
            
        }
        res = warpReduceSum<warp_size>(res);
        if(laneId==0) y[current_row] = res;
    }
}

// N=128
template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); 
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);/
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}
__global__ void sgemv_v2(
    float* __restrict__ A,
    float* __restrict__ x,
    float* __restrict__ y,
    const int M,const int N
)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int warp_size = 32;
    int laneId = tx % warp_size;
    int current_row = blockDim.y * bx + ty;

    if(current_row<M){
        float res = 0;
        int kIteration = (N/warp_size)/4;
        if(kIteration == 0) kIteration = 1;
        A = &A[current_row*N];
        #pragma unroll
        for(int i=0;i<kIteration;i++) {
            int current_col_vec = (i*warp_size+laneId);
            float4 current_val = reinterpret_cast<float4*>(A)[current_col_vec];
            float4 current_x = reinterpret_cast<float4*>(x)[current_col_vec];

            res += current_val.x * current_x.x;
            res += current_val.y * current_x.y;
            res += current_val.z * current_x.z;
            res += current_val.w * current_x.w;

        }

        res = warpReduceSum<warp_size>(res);
        if(laneId==0) y[current_row] = res;
    }
}

// N = 16
template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); 
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);/
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

template<const int ROW_PER_WARP>
__global__ void sgemv_v3(
    float* __restrict__ A,
    float* __restrict__ x,
    float* __restrict__ y,
    const int M,const int N
)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int warp_size = 32;
    int laneId = tx % warp_size;
    int current_warp_row = (blockDim.y * bx + ty) * ROW_PER_WARP;
    const int kWarp_size = warp_size / ROW_PER_WARP;
    int kLaneId = laneId % kWarp_size;
    int current_thread_row = current_warp_row + laneId / kWarp_size;

    if(current_thread_row < M) {
        float res = 0;
        int current_col = kLaneId;
        res += A[current_thread_row * N + current_col] * x[current_col];
        res = warpReduceSum<kWarp_size>(res);
        if(kLaneId==0) y[current_thread_row] = res;
    }

}
