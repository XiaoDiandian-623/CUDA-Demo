

// C(MxN) = A(MxK) * B(KxN) 行优先
#define CEIL(a,b) ((a+b-1)/(b))
#define OFFSET(row,col,ld) ((row)*(ld)+(col))
void gemm_cpu(float* A,float* B,float* C,int M,int N,int K)
{
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            float sum = 0.0f;
            
            for(int r=0; r<K; r++){
                sum += A[i*K + r] * B[r*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

__global__ void naive_sgemm(float* A,float* B,float* C,int M,int N,int K) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if(row>=M || col>=N) return;

    float accum = 0;
    for(int i=0;i<K;i++) {
        accum += A[OFFSET(row,i,K)] * B[OFFSET(i,col,N)];
        // accum += A[row*K+i] * B[i*N+col];
    }
    C[OFFSET(row,col,N)] = accum;
    // C[row*N+col] = accum;
}

// 引入 block tile,shared mem作缓存
template<int BLOCK_SIZE>
__global__ void sgemm_v1(float* A,float* B,float* C,int M,int N,int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if(idx>=M || idy>=N) return;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;
    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];

    // 初始化block tile起始位置
    A = &A[(by*BM)*K]; // 定位到当前块在A矩阵中的起始行
    B = &B[bx*BN]; // 定位到当前块在B矩阵中的起始列
    C = &C[(by*BM)*N+bc*BN]; // 定位到当前块在C矩阵中的起始位置
    float accum = 0.0f;
    for(int k=0;k<K;k+=BK) {
        As[ty*BK+tx] = A[ty*K+tx];
        Bs[ty*BN+ty] = B[ty*N+tx];
        __syncthreads();
        A = A + BK;
        B = B + BK* N;
        for(int i=0;i<BK;i++) {
            accum += As[ty*BK+i] * Bs[i*BN+tx];
        }
        __syncthreads();
    }
    C[ty*N+tx] = accum;
}


// thread tile 一个线程承担更多计算
template<const int BM,
        const int BK,
        const int BN,
        const int TM, // thread tile的高
        const int TN> // thread tile的宽
__global__ void sgemm_v2(float* A,float* B,float* C,int M,int N,int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    int thread_num = block_col_thread * block_row_thread;

    int tx = (threadIdx.x % block_col_thread) * TN;
    int ty = (threadIdx.y % block_row_thread) * TM;
     
    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];

    A = &A[by*BM*K];
    B = &B[bx*BN];
    C = &C[by(BM*N + bx*BN)];

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;  // BM/(BM/(thread_num/BK)) = thread_num/BK = stride

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float accum[TM][TN] = {0.0f};

    for (int k = 0; k < K; k += BK) {
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
 
        A += BK;
        B += BK * N;
 
        for (int row = 0; row < TM; row++) {
            for (int col = 0; col < TN; col++) {
                for (int i = 0; i < BK; i++) {
                     accum[row][col] += As[(ty + row) * BK + i] * Bs[i * BN + (tx + col)];
                }
            }
        }
        __syncthreads();
    }
    for (int row = 0; row < TM; row++) {
        for (int col = 0; col < TN; col++) {
            C[(ty + row) * N + (tx + col)] = accum[row][col];
        }
    }


}