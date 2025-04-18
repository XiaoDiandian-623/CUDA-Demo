
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
// Histogram
// grid(N/128) block(128)
// a:N*1 y:count histogram
__global__
void histogram(int* a,int* y,int N)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < N) {
        atomicAdd(&y[a[idx]],1);
    }
}
// Histogram + Vec4
// grid(N/128), block(128/4)
// a: Nx1, y: count histogram
__global__ void histogram_vec4(int* a, int* y, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    int4 reg_a = INT4(a[idx]);
    atomicAdd(&(y[reg_a.x]), 1);
    atomicAdd(&(y[reg_a.y]), 1);
    atomicAdd(&(y[reg_a.z]), 1);
    atomicAdd(&(y[reg_a.w]), 1);
  }
}