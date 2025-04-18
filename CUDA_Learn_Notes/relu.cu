
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
// Relu x:N y:N y=max(0,x)
// grid(N/128), block(K=128) 
__global__ 
void relu(float* x,float* y,int N)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < N) {
        y[idx] = fmaxf(0.0f,x[idx]);
    } 
}

__global__
void relu_vec4(float* x,float* y,int N)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = fmaxf(0.0f,reg_x.x);
    reg_y.y = fmaxf(0.0f,reg_x.y);
    reg_y.z = fmaxf(0.0f,reg_x.z);
    reg_y.w = fmaxf(0.0f,reg_x.w);
    FLOAT4(y[idx]) = reg_y;
}

