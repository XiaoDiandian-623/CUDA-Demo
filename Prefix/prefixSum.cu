#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA Kernel
__global__ void prefixSumCUDA(int* d_input, int* d_output, int n) {
    extern __shared__ int temp[]; 
    int tid = threadIdx.x;
    int offset = 1;

    int idx = 2 * tid;
    if (idx < n) {
        temp[idx] = d_input[idx];
        if (idx + 1 < n) {
            temp[idx + 1] = d_input[idx + 1];
        }
    }
    __syncthreads();


    for (int d = n >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
        __syncthreads();
    }


    if (tid == 0) {
        temp[n - 1] = 0;
    }
    __syncthreads();

    // 下降阶段：从根节点回到叶节点更新值
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        __syncthreads();
    }

    // 将共享内存数据写回全局内存
    if (idx < n) {
        d_output[idx] = temp[idx];
        if (idx + 1 < n) {
            d_output[idx + 1] = temp[idx + 1];
        }
    }
}




void prefixSumCUDAWrapper(const std::vector<int>& input, std::vector<int>& output) {
    int n = input.size();
    int* d_input;
    int* d_output;

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    
    cudaMemcpy(d_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice);


    int blockSize = (n + 1) / 2; // 每个线程负责两个元素
    prefixSumCUDA<<<1, blockSize, n * sizeof(int)>>>(d_input, d_output, n);


    cudaMemcpy(output.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost);


    cudaFree(d_input);
    cudaFree(d_output);
}

// CPU实现
void prefixSumCPU(const std::vector<int>& input, std::vector<int>& output) {
    //output[0] = input[0];
    size_t sum = 0;
    for (size_t i = 0; i < input.size(); i++) {
        sum += input[i];
        output[i] = sum;
        //output[i] = output[i - 1] + input[i];
    }
}

// 主函数：测试CPU和GPU
int main() {
    const int n = 8; // 数组大小，假设为2的幂次
    std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> cpu_output(n);
    std::vector<int> gpu_output(n);

    // CPU计算前缀和
    prefixSumCPU(input, cpu_output);

    // GPU计算前缀和
    prefixSumCUDAWrapper(input, gpu_output);

    // 打印结果
    std::cout << "Input: ";
    for (int v : input) {
        std::cout << v << " ";
    }
    std::cout << "\n";

    std::cout << "Prefix Sum (CPU): ";
    for (int v : cpu_output) {
        std::cout << v << " ";
    }
    std::cout << "\n";

    std::cout << "Prefix Sum (GPU): ";
    for (int v : gpu_output) {
        std::cout << v << " ";
    }
    std::cout << "\n";

    // 验证结果是否一致
    bool is_correct = true;
    for (size_t i = 0; i < n; ++i) {
        if (cpu_output[i] != gpu_output[i]) {
            is_correct = false;
            break;
        }
    }

    if (is_correct) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    return 0;
}
