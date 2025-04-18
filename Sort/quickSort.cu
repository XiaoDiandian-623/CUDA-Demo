#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

// 定义块大小
const int BLOCK_SIZE = 256;

// 主机上的快速排序
void quickSortHost(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pivot = arr[high];
        int i = (low - 1);

        for (int j = low; j <= high - 1; ++j) {
            if (arr[j] < pivot) {
                i++;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[high]);

        int pi = i + 1;

        // 递归地对分区后的子数组进行排序
        quickSortHost(arr, low, pi - 1);
        quickSortHost(arr, pi + 1, high);
    }
}

// 设备上的快速排序内核
__global__ void quickSortKernel(int* arr, int* temp, int low, int high, int size) {
    __shared__ int shared[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= size) return;

    if (low < high) {
        int pivot = arr[high];
        int i = (low - 1);

        for (int j = low; j <= high - 1; ++j) {
            if (arr[j] < pivot) {
                i++;
                shared[tid] = arr[i];
                __syncthreads();
                arr[i] = arr[j];
                __syncthreads();
                arr[j] = shared[tid];
                __syncthreads();
            }
        }
        shared[tid] = arr[i + 1];
        __syncthreads();
        arr[i + 1] = arr[high];
        __syncthreads();
        arr[high] = shared[tid];
        __syncthreads();

        int pi = i + 1;

        // 递归地对分区后的子数组进行排序
        if (pi - 1 > low) {
            quickSortKernel<<<(pi - low + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(arr, temp, low, pi - 1, size);
        }
        if (high > pi + 1) {
            quickSortKernel<<<(high - pi + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(arr, temp, pi + 1, high, size);
        }
    }
}

// 打印数组
void printArray(const std::vector<int>& arr) {
    for (int num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
}

int main() {
    // 初始化随机数生成器
    srand(time(0));

    // 创建一个测试数组
    const int size = 1024 * 1024; // 1M 个元素
    std::vector<int> h_arr(size);

    // 填充数组
    for (int i = 0; i < size; ++i) {
        h_arr[i] = rand() % 1000;
    }

    // 分配设备内存
    int* d_arr;
    cudaMalloc(&d_arr, size * sizeof(int));
    cudaMemcpy(d_arr, h_arr.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    // 分配临时数组
    int* d_temp;
    cudaMalloc(&d_temp, size * sizeof(int));

    // 启动内核
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    quickSortKernel<<<blocks, BLOCK_SIZE>>>(d_arr, d_temp, 0, size - 1, size);

    // 复制结果回主机
    cudaMemcpy(h_arr.data(), d_arr, size * sizeof(int), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_arr);
    cudaFree(d_temp);

    // 打印排序后的数组
    std::cout << "Sorted array: ";
    printArray(h_arr);

    return 0;
}