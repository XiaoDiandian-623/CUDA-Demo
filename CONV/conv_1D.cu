#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"

#define MAX_MASK_WIDTH 10

__global__ void ConvGPU1D_basic(float* input, float* mask, float* output, int maskWidth, int width)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float tempVal = 0;
    int startPoint = idx - (maskWidth / 2);
    for (int i = 0; i < maskWidth; i++) {
        if (startPoint + i >= 0 && startPoint + i < width) {
            tempVal += input[startPoint + i] * mask[i];
        }
    }
    output[idx] = tempVal;
}


__constant__ float M[MAX_MASK_WIDTH];
__global__ void ConvGPU1D_constantMASK(float* input, float* output, int maskWidth, int width)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float tempVal = 0;
    int startPoint = idx - (maskWidth / 2);
    for (int i = 0; i < maskWidth; i++) {
        if (startPoint + i >= 0 && startPoint + i < width) {
            tempVal += input[startPoint + i] * M[i];
        }
    }
    output[idx] = tempVal;
}


void ConvCPU1D(float* input, float* mask, float* output, int maskWidth, int width)
{
    for (int idx = 0; idx < width; idx++) {
        float tempVal = 0;
        int startPoint = idx - (maskWidth / 2);
        for (int i = 0; i < maskWidth; i++) {
            if (startPoint + i >= 0 && startPoint + i < width) {
                tempVal += input[startPoint + i] * mask[i];
            }
        }
        output[idx] = tempVal;
    }
}

int main()
{
    int width = 1000;  
    int maskWidth = 5; 

    float* h_input = (float*)malloc(width * sizeof(float));
    float* h_mask = (float*)malloc(maskWidth * sizeof(float));
    float* h_outputCPU = (float*)malloc(width * sizeof(float));
    float* h_outputGPU = (float*)malloc(width * sizeof(float));

    // Initialize input and mask
    for (int i = 0; i < width; i++) {
        h_input[i] = (float)(i); 
    }

    for (int i = 0; i < maskWidth; i++) {
        h_mask[i] = (float)(i);  
    }

    ConvCPU1D(h_input, h_mask, h_outputCPU, maskWidth, width);

    float *d_input, *d_mask, *d_output;
    cudaMalloc((void**)&d_input, width * sizeof(float));
    cudaMalloc((void**)&d_mask, maskWidth * sizeof(float));
    cudaMalloc((void**)&d_output, width * sizeof(float));

   
    cudaMemcpy(d_input, h_input, width * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_mask, h_mask, maskWidth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M, h_mask, maskWidth * sizeof(float));  
    
    int blockSize = 256;  
    int numBlocks = (width + blockSize - 1) / blockSize;
    ConvGPU1D_constantMASK<<<numBlocks, blockSize>>>(d_input,d_output, maskWidth, width);

    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(h_outputGPU, d_output, width * sizeof(float), cudaMemcpyDeviceToHost);

   
    bool resultsMatch = true;
    for (int i = 0; i < width; i++) {
        if (abs(h_outputCPU[i] - h_outputGPU[i]) > 1e-5) {  
            resultsMatch = false;
            break;
        }
    }

    if (resultsMatch) {
        std::cout << "CPU and GPU results match!" << std::endl;
    } else {
        std::cout << "CPU and GPU results do not match." << std::endl;
    }

    free (h_input);
    free (h_mask);
    free (h_outputCPU);
    free (h_outputGPU);

    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);

    return 0;
}
