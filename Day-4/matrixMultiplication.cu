#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(float* A, float* B, float* C, int a, int b, int c) {
    // Calculate the row and column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if the thread is within matrix bounds
    if (row < a && col < c) {
        float sum = 0.0f;
        // Each thread computes one element of the result matrix
        for (int k = 0; k < b; k++) {
            sum += A[row * b + k] * B[k * c + col];
        }
        C[row * c + col] = sum;
    }
}

// Helper function to initialize a matrix with random values
void initializeMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() / (float)RAND_MAX;
    }
}

// Helper function to print a matrix
void printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Matrix dimensions
    int a = 3;  // rows of A
    int b = 4;  // cols of A and rows of B
    int c = 2;  // cols of B

    // Size calculations
    size_t size_A = a * b * sizeof(float);
    size_t size_B = b * c * sizeof(float);
    size_t size_C = a * c * sizeof(float);


    // Host matrices
    float *h_A, *h_B, *h_C;
    
    // Allocate host memory
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    // Initialize input matrices
    initializeMatrix(h_A, a, b);
    initializeMatrix(h_B, b, c);

    // Device matrices
    float *d_A, *d_B, *d_C;
    
    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);  // 16x16 threads per block
    dim3 gridDim(
        (c + blockDim.x - 1) / blockDim.x,
        (a + blockDim.y - 1) / blockDim.y
    );

    // Launch kernel
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, a, b, c);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Print results
    printf("Matrix A (%dx%d):\n", a, b);
    printMatrix(h_A, a, b);
    
    printf("Matrix B (%dx%d):\n", b, c);
    printMatrix(h_B, b, c);
    
    printf("Result Matrix C (%dx%d):\n", a, c);
    printMatrix(h_C, a, c);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}