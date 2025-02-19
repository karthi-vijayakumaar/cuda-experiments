#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

void readVectorFromFile(const char* filename, std::vector<int>& vec) {
    std::ifstream file(filename);
    int value;
    while (file >> value) {
        vec.push_back(value);
    }
}

__global__ void vector_add(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const char *filename1 = "vector1.txt";
    const char *filename2 = "vector2.txt";

    std::vector<int> vector1, vector2, result;

    readVectorFromFile(filename1, vector1);
    readVectorFromFile(filename2, vector2); 

    if (vector1.size() != vector2.size()) {
        std::cerr << "Vectors must have the same size." << std::endl;
        return 1;
    }

    int n = vector1.size();

    int *d_1, *d_2, *d_result;

    cudaMalloc((void**)&d_1, n * sizeof(int));
    cudaMalloc((void**)&d_2, n * sizeof(int));
    cudaMalloc((void**)&d_result, n * sizeof(int));

    cudaMemcpy(d_1, vector1.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_2, vector2.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_1, d_2, d_result, n);
    result.resize(n);
    cudaMemcpy(result.data(), d_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Result of vector addition:\n";
    for (int i = 0; i < n; i++) {
        std::cout << vector1[i] << " + " << vector2[i] << " = "<<result[i] << "\n";
    }

    // Free device memory
    cudaFree(d_1);
    cudaFree(d_2);
    cudaFree(d_result);
}

