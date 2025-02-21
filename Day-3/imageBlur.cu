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

__global__ void blur(int *a, int *b, int height, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < height && col < width){
        int sum_of_surround = 0;
        int num_of_surround = 0;
        for(i=-1;i<=1;i++){
            for(j=-1;j<=1;j++){
            if (row+i>=0 && row+i < height && col+j >=0 && col+j < width) {
                    int offset = (row+i) * width + (col+j);
                    sum_of_surround += a[offset];
                    num_of_surround++;

                } 
            }
        }
        b[row * width + col] = sum_of_surround / num_of_surround;
    }
    
}

int main() {
    const char *filename1 = "image.txt";

    std::vector<int> vector1, result;

    readVectorFromFile(filename1, vector1);

    int n = vector1.size();
    int height=210, width=236;

    std::cout << "size of the vector is " << n << std::endl;
    std::cout << "size of the image is " << height << "x" << width << std::endl;

    int *d_1, *d_result;

    cudaMalloc((void**)&d_1, n * sizeof(int));
    cudaMalloc((void**)&d_result, n * sizeof(int));

    cudaMemcpy(d_1, vector1.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    blur<<<blocksPerGrid, threadsPerBlock>>>(d_1, d_result, height, width);
    result.resize(n);
    cudaMemcpy(result.data(), d_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Store the result in a space separated txt file
    std::ofstream file("result.txt");
    for (int i = 0; i < n; i++) {
        file << result[i] << " ";
    }
    file.close();

    std::cout << "Result of color to bw conversion:\n";
    for (int i = 0; i < n; i++) {
        std::cout <<result[i] << " ";
    }

    // Free device memory
    cudaFree(d_1);
    cudaFree(d_result);
}