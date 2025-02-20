### Day 1 - Vector Addition

Reading 2 vectors from 2 files and adding them. Read the `vector_add.cu` file to understand the code and how it works.

Run the code using `nvcc vector_add.cu -o vector_add` and then execute the generated binary. I used kaggle to run the code, was facing some issues in colab.

Guide to running Cuda kernals from notebooks - [Blog](https://hamdi.bearblog.dev/learning-cuda-with-a-weak-gpu-or-no-gpu-at-all-yes-you-can/)

Mistakes to avoid
- forgetting to free the allocated device memory