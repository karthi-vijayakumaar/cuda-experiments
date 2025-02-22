### Day - 4 Kernel to Multipy Matrices

The kernel is defined in `matrixMultiplication.cu`. Learnt stdio operations from claude. 

Mistakes to avoid:
- Remember this particular logic to add 2d matrices stored in 1d array

```
for (int k = 0; k < b; k++) {
            sum += A[row * b + k] * B[k * c + col];
        }
```