### Day - 2 Kernel to convert a RGB image to Grayscale

The kernel is defined in `color_to_bw.cu`. Learnt how to launch a 2d grid of threads. 

Mistakes to avoid 
- shape mismatch while declaring 2d grid of threads.
- Declaring result same size as the input