// A simple example to demonstrate speedup by using cuda.

#include <iostream>
#include <math.h>

// Kernel function to add the elements of two arrays in a single thread.
__global__ void add(int n, float* x, float* y) {
  for (int i = 0; i < n; ++i) {
    y[i] += x[i];
  }
}

// Kernel function to add the elements in one block with many threads.
__global__ void add_threads(int n, float* x, float* y) {
  int index = threadIdx.x;  // the index of the current thread within its block.
  int stride = blockDim.x;  // the number of threads in the block
  for (int i = index; i < n; i += stride) {
    y[i] += x[i];
  }
}

// Kernel function to add the elements in multiple blocks with many threads.
__global__ void add_block_threads(int n, float* x, float* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] += x[i];
  }
}

int main() {
  int N = 1<<20;
  float* x;
  float* y;

  // Allocate unified memory - accessible from cpu to gpu
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // Initialize x and y arrays on the host
  for (int i = 0; i < N; ++i) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run the kernel on 1M elements on the GPU.
  add<<<1, 1>>>(N, x, y);  // one block and one thread

  // Run the kernel with one block and 256 threads.
  add_threads<<<1, 256>>>(N, x, y);

  // Run the kernel with many blocks and threads.
  int block_size = 256;
  int num_blocks = (N  + block_size - 1)/ block_size;
  add_block_threads<<<num_blocks, block_size>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; ++i) {
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  }
  std::cout << "Max error: " << maxError << '\n';

  // Free memory
  cudaFree(x);
  cudaFree(y);
}
