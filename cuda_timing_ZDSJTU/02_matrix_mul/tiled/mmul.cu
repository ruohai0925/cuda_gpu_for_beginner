// This program computes matrix multiplication using shared memory tiling
// By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU
// Profiling support: accepts N (matrix size) and M (iterations) from command line

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

// =================================================================
// CONSTANTS & CONFIGURATION
// =================================================================
// Shared Memory Size per Tile: 32 * 32 = 1024 integers (fixed tile dimension)
const int SHMEM_SIZE = 1 << 10;

// =================================================================
// GPU KERNEL: Tiled Implementation using Shared Memory (1D flattened)
// =================================================================
__global__ void matrixMul(const int *a, const int *b, int *c, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int s_a[SHMEM_SIZE];
  __shared__ int s_b[SHMEM_SIZE];

  int tmp = 0;

  // Tiling loop: sweep across the K dimension in tiles of blockDim.x (32)
  for (int i = 0; i < N; i += blockDim.x) {
    // Collaborative loading into shared memory
    int local_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // Load Tile A: boundary check for safety
    if (row < N && (i + threadIdx.x) < N)
      s_a[local_idx] = a[row * N + i + threadIdx.x];
    else
      s_a[local_idx] = 0;

    // Load Tile B: boundary check for safety
    if ((i + threadIdx.y) < N && col < N)
      s_b[local_idx] = b[(i + threadIdx.y) * N + col];
    else
      s_b[local_idx] = 0;

    __syncthreads();

    // Compute partial dot product from shared memory
    for (int j = 0; j < blockDim.x; j++) {
      tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    __syncthreads();
  }

  // Write back with boundary check
  if (row < N && col < N) {
    c[row * N + col] = tmp;
  }
}

// Check result on the CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        tmp += a[i * N + k] * b[k * N + j];
      }
      assert(tmp == c[i * N + j]);
    }
  }
}

int main(int argc, char *argv[]) {
  // Parse command-line arguments: N (matrix size) and M (timed iterations)
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <N> <M>\n", argv[0]);
    return 1;
  }
  int N = atoi(argv[1]);
  int M = atoi(argv[2]);

  size_t bytes = N * N * sizeof(int);

  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  // Execution configuration with ceiling division
  int THREADS = 32;
  int BLOCKS = (N + THREADS - 1) / THREADS;
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Warmup launch
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();

  // Verify correctness once
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
  verify_result(h_a, h_b, h_c, N);

  // Timed iterations (kernel-only, excluding memcpy)
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float total_ms = 0.0f;

  for (int i = 0; i < M; i++) {
    cudaEventRecord(start);
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    total_ms += ms;
  }

  // Output average time in ms
  printf("%f\n", total_ms / M);

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
