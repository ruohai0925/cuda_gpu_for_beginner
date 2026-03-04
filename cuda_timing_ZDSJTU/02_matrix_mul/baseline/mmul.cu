// This program computes a simple version of matrix multiplication
// C = A * B
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
// GPU KERNEL: Naive Implementation (Global Memory Only)
// =================================================================
__global__ void matrixMul(const int *a, const int *b, int *c, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary check for non-multiple-of-32 sizes
  if (row < N && col < N) {
    int temp_sum = 0;
    for (int k = 0; k < N; k++) {
      temp_sum += a[row * N + k] * b[k * N + col];
    }
    c[row * N + col] = temp_sum;
  }
}

// Check result on the CPU (Serial verification)
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

  // Size (in bytes) of matrix
  size_t bytes = N * N * sizeof(int);

  // Host vectors
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

  // Initialize matrices
  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  // Allocate device memory
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data to device
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

  // Output average time in ms (parsed by profiling script)
  printf("%f\n", total_ms / M);

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
