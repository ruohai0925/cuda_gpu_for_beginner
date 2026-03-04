// This program computes a simple version of matrix multiplication
// C = A * B
// By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU

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
// __global__: Function executed on the Device (GPU), called from Host (CPU).
__global__ void matrixMul(const int *a, const int *b, int *c, int N) {
  // 1. 2D Thread Mapping (The GPS Logic)
  // We map the 2D grid of threads to the 2D matrix coordinates.
  // Standard CUDA convention: 
  //   y dimension -> rows
  //   x dimension -> columns
  //
  // Formula: Global_Coord = (Block_ID * Block_Size) + Thread_ID
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // 2. Boundary Check (Implicit Assumption)
  // In this specific example, N is 1024 and block size is 32. 
  // 1024 % 32 == 0, so we fit perfectly.
  // In a robust production kernel, we MUST add:
  // if (row < N && col < N) { ... }

  // 3. The Dot Product (Row of A dot Column of B)
  // We compute a single element C[row][col].
  // Intermediate sum register (lives in high-speed thread-private register)
  int temp_sum = 0; 

  for (int k = 0; k < N; k++) {
    // 4. Memory Access Pattern Analysis
    // ---------------------------------
    // Accessing A[row][k]:
    // Matrix A is stored in Row-Major order.
    // Index = row * N + k.
    // Since 'row' is constant for all threads in a Warp (same Y), 
    // and 'k' is the loop variable (same for all),
    // all threads in a Warp read the SAME memory address simultaneously.
    // Hardware Behavior: BROADCAST (Efficient).
    
    // Accessing B[k][col]:
    // Matrix B is stored in Row-Major order.
    // Index = k * N + col.
    // 'col' varies across the Warp (threadIdx.x 0 to 31).
    // The addresses are: Base + 0, Base + 1, ..., Base + 31.
    // Hardware Behavior: COALESCED ACCESS (Very Efficient).
    
    temp_sum += a[row * N + k] * b[k * N + col];
  }

  // 5. Write Back
  // Write the final accumulated sum to global memory.
  c[row * N + col] = temp_sum;
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

int main() {
  // Matrix size of 1024 x 1024 (2^10)
  int N = 1 << 10;

  // Size (in bytes) of matrix
  size_t bytes = N * N * sizeof(int);

  // Host vectors (Pageable memory by default)
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

  // Initialize matrices using a lambda function
  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  // Allocate device memory (VRAM)
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data to the device (Host -> Device)
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  // =============================================================
  // EXECUTION CONFIGURATION (The 2D Setup)
  // =============================================================
  
  // Threads per Block (CTA): 32 x 32 = 1024 threads
  // 1024 is the maximum threads per block for modern NV GPUs.
  int THREADS = 32;

  // Blocks per Grid: 1024 / 32 = 32 blocks in each dimension.
  // We use integer division here because we know N is a multiple of 32.
  // Robust formula: (N + THREADS - 1) / THREADS
  int BLOCKS = N / THREADS;

  // Use dim3 structs for block and grid dimensions
  // dim3 is a built-in CUDA vector type (x, y, z).
  // x maps to columns, y maps to rows.
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Launch kernel
  // The <<<grid, block>>> syntax now accepts dim3 structs.
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);

  // Copy back to the host (Device -> Host)
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  // Check result
  verify_result(h_a, h_b, h_c, N);

  cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}