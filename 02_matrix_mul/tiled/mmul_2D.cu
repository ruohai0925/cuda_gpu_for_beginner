// This program computes matrix multiplication using shared memory tiling
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
// CONSTANTS & CONFIGURATION
// =================================================================
// Matrix dimension N = 1024 (2^10)
const int N = 1 << 10;

// Shared Memory Size per Tile
// We assume a square block of 32x32 threads.
// 32 * 32 = 1024 integers.
// Updated Kernel using 2D Shared Memory Arrays
// Optimized for Readability

// TILE_DIM must be a compile-time constant for static declaration
const int TILE_DIM = 32; 

__global__ void matrixMul(const int *a, const int *b, int *c) {
    // Standard Global Coordinate Calculation
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. CHANGE: Define 2D arrays directly
    // This is much cleaner and logically matches the tile concept.
    // [32][32] corresponds to [Row][Col] inside the tile.
    __shared__ int s_a[TILE_DIM][TILE_DIM];
    __shared__ int s_b[TILE_DIM][TILE_DIM];

    int tmp = 0;

    // Loop over tiles
    for (int i = 0; i < N; i += TILE_DIM) {
        
        // 2. CHANGE: Loading Data (No more manual flattening!)
        // Notice how natural this looks: s_a[local_y][local_x]
        
        // Load Tile A
        s_a[threadIdx.y][threadIdx.x] = a[row * N + (i + threadIdx.x)];

        // Load Tile B
        s_b[threadIdx.y][threadIdx.x] = b[(i + threadIdx.y) * N + col];

        // Barrier
        __syncthreads();

        // 3. CHANGE: Computation (Clean 2D indexing)
        // We perform dot product on the local tiles.
        for (int k = 0; k < TILE_DIM; k++) {
            // Compare this to: s_a[threadIdx.y * TILE_DIM + k]
            // This clearly shows: Row of A * Column of B
            tmp += s_a[threadIdx.y][k] * s_b[k][threadIdx.x];
        }

        // Barrier
        __syncthreads();
    }

    c[row * N + col] = tmp;
}

// Check result on the CPU (Reference Implementation)
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) {
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
  // Size calculation
  size_t bytes = N * N * sizeof(int);

  // Use std::vector for RAII memory management on Host
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

  // Initialize with random data using lambda
  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  // Device pointers
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy Host -> Device
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  // Execution Configuration
  // We use 32x32 threads per block to match the tile size logic in the kernel.
  // 32 * 32 = 1024 threads.
  int THREADS = 32;
  int BLOCKS = N / THREADS;

  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Launch the Tiled Kernel
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);

  // Copy Device -> Host
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  // Verification
  verify_result(h_a, h_b, h_c);

  cout << "COMPLETED SUCCESSFULLY\n";

  // Cleanup
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
