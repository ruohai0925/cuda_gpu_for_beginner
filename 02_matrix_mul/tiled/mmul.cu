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
const int SHMEM_SIZE = 1 << 10;

// =================================================================
// GPU KERNEL: Tiled Implementation using Shared Memory
// =================================================================
__global__ void matrixMul(const int *a, const int *b, int *c) {
  // 1. Thread & Block Coordinate Mapping
  // ------------------------------------
  // Calculate the global row and column index for this thread.
  // This determines which element C[row][col] this thread is responsible for computing.
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // 2. Shared Memory Allocation (The "Scratchpad")
  // ----------------------------------------------
  // __shared__: Allocates memory in the on-chip Shared Memory (L1 Cache).
  // Scope: Visible to all threads within the SAME block.
  // Speed: ~100x faster than Global Memory.
  // Note: We use a 1D array to simulate a 2D tile (32x32 flattened).
  __shared__ int s_a[SHMEM_SIZE];
  __shared__ int s_b[SHMEM_SIZE];

  // Register to accumulate the partial dot product result.
  // Registers are the fastest memory (zero latency).
  int tmp = 0;

  // 3. The Tiling Loop (Sweeping across the matrix)
  // -----------------------------------------------
  // Instead of iterating k = 0 to N one by one, we stride by 'blockDim.x' (Tile Width).
  // Concept: The thread block moves horizontally across A and vertically down B,
  // processing one "Tile" (sub-matrix) at a time.
  for (int i = 0; i < N; i += blockDim.x) {
    
    // 4. Collaborative Loading (The Teamwork)
    // ---------------------------------------
    // Each thread loads ONE element from Global Memory into Shared Memory.
    // Even though the thread computes C[row][col], it might load a different 
    // element (relative to the tile) to s_a or s_b.
    
    // Flattened 2D index for Shared Memory: [y * width + x]
    int local_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // Load Tile A:
    // Global Row: 'row' (Fixed for this thread)
    // Global Col: 'i' (Tile offset) + 'threadIdx.x' (Local Col offset)
    // Access Pattern: Coalesced (Threads in a warp access adjacent addresses).
    s_a[local_idx] = a[row * N + i + threadIdx.x];

    // Load Tile B:
    // Global Row: 'i' (Tile offset) + 'threadIdx.y' (Local Row offset)
    // Global Col: 'col' (Fixed for this thread)
    // Access Pattern: Coalesced (Threads in a warp vary 'col', accessing adjacent addresses).
    s_b[local_idx] = b[(i + threadIdx.y) * N + col];

    // 5. Barrier Synchronization (Wait for Load)
    // ------------------------------------------
    // Essential! We must ensure the ENTIRE tile is loaded into s_a and s_b
    // before any thread begins computation. 
    // Without this, some threads might read garbage data.
    __syncthreads();

    // 6. Compute Phase (Dot Product on Shared Memory)
    // -----------------------------------------------
    // Perform the sub-dot product for the current tile.
    // We access s_a and s_b from fast Shared Memory, avoiding Global Memory latency.
    for (int j = 0; j < blockDim.x; j++) {
      // s_a row: threadIdx.y (Local Row)
      // s_b col: threadIdx.x (Local Col)
      // 'j' acts as the sliding k-index within the tile.
      tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    // 7. Barrier Synchronization (Wait for Compute)
    // ---------------------------------------------
    // Essential! We must ensure all threads are finished using the current tile
    // before we overwrite s_a and s_b with the next tile's data in the next iteration.
    __syncthreads();
  }

  // 8. Write Back to Global Memory
  // Store the final accumulated result.
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
