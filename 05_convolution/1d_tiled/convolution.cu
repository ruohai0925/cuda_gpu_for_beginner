// This program implements a 1D convolution using CUDA,
// and significantly optimizes performance by storing the mask in Constant Memory,
// and loading the input array into Shared Memory (On-Chip SRAM).
// By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU

#include <cassert>
#include <cstdlib>
#include <iostream>

// Length of our convolution mask
#define MASK_LENGTH 7

// Allocate space for the mask in constant memory (Read-Only, Broadcasts to Warps)
__constant__ int mask[MASK_LENGTH];

// =================================================================================
// CUDA KERNEL: 1-D Convolution (Shared Memory + Constant Memory Tiled Version)
// =================================================================================
__global__ void convolution_1d(int *array, int *result, int n) {
  // 1. GLOBAL THREAD ID CALCULATION
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // =================================================================================
  // KNOWLEDGE HINT 1: Dynamic Shared Memory Allocation
  // 'extern __shared__' tells the compiler: "Reserve a block of On-Chip SRAM for this 
  // Thread Block. I won't tell you the size now; the CPU will tell you when it launches 
  // the kernel." This array is visible to ALL threads within this specific block.
  // =================================================================================
  extern __shared__ int s_array[];

  // r: The number of padded elements on ONE side (Radius = 3)
  int r = MASK_LENGTH / 2;

  // d: The total number of padded elements (Diameter/Total Halo = 6)
  int d = 2 * r;

  // Size of the padded shared memory array (e.g., 256 + 6 = 262 elements)
  int n_padded = blockDim.x + d;

  // Offset used by the threads designated to load the "Right Halo"
  int offset = threadIdx.x + blockDim.x;

  // Global offset to figure out exactly where in DRAM those Right Halo elements live
  int g_offset = blockDim.x * blockIdx.x + offset;

  // =================================================================================
  // KNOWLEDGE HINT 2: The Two-Stage Collaborative Loading Strategy
  // Goal: Fill 262 Shared Memory slots using only 256 Threads.
  // 
  // STAGE 1: The Main Chunk (100% Coalesced, 0% Divergence)
  // All 256 threads grab exactly one element from DRAM and put it into SRAM.
  // Note: Because the CPU padded the array with zeros at the front, s_array[0] 
  // naturally grabs the left-side padding!
  // =================================================================================
  
  s_array[threadIdx.x] = array[tid];

  // =================================================================================
  // STAGE 2: The Right Halo (Minimal Divergence)
  // We still need to load the last 6 elements (the right halo). 
  // We make the first 6 threads of the block (threadIdx.x 0 to 5) do this extra work.
  // 'offset' for thread 0 is 256. Since 256 < 262 (n_padded), it executes the load.
  // Threads 6 through 255 will fail this 'if' check and wait.
  // =================================================================================
  if (offset < n_padded) {
    s_array[offset] = array[g_offset];
  }

  // =================================================================================
  // KNOWLEDGE HINT 3: The Wall (Barrier Synchronization)
  // ABSOLUTELY CRITICAL. We must force fast threads to wait for the slower threads 
  // (especially those 6 threads that had to do double the loading work). 
  // If we don't sync here, some threads will start calculating using empty/garbage 
  // values in s_array.
  // =================================================================================
  __syncthreads();

  // Temp value for calculation (in Registers)
  int temp = 0;

  // =================================================================================
  // KNOWLEDGE HINT 4: The Tiled Computation (Zero-Padding Magic)
  // Look closely: THERE ARE NO 'IF' STATEMENTS HERE! 
  // Because the CPU physically added zeros to the ends of the input array, we completely 
  // eliminated the Boundary Checks. This completely destroys the Branch Divergence 
  // that slowed down our earlier versions.
  // Furthermore, ALL data fetched in this loop comes from On-Chip SRAM!
  // s_array -> Shared Memory
  // mask    -> Constant Cache
  // =================================================================================
  for (int j = 0; j < MASK_LENGTH; j++) {
    temp += s_array[threadIdx.x + j] * mask[j];
  }

  // Write-back the results to Global Memory (DRAM)
  // Result array is NOT padded, so tid perfectly aligns with the output.
  result[tid] = temp;
}

// =================================================================================
// CPU VERIFICATION FUNCTION
// Notice how much simpler this became because of the padded arrays!
// =================================================================================
void verify_result(int *array, int *mask, int *result, int n) {
  int temp;
  for (int i = 0; i < n; i++) {
    temp = 0;
    for (int j = 0; j < MASK_LENGTH; j++) {
      temp += array[i + j] * mask[j];
    }
    assert(temp == result[i]);
  }
}

int main() {
  // Number of elements in result array (2^20)
  int n = 1 << 20;
  int bytes_n = n * sizeof(int);
  size_t bytes_m = MASK_LENGTH * sizeof(int);

  // Radius for padding the array
  int r = MASK_LENGTH / 2;
  
  // =================================================================================
  // KNOWLEDGE HINT 5: Physical Zero-Padding on the Host
  // The padded array size is n + 6.
  // =================================================================================
  int n_p = n + r * 2;
  size_t bytes_p = n_p * sizeof(int);

  // Allocate the padded array
  int *h_array = new int[n_p];

  // Initialize it: Inject zeros at the exact edges (the halos)
  for (int i = 0; i < n_p; i++) {
    if ((i < r) || (i >= (n + r))) {
      h_array[i] = 0; // The Ghost Cells
    } else {
      h_array[i] = rand() % 100;
    }
  }

  int *h_mask = new int[MASK_LENGTH];
  for (int i = 0; i < MASK_LENGTH; i++) {
    h_mask[i] = rand() % 10;
  }

  int *h_result = new int[n];

  int *d_array, *d_result;
  cudaMalloc(&d_array, bytes_p); // Note: Allocating padded size
  cudaMalloc(&d_result, bytes_n); // Note: Result is normal size

  // Copy padded array to device
  cudaMemcpy(d_array, h_array, bytes_p, cudaMemcpyHostToDevice);

  // Copy mask to Constant Memory symbol
  cudaMemcpyToSymbol(mask, h_mask, bytes_m);

  int THREADS = 256;
  int GRID = (n + THREADS - 1) / THREADS;

  // =================================================================================
  // KNOWLEDGE HINT 6: Launch Configuration for Dynamic Shared Memory
  // We pass a third parameter inside the <<< >>> brackets. 
  // This tells the GPU exactly how many bytes to allocate for 'extern __shared__ s_array'
  // Size = (256 threads + 6 halo elements) * 4 bytes = 1048 bytes per Block.
  // =================================================================================
  size_t SHMEM = (THREADS + r * 2) * sizeof(int);

  // Call the kernel
  convolution_1d<<<GRID, THREADS, SHMEM>>>(d_array, d_result, n);

  // Copy back the result
  cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

  // Verify the result
  verify_result(h_array, h_mask, h_result, n);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory
  delete[] h_array;
  delete[] h_result;
  delete[] h_mask;
  cudaFree(d_result);
  cudaFree(d_array);

  return 0;
}