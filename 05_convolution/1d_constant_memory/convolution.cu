// This program implements a 1D convolution using CUDA,
// and significantly optimizes performance by storing the mask in Constant Memory.
// By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU

#include <cassert>
#include <cstdlib>
#include <iostream>

// =================================================================================
// KNOWLEDGE HINT 1: Static Sizing
// Constant memory requires the size to be known at compile time.
// =================================================================================
#define MASK_LENGTH 7

// =================================================================================
// KNOWLEDGE HINT 2: The __constant__ Qualifier
// This is the star of the show. By declaring this globally with __constant__, 
// we tell the GPU to store this array in the ultra-fast, read-only Constant Cache 
// (On-Chip SRAM). 
// 
// When a Warp (32 threads) requests mask[j], the hardware fetches it ONCE and 
// BROADCASTS it to all 32 threads simultaneously. This eliminates millions of 
// slow Off-Chip DRAM accesses!
// =================================================================================

__constant__ int mask[MASK_LENGTH];

// =================================================================================
// CUDA KERNEL: 1-D Convolution (Constant Memory Version)
// 
// KNOWLEDGE HINT 3: Kernel Signature Slim-down
// Notice what is missing! We no longer pass 'int *mask' or 'int m' as parameters.
// Because 'mask' is a global symbol in Constant Memory, the kernel can access it 
// directly without needing a pointer.
// =================================================================================
__global__ void convolution_1d(int *array, int *result, int n) {
  // 1. GLOBAL THREAD ID CALCULATION
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // 2. RADIUS CALCULATION (Now using the macro)
  int r = MASK_LENGTH / 2;

  // 3. STARTING POINT CALCULATION
  int start = tid - r;

  // Temp value for calculation (stored in ultra-fast On-Chip Registers)
  int temp = 0;

  // 4. THE CONVOLUTION LOOP
  for (int j = 0; j < MASK_LENGTH; j++) {
    // 5. BOUNDARY CHECKING
    if (((start + j) >= 0) && (start + j < n)) {
      // THE PERFORMANCE LEAP HAPPENS HERE:
      // array[start + j] -> Still a slow Off-Chip DRAM read (we'll fix this later).
      // mask[j]          -> Now a lightning-fast On-Chip SRAM read via Constant Cache!
      temp += array[start + j] * mask[j];
    }
  }

  // 6. WRITE-BACK (Off-Chip DRAM write)
  result[tid] = temp;
}

// =================================================================================
// CPU VERIFICATION FUNCTION
// =================================================================================
void verify_result(int *array, int *mask, int *result, int n) {
  int radius = MASK_LENGTH / 2;
  int temp;
  int start;
  for (int i = 0; i < n; i++) {
    start = i - radius;
    temp = 0;
    for (int j = 0; j < MASK_LENGTH; j++) {
      if ((start + j >= 0) && (start + j < n)) {
        temp += array[start + j] * mask[j];
      }
    }
    assert(temp == result[i]);
  }
}

int main() {
  // Number of elements in result array (2^20)
  int n = 1 << 20;

  // Size of the array in bytes
  int bytes_n = n * sizeof(int);

  // Size of the mask in bytes
  size_t bytes_m = MASK_LENGTH * sizeof(int);

  // Allocate Host arrays (using new instead of std::vector this time)
  int *h_array = new int[n];
  for (int i = 0; i < n; i++) {
    h_array[i] = rand() % 100;
  }

  int *h_mask = new int[MASK_LENGTH];
  for (int i = 0; i < MASK_LENGTH; i++) {
    h_mask[i] = rand() % 10;
  }

  int *h_result = new int[n];

  // Allocate space on the device (Global Memory / DRAM)
  int *d_array, *d_result;
  cudaMalloc(&d_array, bytes_n);
  cudaMalloc(&d_result, bytes_n);
  
  // NOTE: We do NOT cudaMalloc d_mask! The __constant__ declaration already 
  // reserved that physical space on the GPU for us.

  // Copy the main array to standard Global Memory
  cudaMemcpy(d_array, h_array, bytes_n, cudaMemcpyHostToDevice);

  // =================================================================================
  // KNOWLEDGE HINT 4: The Magic Copy API
  // Because 'mask' is a symbol, not a dynamic pointer, we must use a special API.
  // cudaMemcpyToSymbol directly targets the Constant Memory region associated with 
  // the variable name 'mask'. 
  // =================================================================================
  cudaMemcpyToSymbol(mask, h_mask, bytes_m);

  // Threads per TB
  int THREADS = 256;

  // Number of TBs
  int GRID = (n + THREADS - 1) / THREADS;

  // Call the kernel
  // Look how clean the arguments are without the mask pointers!
  convolution_1d<<<GRID, THREADS>>>(d_array, d_result, n);

  // Copy back the result
  cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

  // Verify the result
  verify_result(h_array, h_mask, h_result, n);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  // Free allocated memory
  delete[] h_array;
  delete[] h_result;
  delete[] h_mask;
  cudaFree(d_result);
  cudaFree(d_array);
  
  // NOTE: No cudaFree for the mask, because we never called cudaMalloc for it.

  return 0;
}