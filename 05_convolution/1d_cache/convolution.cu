// This program implements a 1D convolution using CUDA.
// It stores the mask in Constant Memory. 
// It loads ONLY the primary array into Shared Memory, deliberately ignoring the halo elements.
// Halo elements are handled as a fallback to the Hardware L1 Cache.
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
// CUDA KERNEL: 1-D Convolution (Shared Memory + L1 Cache Fallback)
// 
// KNOWLEDGE HINT 1: The Philosophy of Simplicity
// In Version 3, we wrote complex, error-prone code to force 256 threads to load 
// 262 elements (handling the Right Halo manually). 
// Here, we abandon that complexity. Every thread loads exactly ONE element. 
// If a thread needs a Right Halo element, it simply reads it directly from Global 
// Memory, trusting the GPU's hardware L1 Cache to make it fast.
// =================================================================================
__global__ void convolution_1d(int *array, int *result, int n) {
  // Global thread ID calculation
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Store all elements needed to compute output in shared memory
  extern __shared__ int s_array[];

  // =================================================================================
  // KNOWLEDGE HINT 2: The 1-to-1 Load (Zero Complexity)
  // No offsets. No two-stage loading. No bounds checking.
  // Thread 0 loads element 0. Thread 255 loads element 255. 
  // Note: Because the CPU padded the array, array[tid] for thread 0 safely 
  // grabs the left-side padding zero automatically.
  // =================================================================================
  s_array[threadIdx.x] = array[tid];

  // We still must sync! Even though we didn't load halos, threads still share 
  // the main chunk of data loaded by their neighbors.
  __syncthreads();

  // Temp value for calculation (in Registers)
  int temp = 0;

  // Go over each element of the mask
  for (int j = 0; j < MASK_LENGTH; j++) {
      
    // =================================================================================
    // KNOWLEDGE HINT 3: The Hybrid Divergent Read (SRAM vs. L1 Cache)
    // Here we explicitly check: "Am I trying to read the Right Halo?"
    // If (threadIdx.x + j) >= 256, it means we have stepped out of the bounds of 
    // the data we loaded into our Shared Memory (s_array).
    // =================================================================================
    
    if ((threadIdx.x + j) >= blockDim.x) {
        
      // FALLBACK TO GLOBAL MEMORY: 
      // We read directly from 'array'. The first time a thread does this, it might 
      // be a slow DRAM read, but it immediately populates the hardware L1 Cache. 
      // Subsequent threads in the warp asking for this element will hit the L1 Cache!
      temp += array[tid + j] * mask[j];
      
    } else {
        
      // FAST PATH: 
      // The data is safely inside our software-managed Shared Memory.
      temp += s_array[threadIdx.x + j] * mask[j];
      
    }
    // =================================================================================
    // KNOWLEDGE HINT 4: The Cost of Divergence
    // Yes, this if/else introduces Warp Divergence! Some threads go to Global Memory, 
    // others go to Shared Memory. 
    // However, because MASK_LENGTH is 7, ONLY the threads at the very end of the block 
    // (threads 250 to 255) will trigger the 'if'. This means ONLY THE VERY LAST WARP 
    // out of the 8 warps in our block suffers from divergence. The penalty is negligible!
    // =================================================================================
  }

  // Write-back the results
  result[tid] = temp;
}

// =================================================================================
// CPU VERIFICATION FUNCTION (Unchanged)
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
  // Number of elements in result array
  int n = 1 << 20;
  int bytes_n = n * sizeof(int);
  size_t bytes_m = MASK_LENGTH * sizeof(int);

  // Radius for padding the array
  int r = MASK_LENGTH / 2;
  int n_p = n + r * 2;

  // Size of the padded array in bytes
  size_t bytes_p = n_p * sizeof(int);

  // Allocate the array (include edge elements)...
  int *h_array = new int[n_p];

  // ... and initialize it (Zero-Padding is still applied!)
  for (int i = 0; i < n_p; i++) {
    if ((i < r) || (i >= (n + r))) {
      h_array[i] = 0;
    } else {
      h_array[i] = rand() % 100;
    }
  }

  // Allocate the mask and initialize it
  int *h_mask = new int[MASK_LENGTH];
  for (int i = 0; i < MASK_LENGTH; i++) {
    h_mask[i] = rand() % 10;
  }

  // Allocate space for the result
  int *h_result = new int[n];

  // Allocate space on the device
  int *d_array, *d_result;
  cudaMalloc(&d_array, bytes_p);
  cudaMalloc(&d_result, bytes_n);

  // Copy the data to the device
  cudaMemcpy(d_array, h_array, bytes_p, cudaMemcpyHostToDevice);

  // Copy the mask directly to the symbol
  cudaMemcpyToSymbol(mask, h_mask, bytes_m);

  // Threads per TB
  int THREADS = 256;
  int GRID = (n + THREADS - 1) / THREADS;

  // =================================================================================
  // KNOWLEDGE HINT 5: Simplified Shared Memory Allocation
  // We no longer add "+ r * 2" to our size. 
  // We only allocate exactly enough memory for 256 threads to store 256 elements.
  // =================================================================================
  size_t SHMEM = THREADS * sizeof(int);

  // Call the kernel
  convolution_1d<<<GRID, THREADS, SHMEM>>>(d_array, d_result, n);

  // Copy back the result
  cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

  // Verify the result
  verify_result(h_array, h_mask, h_result, n);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  // Free allocated memory on the device and host
  delete[] h_array;
  delete[] h_result;
  delete[] h_mask;
  cudaFree(d_array);
  cudaFree(d_result);

  return 0;
}