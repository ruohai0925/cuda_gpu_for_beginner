// This program implements a naive 1D convolution using CUDA
// By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

// =================================================================================
// CUDA KERNEL: 1-D Convolution (Naive Version)
// 
// KNOWLEDGE HINT - Parallelism Strategy:
// We map EXACTLY ONE thread to ONE output element in the result array.
// There is no dependency between calculating result[0] and result[1].
// =================================================================================
__global__ void convolution_1d(int *array, int *mask, int *result, int n, int m) {
  
  // 1. GLOBAL THREAD ID CALCULATION
  // Calculate the unique global thread index (tid). 
  // This tid directly dictates which element of the output 'result' array this thread will compute.
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // 2. RADIUS CALCULATION
  // The mask is centered around the element we are currently computing.
  // Integer division truncates, so if m=7, r = 7/2 = 3. 
  // This means 3 elements to the left, the center element itself, and 3 elements to the right.
  int r = m / 2;

  // 3. STARTING POINT CALCULATION
  // To apply the mask, we need to know where the left-most element of the mask 
  // aligns with our input array. 
  // Example: If we are thread 0 (tid=0) and radius is 3, start = 0 - 3 = -3.
  // This means our mask "hangs off" the left edge of the input array.
  int start = tid - r;

  // Temp value to accumulate the sum of products
  int temp = 0;

  // 4. THE CONVOLUTION LOOP (SLIDING THE MASK)
  // We iterate through every element of the small mask array (size m).
  for (int j = 0; j < m; j++) {
    
    // 5. BOUNDARY CHECKING (GHOST CELLS / HALO ELEMENTS)
    // If start + j is negative, it means the mask is hanging off the left edge.
    // If start + j >= n, it means the mask is hanging off the right edge.
    // We treat out-of-bounds elements as zeros. Since 0 * mask[j] = 0, 
    // we simply bypass the computation for out-of-bounds indices to prevent SegFaults.
    if (((start + j) >= 0) && ((start + j) < n)) {
      // Accumulate partial results: Input_Element * Mask_Element
      temp += array[start + j] * mask[j];
    }
  }

  // 6. WRITE-BACK
  // Store the final accumulated result into the output array.
  result[tid] = temp;
}

// =================================================================================
// CPU VERIFICATION FUNCTION
// This serves as our ground truth. Notice how the logic is identical to the GPU
// kernel, but wrapped inside a sequential `for (int i = 0; i < n; i++)` loop instead
// of utilizing parallel threads.
// =================================================================================
void verify_result(int *array, int *mask, int *result, int n, int m) {
  int radius = m / 2;
  int temp;
  int start;
  for (int i = 0; i < n; i++) {
    start = i - radius;
    temp = 0;
    for (int j = 0; j < m; j++) {
      if ((start + j >= 0) && (start + j < n)) {
        temp += array[start + j] * mask[j];
      }
    }
    assert(temp == result[i]); // If GPU is wrong, program crashes here
  }
}

int main() {
  // Number of elements in result array: 2^20 (approx 1 million elements)
  int n = 1 << 20;
  int bytes_n = n * sizeof(int);

  // Number of elements in the convolution mask: 7
  int m = 7;
  int bytes_m = m * sizeof(int);

  // Allocate Host (CPU) memory using std::vector
  std::vector<int> h_array(n);
  // Initialize with random numbers 0-99
  std::generate(begin(h_array), end(h_array), [](){ return rand() % 100; });

  // Allocate and initialize the Mask
  std::vector<int> h_mask(m);
  std::generate(begin(h_mask), end(h_mask), [](){ return rand() % 10; });

  // Allocate space for the CPU result
  std::vector<int> h_result(n);

  // Allocate Device (GPU) memory
  int *d_array, *d_mask, *d_result;
  cudaMalloc(&d_array, bytes_n);
  cudaMalloc(&d_mask, bytes_m);
  cudaMalloc(&d_result, bytes_n);

  // KNOWLEDGE HINT - The I/O Bottleneck:
  // From our profiling experience, sending these 4MB of data across the PCIe bus 
  // will take significantly more time than the actual kernel execution.
  cudaMemcpy(d_array, h_array.data(), bytes_n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, h_mask.data(), bytes_m, cudaMemcpyHostToDevice);

  // Execution Configuration
  int THREADS = 256; // 256 threads per block
  int GRID = (n + THREADS - 1) / THREADS; // Ceiling division to cover all 'n' elements

  // Launch the Kernel
  convolution_1d<<<GRID, THREADS>>>(d_array, d_mask, d_result, n, m);

  // Wait for GPU to finish and copy the result back to Host
  cudaMemcpy(h_result.data(), d_result, bytes_n, cudaMemcpyDeviceToHost);

  // Verify GPU results against CPU serial execution
  verify_result(h_array.data(), h_mask.data(), h_result.data(), n, m);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  // Clean up Device memory
  cudaFree(d_result);
  cudaFree(d_mask);
  cudaFree(d_array);

  return 0;
}