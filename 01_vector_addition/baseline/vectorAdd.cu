// This program computes the sum of two vectors of length N
// By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU

// Pipeline:
// 1. Read the codes and Explain GPU and C++ knowledge;
// 2. Run the codes and check the results;

// nvcc vectorAdd.cu -o vectorAdd -arch=sm_86
// nsys profile --stats=true .\vectorAdd.exe

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int N) {
  // Calculate global thread ID
  // CUDA provides built-in variables to identify threads in the execution grid:
  //
  // blockIdx.x: The index of the current CTA (block) within the grid in the x-dimension.
  //             Ranges from 0 to (NUM_BLOCKS - 1).
  //             Each CTA has a unique blockIdx.x value.
  //
  // blockDim.x: The dimension (number of threads) of a CTA in the x-dimension.
  //             This is the same as NUM_THREADS passed to the kernel launch.
  //             In this code: blockDim.x = 1024 (NUM_THREADS).
  //
  // threadIdx.x: The index of the current thread within its CTA in the x-dimension.
  //              Ranges from 0 to (blockDim.x - 1), i.e., 0 to 1023.
  //              Each thread within a CTA has a unique threadIdx.x value.
  //
  // Global thread ID formula: tid = (blockIdx.x * blockDim.x) + threadIdx.x
  // This calculates a unique ID for each thread across the entire grid.
  // Example: If blockIdx.x = 2, blockDim.x = 1024, threadIdx.x = 5,
  //          then tid = (2 * 1024) + 5 = 2053
  // This ensures each thread processes a different array element.
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Boundary check
  if (tid < N) c[tid] = a[tid] + b[tid];
}

// Check vector add result
void verify_result(std::vector<int> &a, std::vector<int> &b,
                   std::vector<int> &c) {
  for (int i = 0; i < a.size(); i++) {
    assert(c[i] == a[i] + b[i]);
  }
}

int main() {
  // Array size of 2^16 (65536 elements)
  // constexpr: Compile-time constant - the value is known and evaluated at compile time.
  //            This allows the compiler to optimize and use N in contexts requiring
  //            compile-time constants (like template parameters or array sizes).
  //            The value cannot change during program execution.
  constexpr int N = 1 << 26;  // 1 << 16 = 2^16 = 65536
  // size_t: Unsigned integer type used to represent sizes and counts in bytes.
  //         It's guaranteed to be large enough to hold the size of any object in memory.
  //         On 64-bit systems, it's typically 64 bits (unsigned long), on 32-bit systems, 32 bits.
  //         Using size_t for byte counts is idiomatic in C++ and prevents overflow issues.
  //         sizeof(int) returns size_t, so using size_t here is type-safe.
  constexpr size_t bytes = sizeof(int) * N;  // Total bytes needed: 4 bytes * 65536 = 262144 bytes

  // Vectors for holding the host-side (CPU-side) data
  std::vector<int> a;
  // reserve(N): Pre-allocates memory for N elements without actually adding elements.
  //             This avoids multiple reallocations as we push_back elements later.
  //             Without reserve(), the vector might need to reallocate and copy data
  //             multiple times as it grows, which is inefficient.
  //             Note: reserve() only allocates memory; size() is still 0 until we push_back.
  a.reserve(N);
  std::vector<int> b;
  b.reserve(N);
  //std::vector<int> c;
  //c.reserve(N);
  std::vector<int> c(N);
  //or
  //std::vector<int> c; 
  //c.resize(N)


  // Initialize random numbers in each array
  // rand(): C standard library function that generates pseudo-random integers.
  //         Returns a value between 0 and RAND_MAX (typically 32767).
  //         rand() % 100: Uses modulo operator to get a value between 0 and 99.
  //                       This gives us random integers in the range [0, 99].
  //         Note: For better randomness, consider using <random> header in modern C++.
  for (int i = 0; i < N; i++) {
    a.push_back(rand() % 100);  // Add random number 0-99 to vector a
    b.push_back(rand() % 100);  // Add random number 0-99 to vector b
  }

  // Allocate memory on the device
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data from the host to the device (CPU -> GPU)
  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

  // Threads per CTA (1024)
  // CTA: Cooperative Thread Array - CUDA terminology for a "thread block".
  //      A CTA is a group of threads that can cooperate and share resources
  //      (like shared memory). CTAs are organized into a grid.
  //      In CUDA syntax: <<<NUM_BLOCKS, NUM_THREADS>>>, NUM_BLOCKS is the number of CTAs,
  //      and NUM_THREADS is the number of threads per CTA.
  int NUM_THREADS = 1 << 10;  // 1 << 10 = 2^10 = 1024 threads per CTA

  // CTAs per Grid (also called "blocks per grid")
  // Grid: The entire collection of CTAs launched for a kernel.
  //       A grid contains multiple CTAs, and each CTA contains multiple threads.
  // We need to launch at LEAST as many threads as we have elements
  // This equation pads an extra CTA to the grid if N cannot evenly be divided
  // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
  // Formula: (N + NUM_THREADS - 1) / NUM_THREADS ensures we have enough threads
  //          by rounding up the division (ceiling division).
  // similar to: ceil((float)N / NUM_THREADS)
  // ceil((float)N / NUM_THREADS): Rounds up the division to the nearest integer.
  //                        This ensures we have enough threads to cover all elements.
  //                        For example, if N = 1025 and NUM_THREADS = 1024,
  //                        ceil((float)1025 / 1024) = 2, so we need 2 CTAs.
  //                        If N = 1024 and NUM_THREADS = 1024,
  //                        ceil((float)1024 / 1024) = 1, so we need 1 CTA.
  //                        If N = 1023 and NUM_THREADS = 1024,
  //                        ceil((float)1023 / 1024) = 1, so we need 1 CTA.
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
  std::cout << "NUM_BLOCKS: " << NUM_BLOCKS << std::endl;
  std::cout << "NUM_THREADS: " << NUM_THREADS << std::endl;

  // Launch the kernel on the GPU
  // Kernel calls are asynchronous (the CPU program continues execution after
  // call, but no necessarily before the kernel finishes)
  vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

  for(int i=0; i<100; i++) {
      vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
  }

  // Copy sum vector from device to host
  // cudaMemcpy is a synchronous operation, and waits for the prior kernel
  // launch to complete (both go to the default stream in this case).
  // Therefore, this cudaMemcpy acts as both a memcpy and synchronization
  // barrier.
  cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  // Check result for errors
  verify_result(a, b, c);

  // Free memory on device (GPU)
  // Note: We explicitly free GPU memory using cudaFree() because CUDA device memory
  //       is managed separately from CPU memory and requires manual deallocation.
  //       Unlike CPU memory, GPU memory is not automatically freed when variables
  //       go out of scope.
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // Note: CPU vectors (std::vector<int> a, b, c) do NOT need explicit freeing.
  //       std::vector is a C++ RAII (Resource Acquisition Is Initialization) container
  //       that automatically manages its memory. When the vectors go out of scope
  //       (at the end of main()), their destructors are automatically called,
  //       which deallocates the memory they hold. This is automatic memory management
  //       provided by C++ - no manual free() or delete is needed.

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
