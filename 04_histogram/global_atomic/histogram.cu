// This program shows off a naive Global Memory implementation of a histogram
// kernel in CUDA. It relies heavily on Global Atomics.
// By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

using std::accumulate;
using std::cout;
using std::generate;
using std::ios;
using std::ofstream;
using std::vector;

// =================================================================================
// KNOWLEDGE HINT 1: Bin Calculation
// We are sorting 26 lowercase English letters into 7 bins.
// DIV calculates the ceiling division: ceil(26 / 7) = 4.
// This means the first 6 bins will hold 4 letters each, and the last bin holds 2.
// =================================================================================
constexpr int BINS = 7;
constexpr int DIV = ((26 + BINS - 1) / BINS); // ceil(26 / 7) = 4

// =================================================================================
// GPU KERNEL: Global Memory Histogram
// =================================================================================
__global__ void histogram(char *a, int *result, int N) {
  // Calculate global thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int alpha_position;
  
  // =================================================================================
  // KNOWLEDGE HINT 2: The Grid-Stride Loop (Thread Coarsening)
  // Instead of assuming `tid < N` and exiting, we loop!
  // The stride is `gridDim.x * blockDim.x` (the total number of threads in the grid).
  // If we launch fewer threads than elements in the array, the threads will automatically 
  // wrap around and process the next chunk of data. This allows one thread to process 
  // multiple elements, enabling "Thread Coarsening" without changing the kernel code!
  // =================================================================================
  for (int i = tid; i < N; i += (gridDim.x * blockDim.x)) {
    
    // Calculate the position in the alphabet (0 to 25)
    // 'a' - 'a' = 0, 'b' - 'a' = 1, etc.
    alpha_position = a[i] - 'a';
    
    // =================================================================================
    // KNOWLEDGE HINT 3: Global Atomic Operations (The Bottleneck)
    // [Image of memory data race condition]
    // If we just wrote `result[...] += 1;`, thousands of threads would read the 
    // same bin value simultaneously, add 1, and overwrite each other (Data Race).
    // atomicAdd() forces the hardware to serialize these operations.
    // However, because 'result' is in Global Memory, millions of threads are now 
    // queuing up at the L2 Cache to update just 7 memory addresses. 
    // This creates massive lock contention and destroys parallelism.
    // =================================================================================
    atomicAdd(&result[alpha_position / DIV], 1); // ID 0 - 6, 6th bin is the last bin
  }
}

int main() {
  // Declare our problem size: 2^24 = ~16.7 million elements
  int N = 1 << 24;

  // Allocate memory on the host
  vector<char> h_input(N);

  // Allocate space for the binned result (initialized to 0 automatically)
  vector<int> h_result(BINS);

  // Initialize the array with random lowercase letters
  srand(1);
  generate(begin(h_input), end(h_input), []() { return 'a' + (rand() % 26); });

  // Allocate memory on the device
  char *d_input;
  int *d_result;
  cudaMalloc(&d_input, N);
  cudaMalloc(&d_result, BINS * sizeof(int));

  // Copy the array to the device
  cudaMemcpy(d_input, h_input.data(), N, cudaMemcpyHostToDevice);
  
  // Initialize device result array to 0 by copying the empty host vector
  cudaMemcpy(d_result, h_result.data(), BINS * sizeof(int),
             cudaMemcpyHostToDevice);

  // Number of threads per threadblock
  int THREADS = 512;

  // Calculate the number of threadblocks
  // Note: Here, BLOCKS * THREADS exactly equals N. 
  // So the Grid-Stride loop in the kernel will only execute exactly 1 time per thread.
  // To test Thread Coarsening, you could change this to: int BLOCKS = (N / THREADS) / 4;
  int BLOCKS = N / THREADS;

  // Launch the kernel
  histogram<<<BLOCKS, THREADS>>>(d_input, d_result, N);

  // Copy the result back
  cudaMemcpy(h_result.data(), d_result, BINS * sizeof(int),
             cudaMemcpyDeviceToHost);

  // Functional test: 
  // The sum of all values in our 7 bins MUST perfectly equal the total number of elements N.
  assert(N == accumulate(begin(h_result), end(h_result), 0));

  // Write the data out for gnuplot
  ofstream output_file;
  output_file.open("histogram.dat", ios::out | ios::trunc);
  for (auto i : h_result) {
    output_file << i << " \n\n";
  }
  output_file.close();

  // Free memory
  cudaFree(d_input);
  cudaFree(d_result);

  return 0;
}