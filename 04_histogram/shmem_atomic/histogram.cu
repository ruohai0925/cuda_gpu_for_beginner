// This program shows off a Shared Memory implementation of a histogram
// kernel in CUDA. It drastically reduces global memory contention using Privatization.
// By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU

#include <algorithm>
#include <cassert>
#include <cstdlib>
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

// Number of bins for our plot
constexpr int BINS = 7;
constexpr int DIV = ((26 + BINS - 1) / BINS);

// =================================================================================
// GPU KERNEL: Shared Memory Histogram (Privatization Strategy)
// =================================================================================
__global__ void histogram(char *a, int *result, int N) {
  // Calculate global thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // =================================================================================
  // KNOWLEDGE HINT 1: Privatization (The Local Village Ballot Box)
  // Instead of 1 million threads fighting over 7 bins in slow Global Memory,
  // we allocate 7 private bins inside the ultra-fast, on-chip Shared Memory.
  // Every Thread Block gets its own independent 's_result' array.
  // =================================================================================
  
  __shared__ int s_result[BINS];

  // =================================================================================
  // KNOWLEDGE HINT 2: Collaborative Initialization
  // Shared memory is uninitialized by default (contains garbage data).
  // We only need 7 threads to initialize the 7 bins to 0.
  // We use the first 7 threads of the block (threadIdx.x < 7) to do this quickly.
  // =================================================================================
  if (threadIdx.x < BINS) {
    s_result[threadIdx.x] = 0;
  }

  // =================================================================================
  // KNOWLEDGE HINT 3: The First Barrier
  // We MUST wait for those first 7 threads to finish writing the 0s.
  // If we don't sync here, thread 500 might start counting letters and adding to 
  // a bin before thread 0 has cleared that bin's garbage memory!
  // =================================================================================
  __syncthreads();

  // Calculate the bin positions locally
  int alpha_position;
  
  // =================================================================================
  // KNOWLEDGE HINT 4: High-Speed Local Atomics + Thread Coarsening
  // We still use the Grid-Stride loop. If we reduce the number of launched blocks 
  // in main(), each thread will loop multiple times (Thread Coarsening).
  // Because they are now writing to 's_result' (L1 Cache speed), the atomicAdd 
  // collisions are resolved incredibly fast with almost no performance penalty.
  // =================================================================================
  for (int i = tid; i < N; i += (gridDim.x * blockDim.x)) {
    // Calculate the position in the alphabet
    alpha_position = a[i] - 'a';
    
    // FAST ATOMIC ADD: Targets Shared Memory instead of Global Memory!
    atomicAdd(&s_result[(alpha_position / DIV)], 1);
  }

  // =================================================================================
  // KNOWLEDGE HINT 5: The Second Barrier
  // The counting phase is over. But before we report our results to the global memory,
  // we MUST ensure that every single thread in the block has finished counting its letters.
  // =================================================================================
  __syncthreads();

  // =================================================================================
  // KNOWLEDGE HINT 6: Global Reduction (The Village Chiefs)
  // Now, the local counting is perfectly summarized in 's_result'.
  // We appoint the first 7 threads (threadIdx.x < 7) to act as "Village Chiefs".
  // They take the 7 local subtotals and securely add them to the Global Memory bins.
  // We slashed global atomic collisions from N (millions) down to just 7 * BLOCKS!
  // =================================================================================
  if (threadIdx.x < BINS) {
    atomicAdd(&result[threadIdx.x], s_result[threadIdx.x]);
  }
}

int main() {
  // Declare our problem size (Scaled down to 2^16 for this specific run, 
  // but logic remains identical for 2^24)
  int N = 1 << 24;

  // Allocate memory on the host
  vector<char> h_input(N);
  vector<int> h_result(BINS);

  // Initialize the array
  srand(1);
  generate(begin(h_input), end(h_input), []() { return 'a' + (rand() % 26); });

  // Allocate memory on the device
  char *d_input;
  int *d_result;
  cudaMalloc(&d_input, N);
  cudaMalloc(&d_result, BINS * sizeof(int));

  // Copy the array to the device
  cudaMemcpy(d_input, h_input.data(), N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_result, h_result.data(), BINS * sizeof(int),
             cudaMemcpyHostToDevice);

  // Number of threads per threadblock
  int THREADS = 512;

  // Calculate the number of threadblocks
  // NOTE: If you divide this by 4 (e.g., BLOCKS = (N / THREADS) / 4), 
  // you will trigger Thread Coarsening, squeezing even more performance out 
  // of the Shared Memory privatization!
  int BLOCKS = N / THREADS / 4;

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