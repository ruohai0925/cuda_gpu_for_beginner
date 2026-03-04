// This program computes a sum reduction algorithm with warp divergence
// By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>

using std::accumulate;
using std::generate;
using std::cout;
using std::vector;

// Define the size of shared memory (and block size)
// We assume a fixed block size of 256 threads for this example.
#define SHMEM_SIZE 256

// -----------------------------------------------------------------------------
// KERNEL: sumReduction
// -----------------------------------------------------------------------------
// This kernel performs a parallel reduction within a single thread block.
// CRITICAL FLAW: This specific implementation suffers from high WARP DIVERGENCE
// due to the use of the modulo operator (%) for thread selection.
// -----------------------------------------------------------------------------
__global__ void sumReduction(int *v, int *v_r) {
    // [Concept: Shared Memory]
    // We allocate __shared__ memory so threads in this block can exchange data quickly.
    // Accessing shared memory is much faster than Global Memory (DRAM).
    // Scope: Visible to all threads in THIS block.
    __shared__ int partial_sum[SHMEM_SIZE];

    // Calculate global thread ID to locate the data in the main input array (Global Memory)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // [Step 1: Load from Global to Shared Memory]
    // Each thread loads ONE element from the global input vector 'v'
    // into the fast shared memory array 'partial_sum'.
    partial_sum[threadIdx.x] = v[tid];
    
    // [Concept: Synchronization]
    // Barrier: We MUST wait here to ensure ALL threads have finished loading
    // their value into 'partial_sum' before we start reading/modifying it.
    __syncthreads();

    // [Step 2: Reduction Loop - The Source of Warp Divergence]
    // The stride 's' starts at 1 and doubles every iteration: 1, 2, 4, 8...
    for (int s = 1; s < blockDim.x; s *= 2) {
        
        // [Concept: Warp Divergence]
        // The condition `threadIdx.x % (2 * s) == 0` causes divergence.
        // Iteration 1 (s=1): "Even" threads (0, 2, 4...) are ACTIVE. "Odd" threads are IDLE.
        // Iteration 2 (s=2): Threads 0, 4, 8... are ACTIVE. Others are IDLE.
        //
        // Why is this bad?
        // Threads in a Warp (group of 32) execute in lock-step. If threads 0-15 want to work
        // and threads 16-31 want to do nothing, the hardware must disable the idle threads
        // but still execute the instruction cycles. We are wasting hardware cycles on idle threads.
        if (threadIdx.x % (2 * s) == 0) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        
        // Synchronization is required after every addition step to ensure
        // all threads have completed the current partial sum before the next step begins.
        __syncthreads();
    }

    // [Step 3: Write Result to Global Memory]
    // After the loop, the total sum for this entire BLOCK is stored in partial_sum[0].
    // Only Thread 0 needs to write this single value back to Global Memory.
    if (threadIdx.x == 0) {
        // We write to 'v_r' at index 'blockIdx.x'.
        // This effectively creates a smaller array of partial sums (one per block).
        v_r[blockIdx.x] = partial_sum[0];
    }
}

int main() {
	// 1. Setup Data Size
    // ------------------
    int N = 1 << 16; // 65,536 elements
    size_t bytes_input = N * sizeof(int);

    // 2. Calculate Grid Dimensions FIRST
    // ----------------------------------
    // We need to know the Grid Size to allocate the partial sum array correctly.
    const int TB_SIZE = 256;
    int GRID_SIZE = N / TB_SIZE; // 65536 / 256 = 256 Blocks

    // 3. Optimized Memory Allocation
    // ------------------------------
    // d_v (Input): Needs full size (N)
    // d_v_r (Partial Sums): Only needs size equal to the number of blocks (GRID_SIZE)
    size_t bytes_partial = GRID_SIZE * sizeof(int);

    // Host Data
    vector<int> h_v(N);
    vector<int> h_v_r(GRID_SIZE); // Optimized: Only size 256, not 65536

    generate(begin(h_v), end(h_v), [](){ return rand() % 10; });

    // Device Data
    int *d_v, *d_v_r;
    cudaMalloc(&d_v, bytes_input);
    cudaMalloc(&d_v_r, bytes_partial); // Optimized Allocation

    // Copy Input
    cudaMemcpy(d_v, h_v.data(), bytes_input, cudaMemcpyHostToDevice);

    // 4. Kernel Launches (The Implicit Barrier)
    // -----------------------------------------
    
    // LAUNCH 1: Reduce 65,536 elements -> 256 partial sums
    // Input: d_v (Size N)
    // Output: d_v_r (Size GRID_SIZE)
    // Each of the 256 blocks writes one integer to d_v_r.
    sumReduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);

    // LAUNCH 2: Reduce 256 partial sums -> 1 Final Sum
    // We now treat d_v_r as the input.
    // We launch 1 Block with 256 threads.
    // Input: d_v_r (Size 256)
    // Output: d_v_r (Index 0 will hold the total sum)
    sumReduction<<<1, GRID_SIZE>>> (d_v_r, d_v_r);

    // 5. Copy Result
    // --------------
    // We only strictly need the first element, but copying the small partial array is fine.
    cudaMemcpy(h_v_r.data(), d_v_r, bytes_partial, cudaMemcpyDeviceToHost);

    // 6. Verify
    // ---------
    // The result is at index 0 of the partial sum array
    assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));

    cout << "COMPLETED SUCCESSFULLY\n";
    cout << "Memory Saved: " << (bytes_input - bytes_partial) << " bytes on d_v_r.\n";

    cudaFree(d_v);
    cudaFree(d_v_r);

    return 0;
}