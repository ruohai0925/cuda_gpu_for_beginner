// CUDA Sum Reduction - Step 3: No Bank Conflicts (Optimized)
// Original By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <assert.h>

using std::accumulate;
using std::generate;
using std::cout;
using std::vector;

#define SHMEM_SIZE 256

// -----------------------------------------------------------------------------
// KERNEL: sumReductionNoConflicts
// -----------------------------------------------------------------------------
// Improvement over V2:
// V2 used 'index = tid * 2' -> caused 2-Way Bank Conflicts (Strided Access).
// V3 uses 'index = tid' with a shrinking stride -> Linear Access (No Conflicts).
// -----------------------------------------------------------------------------
__global__ void sumReductionNoConflicts(int *v, int *v_r) {
    // 1. Shared Memory Allocation
    // ---------------------------
    __shared__ int partial_sum[SHMEM_SIZE];

    // 2. Load Data (Global -> Shared)
    // ---------------------------
    // Calculate global index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread loads one element from Global Memory to Shared Memory.
    // This is coalesced access (good).
    partial_sum[threadIdx.x] = v[tid];

    // Barrier: Wait for all threads to finish loading.
    __syncthreads();

    // 3. The Optimized Reduction Loop (The "Fold-In" Strategy)
    // --------------------------------------------------------
    // Instead of starting with stride s=1 and doubling it (1, 2, 4...),
    // we start with a large stride (half the block) and shrink it.
    //
    // Initial State (Block=256): s = 128
    // Iteration 1: s = 128. Threads 0-127 are active.
    //              T0 adds partial_sum[0] + partial_sum[128]
    //              T1 adds partial_sum[1] + partial_sum[129]
    //
    // BANK CONFLICT ANALYSIS:
    // Look at the first operand: partial_sum[threadIdx.x]
    // Thread 0 accesses Index 0 -> Bank 0
    // Thread 1 accesses Index 1 -> Bank 1
    // ...
    // Thread 31 accesses Index 31 -> Bank 31
    //
    // Result: LINEAR MAPPING.
    // Every thread in the warp maps to a unique bank.
    // ZERO BANK CONFLICTS. Maximum Bandwidth.
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        // Barrier is mandatory after each fold step
        __syncthreads();
    }

    // 4. Write Result
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

int main() {
    // 1. Setup Data Size
    int N = 1 << 16; // 65,536 elements
    size_t bytes_input = N * sizeof(int);

    // 2. Grid & Block Configuration
    const int TB_SIZE = 256;
    int GRID_SIZE = N / TB_SIZE; // 256 Blocks

    // 3. Optimized Memory Allocation
    // Output vector only needs to hold 1 partial sum per block (256 total)
    size_t bytes_partial = GRID_SIZE * sizeof(int);

    // Host Data
    vector<int> h_v(N);
    vector<int> h_v_r(GRID_SIZE);

    // Initialize with 1s for easy verification (Sum should be 65536)
    // Using std::generate instead of raw loop
    generate(begin(h_v), end(h_v), [](){ return rand() % 10; });

    // Device Data
    int *d_v, *d_v_r;
    cudaMalloc(&d_v, bytes_input);
    cudaMalloc(&d_v_r, bytes_partial); // Optimized size

    // Copy Input
    cudaMemcpy(d_v, h_v.data(), bytes_input, cudaMemcpyHostToDevice);

    // 4. Kernel Launches (Implicit Barrier via Stream 0)
    // --------------------------------------------------
    
    // LAUNCH 1: 65,536 -> 256
    // Threads load data, fold it, and write 256 partial sums to d_v_r.
    sumReductionNoConflicts<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);

    // LAUNCH 2: 256 -> 1
    // We reuse d_v_r as input. One block sums up the previous results.
    sumReductionNoConflicts<<<1, TB_SIZE>>>(d_v_r, d_v_r);

    // 5. Copy Result Back
    cudaMemcpy(h_v_r.data(), d_v_r, bytes_partial, cudaMemcpyDeviceToHost);

    // 6. Verify
    // Sum of 65536 ones should be 65536
    // Using std::accumulate for robust verification
    assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));
    cout << "COMPLETED SUCCESSFULLY\n";

    cudaFree(d_v);
    cudaFree(d_v_r);

    return 0;
}