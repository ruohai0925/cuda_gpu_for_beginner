// CUDA Sum Reduction - Step 4: Reduce Idle Threads (Halve Grid, Double Load)
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
// KERNEL: sumReductionIdleThreads
// -----------------------------------------------------------------------------
// Key Change:
// 1. We process 2 elements per thread during the load phase.
// 2. We perform the first level of reduction BEFORE writing to Shared Memory.
// -----------------------------------------------------------------------------
__global__ void sumReductionIdleThreads(int *v, int *v_r) {
    // 1. Shared Memory Allocation
    __shared__ int partial_sum[SHMEM_SIZE];

    // 2. Global Index Calculation (The "Times 2" Trick)
    // -------------------------------------------------
    // Since each block now covers 2 * blockDim.x elements, the stride 
    // between blocks is blockDim.x * 2.
    // i (global)  = Points to the start of the data chunk for this thread
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 3. The "Double Load & Add" Strategy
    // -----------------------------------
    // Instead of: partial_sum[threadIdx.x] = v[i];
    // We do:      partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
    //
    // Example (Block 0, Thread 0, Dim 256):
    // i = 0.
    // Loads v[0] and v[0 + 256]. Adds them. Stores in partial_sum[0].
    //
    // Advantage:
    // - We just performed 512 -> 256 reduction without using Shared Memory yet.
    // - All 256 threads are working. No one is idle.
    partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];

    __syncthreads();

    // 4. Standard Folding Reduction (Same as V3)
    // ------------------------------------------
    // Now partial_sum contains 256 elements (which represent 512 global elements).
    // We proceed to reduce 256 -> 1.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // 5. Write Result
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

int main() {
    // 1. Setup Data
    int N = 1 << 16; // 65,536
    size_t bytes_input = N * sizeof(int);

    // 2. Configuration (CRITICAL CHANGE)
    const int TB_SIZE = 256;
    
    // GRID_SIZE IS HALVED!
    // V3: 65536 / 256 = 256 Blocks
    // V4: 65536 / (256 * 2) = 128 Blocks
    // Because each block now consumes 512 elements.
    int GRID_SIZE = N / (TB_SIZE * 2); 

    size_t bytes_partial = GRID_SIZE * sizeof(int); // Only 128 ints needed now

    // Host Data
    vector<int> h_v(N);
    vector<int> h_v_r(GRID_SIZE);
    generate(begin(h_v), end(h_v), [](){ return 1; });

    // Device Data
    int *d_v, *d_v_r;
    cudaMalloc(&d_v, bytes_input);
    cudaMalloc(&d_v_r, bytes_partial);

    cudaMemcpy(d_v, h_v.data(), bytes_input, cudaMemcpyHostToDevice);

    // 3. Kernel Launches
    // ------------------
    // LAUNCH 1: 65536 -> 128
    // Grid Size is 128. Block Size is 256.
    sumReductionIdleThreads<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);

    // LAUNCH 2: 128 -> 1
    // We launch 1 Block.
    // NOTE: This single block needs to reduce 128 elements.
    // Our kernel is designed to reduce (BlockDim * 2) elements.
    // If we launch with BlockDim=256, it expects 512 inputs!
    // But we only have 128 inputs.
    // For this specific input size (128), we can just launch with BlockDim=64
    // (since 64*2 = 128). Or we need boundary checks in the kernel.
    // Let's use BlockDim = 128 / 2 = 64 threads to handle 128 elements perfectly.
    sumReductionIdleThreads<<<1, 64>>>(d_v_r, d_v_r);

    // 4. Verify
    cudaMemcpy(h_v_r.data(), d_v_r, bytes_partial, cudaMemcpyDeviceToHost);

    assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));
    cout << "COMPLETED SUCCESSFULLY\n";

    cudaFree(d_v);
    cudaFree(d_v_r);

    return 0;
}