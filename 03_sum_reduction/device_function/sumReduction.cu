// CUDA Sum Reduction - Step 5: Unroll Last Warp
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
// DEVICE FUNCTION: warpReduce
// -----------------------------------------------------------------------------
// This function handles the final reduction for the last 32 threads (Warp 0).
// Because we are within a single warp, we do not need __syncthreads().
//
// CRITICAL: 'volatile' is needed!
// It prevents the compiler from caching 'v[tid]' in registers.
// It forces the threads to write partially calculated sums back to Shared Memory
// immediately, so other threads (e.g., tid 0 reading tid 16's result) see the update.
// -----------------------------------------------------------------------------
__device__ void warpReduce(volatile int* v, int tid) {
    // Unrolled loop. No 'for', no 'if', no 'barrier'.
    // Just raw instruction stream.
    // Logic:
    // v[tid] += v[tid + 32]; (Pre-condition: s=64 finished)
    // v[tid] += v[tid + 16];
    // ...
    // v[tid] += v[tid + 1];
    
    // Note: We don't need bounds checks (tid < 32) here because 
    // this function is ONLY called by the first 32 threads.
    
    v[tid] += v[tid + 32];
    v[tid] += v[tid + 16];
    v[tid] += v[tid + 8];
    v[tid] += v[tid + 4];
    v[tid] += v[tid + 2];
    v[tid] += v[tid + 1];
}

// -----------------------------------------------------------------------------
// KERNEL: sumReductionUnroll
// -----------------------------------------------------------------------------
__global__ void sumReductionUnroll(int *v, int *v_r) {
    // 1. Shared Memory
    __shared__ int partial_sum[SHMEM_SIZE];

    // 2. Load & Double Add (Same as V4)
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
    __syncthreads();

    // 3. The Reduction Loop
    // ---------------------
    // IMPORTANT CHANGE: We stop the loop when s <= 32.
    // Because beyond that point, only Warp 0 is active.
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // 4. Hand over to Warp Reduce (The Last 32 Threads)
    // -------------------------------------------------
    // Only the first warp (tid 0-31) enters here.
    if (threadIdx.x < 32) {
        warpReduce(partial_sum, threadIdx.x);
    }

    // 5. Write Result
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

int main() {
    // Setup (Identical to Version 4)
    int N = 1 << 16; 
    size_t bytes_input = N * sizeof(int);

    const int TB_SIZE = 256;
    // Remember V4 optimization: Halve the grid size
    int GRID_SIZE = N / (TB_SIZE * 2); 
    size_t bytes_partial = GRID_SIZE * sizeof(int);

    vector<int> h_v(N);
    vector<int> h_v_r(GRID_SIZE);
    generate(begin(h_v), end(h_v), [](){ return 1; });

    int *d_v, *d_v_r;
    cudaMalloc(&d_v, bytes_input);
    cudaMalloc(&d_v_r, bytes_partial);

    cudaMemcpy(d_v, h_v.data(), bytes_input, cudaMemcpyHostToDevice);

    // Launch 1
    sumReductionUnroll<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);

    // Launch 2 (Finish the job)
    // Reduce the partial sums.
    // Grid Size 128 -> handled by 1 block with 64 threads (covering 128 elements)
    sumReductionUnroll<<<1, 64>>>(d_v_r, d_v_r);

    cudaMemcpy(h_v_r.data(), d_v_r, bytes_partial, cudaMemcpyDeviceToHost);
    
    assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));

    cout << "COMPLETED SUCCESSFULLY\n";

    cudaFree(d_v);
    cudaFree(d_v_r);

    return 0;
}