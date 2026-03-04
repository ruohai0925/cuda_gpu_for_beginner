// CUDA Sum Reduction - Optimized (No Divergence / No Modulo)
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

__global__ void sumReduction(int *v, int *v_r) {
    // 1. Shared Memory Allocation
    __shared__ int partial_sum[SHMEM_SIZE];

    // 2. Load Data (Global -> Shared)
    // No change here. Linear loading is efficient.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    partial_sum[threadIdx.x] = v[tid];
    
    __syncthreads();

    // 3. Optimized Reduction Loop
    // ---------------------------
    // OLD (Naive): 
    // for (int s = 1; s < blockDim.x; s *= 2) {
    //     if (threadIdx.x % (2 * s) == 0) ...
    // }
    
    // NEW (Sequential):
    // s doubles: 1, 2, 4, 8...
    for (int s = 1; s < blockDim.x; s *= 2) {
        
        // MAPPING STRATEGY:
        // Instead of checking if *this* thread is a multiple of s,
        // we map threadIdx.x to a specific location in the array.
        //
        // Example (s=1):
        // Thread 0 -> index = 0. Adds partial_sum[0] + partial_sum[1]
        // Thread 1 -> index = 2. Adds partial_sum[2] + partial_sum[3]
        // Thread 2 -> index = 4. Adds partial_sum[4] + partial_sum[5]
        // ...
        // Notice threads 0, 1, 2 are CONTINUOUS. No gaps!
        
        int index = 2 * s * threadIdx.x;

        // BOUNDARY CHECK:
        // Only threads whose calculated index fits in the block do work.
        // As 's' grows, 'index' grows fast, so fewer threads qualify.
        // Eventually, only Thread 0 qualifies.
        if (index < blockDim.x) {
            partial_sum[index] += partial_sum[index + s];
        }

        __syncthreads();
    }

    // 4. Write Result
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

int main() {
    int N = 1 << 16; 
    size_t bytes_input = N * sizeof(int);

    const int TB_SIZE = 256;
    int GRID_SIZE = N / TB_SIZE;
    size_t bytes_partial = GRID_SIZE * sizeof(int); // Optimized size

    vector<int> h_v(N);
    vector<int> h_v_r(GRID_SIZE);

    generate(begin(h_v), end(h_v), [](){ return rand() % 10; });

    int *d_v, *d_v_r;
    cudaMalloc(&d_v, bytes_input);
    cudaMalloc(&d_v_r, bytes_partial);

    cudaMemcpy(d_v, h_v.data(), bytes_input, cudaMemcpyHostToDevice);

    // Launch 1
    sumReduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);
    // Launch 2
    sumReduction<<<1, GRID_SIZE>>> (d_v_r, d_v_r);

    cudaMemcpy(h_v_r.data(), d_v_r, bytes_partial, cudaMemcpyDeviceToHost);

    assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));
    cout << "COMPLETED SUCCESSFULLY\n";

    cudaFree(d_v);
    cudaFree(d_v_r);

    return 0;
}