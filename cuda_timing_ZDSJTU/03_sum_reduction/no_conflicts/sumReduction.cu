// CUDA Sum Reduction - Step 3: No Bank Conflicts (Optimized)
// Original By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU
// Profiling support: accepts N (array size) and M (iterations) from command line

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

// Shrinking stride ("fold-in"): no bank conflicts, no divergence
__global__ void sumReductionNoConflicts(int *v, int *v_r) {
    __shared__ int partial_sum[SHMEM_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    partial_sum[threadIdx.x] = v[tid];
    __syncthreads();

    // Fold-in: stride starts at blockDim.x/2 and halves each iteration
    // Each thread accesses partial_sum[threadIdx.x] -> unique bank, zero conflicts
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments: N (array size) and M (timed iterations)
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <N> <M>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);

    size_t bytes_input = N * sizeof(int);
    const int TB_SIZE = 256;
    int GRID_SIZE = N / TB_SIZE;

    // Host data
    vector<int> h_v(N);
    generate(begin(h_v), end(h_v), [](){ return rand() % 10; });

    // Device data: two buffers for ping-pong multi-stage reduction
    int *d_v, *d_v_r, *d_v_r2;
    cudaMalloc(&d_v, bytes_input);
    cudaMalloc(&d_v_r, GRID_SIZE * sizeof(int));
    cudaMalloc(&d_v_r2, GRID_SIZE * sizeof(int));
    cudaMemcpy(d_v, h_v.data(), bytes_input, cudaMemcpyHostToDevice);

    // Warmup: run full reduction once and verify
    sumReductionNoConflicts<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);
    {
        int remaining = GRID_SIZE;
        int *src = d_v_r, *dst = d_v_r2;
        while (remaining > TB_SIZE) {
            int next_grid = remaining / TB_SIZE;
            sumReductionNoConflicts<<<next_grid, TB_SIZE>>>(src, dst);
            remaining = next_grid;
            int *tmp = src; src = dst; dst = tmp;
        }
        sumReductionNoConflicts<<<1, remaining>>>(src, src);
        int gpu_result;
        cudaMemcpy(&gpu_result, src, sizeof(int), cudaMemcpyDeviceToHost);
        assert(gpu_result == std::accumulate(begin(h_v), end(h_v), 0));
    }

    // Timed iterations (kernel-only, excluding memcpy)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_ms = 0.0f;

    for (int iter = 0; iter < M; iter++) {
        cudaEventRecord(start);

        // Stage 1: N -> GRID_SIZE partial sums
        sumReductionNoConflicts<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);

        // Multi-stage reduction with ping-pong buffers
        int remaining = GRID_SIZE;
        int *src = d_v_r, *dst = d_v_r2;
        while (remaining > TB_SIZE) {
            int next_grid = remaining / TB_SIZE;
            sumReductionNoConflicts<<<next_grid, TB_SIZE>>>(src, dst);
            remaining = next_grid;
            int *tmp = src; src = dst; dst = tmp;
        }
        sumReductionNoConflicts<<<1, remaining>>>(src, src);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    // Output average time in ms
    printf("%f\n", total_ms / M);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_v);
    cudaFree(d_v_r);
    cudaFree(d_v_r2);

    return 0;
}
