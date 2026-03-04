// This program computes a sum reduction algorithm with warp divergence
// By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU
// Profiling support: accepts N (array size) and M (iterations) from command line

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

#define SHMEM_SIZE 256

// KERNEL: sumReduction (with warp divergence due to modulo operator)
__global__ void sumReduction(int *v, int *v_r) {
    __shared__ int partial_sum[SHMEM_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    partial_sum[threadIdx.x] = v[tid];
    __syncthreads();

    // Warp-divergent reduction: threadIdx.x % (2*s) causes divergence
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
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

    // Warmup: run full reduction once
    sumReduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);
    {
        int remaining = GRID_SIZE;
        int *src = d_v_r, *dst = d_v_r2;
        while (remaining > TB_SIZE) {
            int next_grid = remaining / TB_SIZE;
            sumReduction<<<next_grid, TB_SIZE>>>(src, dst);
            remaining = next_grid;
            int *tmp = src; src = dst; dst = tmp;
        }
        sumReduction<<<1, remaining>>>(src, src);
        // Verify correctness
        int gpu_result;
        cudaMemcpy(&gpu_result, src, sizeof(int), cudaMemcpyDeviceToHost);
        int cpu_result = std::accumulate(begin(h_v), end(h_v), 0);
        assert(gpu_result == cpu_result);
    }

    // Timed iterations (kernel-only, excluding memcpy)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_ms = 0.0f;

    for (int iter = 0; iter < M; iter++) {
        cudaEventRecord(start);

        // Stage 1: N elements -> GRID_SIZE partial sums
        sumReduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);

        // Multi-stage: reduce partial sums with ping-pong buffers
        int remaining = GRID_SIZE;
        int *src = d_v_r, *dst = d_v_r2;
        while (remaining > TB_SIZE) {
            int next_grid = remaining / TB_SIZE;
            sumReduction<<<next_grid, TB_SIZE>>>(src, dst);
            remaining = next_grid;
            int *tmp = src; src = dst; dst = tmp;
        }
        // Final single-block reduction
        sumReduction<<<1, remaining>>>(src, src);

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
