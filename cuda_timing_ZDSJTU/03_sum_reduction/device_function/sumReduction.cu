// CUDA Sum Reduction - Step 5: Unroll Last Warp
// Original By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU
// Profiling support: accepts N (array size) and M (iterations) from command line
// NOTE: Requires N >= 512 (each block processes 2 * TB_SIZE elements)

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

// Device function: unrolled warp-level reduction (last 64 -> 1 elements)
// 'volatile' forces writes to shared memory (prevents register caching)
__device__ void warpReduce(volatile int* v, int tid) {
    v[tid] += v[tid + 32];
    v[tid] += v[tid + 16];
    v[tid] += v[tid + 8];
    v[tid] += v[tid + 4];
    v[tid] += v[tid + 2];
    v[tid] += v[tid + 1];
}

// Double-load + unrolled warp reduction kernel
__global__ void sumReductionUnroll(int *v, int *v_r) {
    __shared__ int partial_sum[SHMEM_SIZE];

    // Double load (same as reduce_idle)
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
    __syncthreads();

    // Adaptive reduction loop:
    // - When blockDim.x > 64: loop stops at s=32, then use warpReduce
    // - When blockDim.x <= 64: loop runs all the way to s=1 (safe for small blocks)
    int stop_at = (blockDim.x > 64) ? 32 : 0;

    for (int s = blockDim.x / 2; s > stop_at; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Unrolled warp reduce (only safe when blockDim.x > 64)
    if (stop_at == 32 && threadIdx.x < 32) {
        warpReduce(partial_sum, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments: N (array size) and M (timed iterations)
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <N> <M>\n", argv[0]);
        fprintf(stderr, "  N must be >= 512 and a power-of-2 multiple of 256\n");
        return 1;
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);

    size_t bytes_input = N * sizeof(int);
    const int TB_SIZE = 256;
    // Grid halved: each block processes 2 * TB_SIZE elements
    int GRID_SIZE = N / (TB_SIZE * 2);

    if (GRID_SIZE < 1) {
        fprintf(stderr, "Error: N=%d too small for double-load (need N >= 512)\n", N);
        return 1;
    }

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
    sumReductionUnroll<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);
    {
        int remaining = GRID_SIZE;
        int *src = d_v_r, *dst = d_v_r2;
        while (remaining > 1) {
            if (remaining >= TB_SIZE * 2) {
                int next_grid = remaining / (TB_SIZE * 2);
                sumReductionUnroll<<<next_grid, TB_SIZE>>>(src, dst);
                remaining = next_grid;
                int *tmp = src; src = dst; dst = tmp;
            } else {
                int final_threads = remaining / 2;
                if (final_threads < 1) final_threads = 1;
                sumReductionUnroll<<<1, final_threads>>>(src, src);
                remaining = 1;
            }
        }
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
        sumReductionUnroll<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);

        // Multi-stage reduction with ping-pong buffers
        int remaining = GRID_SIZE;
        int *src = d_v_r, *dst = d_v_r2;
        while (remaining > 1) {
            if (remaining >= TB_SIZE * 2) {
                int next_grid = remaining / (TB_SIZE * 2);
                sumReductionUnroll<<<next_grid, TB_SIZE>>>(src, dst);
                remaining = next_grid;
                int *tmp = src; src = dst; dst = tmp;
            } else {
                int final_threads = remaining / 2;
                if (final_threads < 1) final_threads = 1;
                sumReductionUnroll<<<1, final_threads>>>(src, src);
                remaining = 1;
            }
        }

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
