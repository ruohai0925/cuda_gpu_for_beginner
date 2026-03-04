// CUDA Sum Reduction - Step 6: Vectorized & Atomic (Cooperative Groups)
// Based on V5, enhanced with int4 loads, grid-stride loop, and atomicAdd.
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
#include <cooperative_groups.h>

using namespace cooperative_groups;
using std::accumulate;
using std::generate;
using std::cout;
using std::vector;

#define SHMEM_SIZE 256

// Unrolled warp-level reduction (same as V5)
__device__ void warpReduce(volatile int* v, int tid) {
    v[tid] += v[tid + 32];
    v[tid] += v[tid + 16];
    v[tid] += v[tid + 8];
    v[tid] += v[tid + 4];
    v[tid] += v[tid + 2];
    v[tid] += v[tid + 1];
}

// Single-stage kernel with grid-stride loop, int4 vectorized loads, and atomicAdd
__global__ void sumReductionV6(int *v, int *v_r, int n) {
    __shared__ int partial_sum[SHMEM_SIZE];
    thread_block g = this_thread_block();

    unsigned int tid = threadIdx.x;

    // Grid-stride loop with int4 vectorized loading
    int sum = 0;
    int4* v_vec = (int4*)v;
    int num_vec_elements = n / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < num_vec_elements; i += stride) {
        int4 loaded = v_vec[i];
        sum += loaded.x + loaded.y + loaded.z + loaded.w;
    }

    partial_sum[tid] = sum;
    g.sync();

    // Fold-in reduction loop, stop at 32 for warp unroll
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            partial_sum[tid] += partial_sum[tid + s];
        }
        g.sync();
    }

    // Unroll last warp
    if (tid < 32) {
        warpReduce(partial_sum, tid);
    }

    // Atomic aggregation to single scalar output
    if (tid == 0) {
        atomicAdd(v_r, partial_sum[0]);
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
    // Grid size: use grid-stride loop so any reasonable grid size works
    int GRID_SIZE = N / (TB_SIZE * 2);
    if (GRID_SIZE < 1) GRID_SIZE = 1;

    // Host data
    vector<int> h_v(N);
    generate(begin(h_v), end(h_v), [](){ return rand() % 10; });

    // Device data: single scalar output (atomicAdd)
    int *d_v, *d_v_r;
    cudaMalloc(&d_v, bytes_input);
    cudaMalloc(&d_v_r, sizeof(int));
    cudaMemcpy(d_v, h_v.data(), bytes_input, cudaMemcpyHostToDevice);

    // Warmup: run once and verify
    cudaMemset(d_v_r, 0, sizeof(int));
    sumReductionV6<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r, N);
    {
        int gpu_result;
        cudaMemcpy(&gpu_result, d_v_r, sizeof(int), cudaMemcpyDeviceToHost);
        assert(gpu_result == std::accumulate(begin(h_v), end(h_v), 0));
    }

    // Timed iterations (kernel-only, excluding memcpy)
    // NOTE: cudaMemset to reset d_v_r is placed BEFORE cudaEventRecord(start)
    // so the reset is NOT included in the timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_ms = 0.0f;

    for (int iter = 0; iter < M; iter++) {
        // Reset output to 0 (required for atomicAdd correctness)
        cudaMemset(d_v_r, 0, sizeof(int));

        cudaEventRecord(start);
        sumReductionV6<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r, N);
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

    return 0;
}
