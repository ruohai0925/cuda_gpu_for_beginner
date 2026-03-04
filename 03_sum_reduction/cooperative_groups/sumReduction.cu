// CUDA Sum Reduction - Step 6: Vectorized & Atomic (Comparison Version)
// Based on V5, enhanced with int4 loads and atomicAdd.
// Updated by: ZDSJTU
// https://developer.nvidia.com/blog/cooperative-groups/
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <assert.h>
#include <cooperative_groups.h> // Required for modern sync

using namespace cooperative_groups;
using std::accumulate;
using std::generate;
using std::cout;
using std::vector;

#define SHMEM_SIZE 256

// -----------------------------------------------------------------------------
// DEVICE FUNCTION: warpReduce (EXACTLY THE SAME AS V5)
// -----------------------------------------------------------------------------
// We reuse the exact same unrolling logic from V5 to keep the comparison fair.
// -----------------------------------------------------------------------------
__device__ void warpReduce(volatile int* v, int tid) {
    v[tid] += v[tid + 32];
    v[tid] += v[tid + 16];
    v[tid] += v[tid + 8];
    v[tid] += v[tid + 4];
    v[tid] += v[tid + 2];
    v[tid] += v[tid + 1];
}

// -----------------------------------------------------------------------------
// KERNEL: sumReductionV6
// -----------------------------------------------------------------------------
// DIFFERENCE FROM V5:
// 1. Input: Takes 'n' (array size) because we use a Grid-Stride Loop.
// 2. Load: Uses 'int4' (Vectorized Load) instead of standard int load.
// 3. Output: Uses 'atomicAdd' to a single scalar, not an array.
// -----------------------------------------------------------------------------
__global__ void sumReductionV6(int *v, int *v_r, int n) {
    // 1. Shared Memory (Same as V5)
    __shared__ int partial_sum[SHMEM_SIZE];
    
    // Cooperative Groups handle (replaces raw __syncthreads for style, 
    // though __syncthreads works too)
    thread_block g = this_thread_block();

    unsigned int tid = threadIdx.x;

    // =========================================================================
    // DIFFERENCE 1: Grid-Stride Loop with Vectorized Loading (int4)
    // =========================================================================
    // In V5, we calculated one global index 'i'.
    // In V6, we loop through the array processing 4 integers at a time.
    // This allows the kernel to handle ANY array size 'n' with a fixed grid size.
    
    int sum = 0;
    
    // Cast input to int4* to load 128 bits (4 ints) per instruction
    int4* v_vec = (int4*)v; 
    int num_vec_elements = n / 4; // Assuming n is multiple of 4

    // Grid Stride: Total number of threads * 4 (since each thread does int4)
    // Actually, strictly speaking for grid stride on int4 types:
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < num_vec_elements; i += stride) {
        int4 loaded = v_vec[i]; // Single instruction LD.E.128
        sum += loaded.x + loaded.y + loaded.z + loaded.w;
    }
    // Blocks 0 - 63;
	// Blocks 64 - 127;
	

    // Store local sum to shared memory
    partial_sum[tid] = sum;
    
    g.sync(); // Equivalent to __syncthreads();

    // =========================================================================
    // PART 2: Reduction Loop (EXACTLY THE SAME AS V5)
    // =========================================================================
    // We use the exact same logic: fold until 32, then unroll.
    
    // Fold
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            partial_sum[tid] += partial_sum[tid + s];
        }
        g.sync();
    }

    // Unroll Last Warp (Same helper function as V5)
    if (tid < 32) {
        warpReduce(partial_sum, tid);
    }

    // =========================================================================
    // DIFFERENCE 2: Atomic Write Back
    // =========================================================================
    // V5: v_r[blockIdx.x] = partial_sum[0]; (Writes to array)
    // V6: atomicAdd(v_r, partial_sum[0]);   (Aggregates to single scalar)
    
    if (tid == 0) {
        atomicAdd(v_r, partial_sum[0]);
    }
}

int main() {
    // 1. Setup Data (Same N as V5 for fair comparison)
    int N = 1 << 16; 
    size_t bytes_input = N * sizeof(int);

    const int TB_SIZE = 256;
    int GRID_SIZE = N / (TB_SIZE * 2); 
    vector<int> h_v(N);
	int h_v_r; // Single scalar result
    generate(begin(h_v), end(h_v), [](){ return 1; });

    // DIFFERENCE: v_r is only 1 integer (4 bytes), not an array!
    int *d_v, *d_v_r;
    cudaMalloc(&d_v, bytes_input);
    cudaMalloc(&d_v_r, sizeof(int)); // Only 1 int needed for Atomic Add

    // Initialize result to 0 on GPU (Crucial for atomicAdd)
    int h_result_init = 0;
    cudaMemcpy(d_v_r, &h_result_init, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), bytes_input, cudaMemcpyHostToDevice);

    // 3. Launch Configuration
    // DIFFERENCE: V6 uses a fixed number of blocks (e.g., 128 or 256)
    // because the Grid-Stride Loop handles the full array size internally.
    // Let's use 128 blocks to match the V5 Grid Size, keeping variables controlled.


    // 4. Single Kernel Launch (V6 Style)
    // No need for a second launch.
    sumReductionV6<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r, N);

    // 5. Verify
    cudaMemcpy(&h_v_r, d_v_r, sizeof(int), cudaMemcpyDeviceToHost);
    
    assert(h_v_r == std::accumulate(begin(h_v), end(h_v), 0));

    cout << "COMPLETED SUCCESSFULLY\n";

    cudaFree(d_v);
    cudaFree(d_v_r);

    return 0;
}