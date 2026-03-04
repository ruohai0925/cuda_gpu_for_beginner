// This program performs sum reduction with an optimization removing warp bank conflicts.
// MORE IMPORTANTLY: It demonstrates Microscopic Profiling using the clock() function
// to expose Load Imbalance across different Thread Blocks.
// By: Nick from CoffeeBeforeArch
// Updated by: ZDSJTU

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <iostream>

using namespace std;

#define SIZE 256
// Note: 256 * 4 is hardcoded here, but strictly we only need 256 elements per block
#define SHMEM_SIZE 256 * 4 

// =================================================================================
// GPU KERNEL: Sum Reduction with Block-Level Profiling
// Takes:
//  v: Input vector
//  v_r: Partial result vector
//  time: Array to store the start and end clock cycles for each Thread Block
// =================================================================================
__global__ void sum_reduction(int *v, int *v_r, clock_t *time) {
    
    // =================================================================================
    // KNOWLEDGE HINT 1: Delegating the Timekeeper & The Hardware Stopwatch
    // ---------------------------------------------------------------------------------
    // We want to measure the lifespan of the ENTIRE Thread Block. We don't need 256 
    // threads to press the stopwatch. We elect Thread 0 as the sole "Timekeeper".
    // 
    // clock() accesses the per-multiprocessor hardware counter. It returns the exact 
    // clock cycle at this specific moment. 
    // 
    // WARNING (The Overflow Trap): clock() returns a 32-bit integer. On a 1.5 GHz GPU, 
    // this counter overflows (wraps around to 0) in less than 3 seconds! 
    // For long-running kernels, you MUST use clock64() instead.
    // =================================================================================
    if(threadIdx.x == 0){
        // Store the START time in the first half of the 'time' array
        time[blockIdx.x] = clock();
    }

    // Allocate shared memory for the block's partial sum
    __shared__ int partial_sum[SHMEM_SIZE];

    // Calculate absolute global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements from slow Global Memory into fast Shared Memory
    partial_sum[threadIdx.x] = v[tid];
    
    // Barrier: Wait until everyone has loaded their data
    __syncthreads();

    // =================================================================================
    // KNOWLEDGE HINT 2: The Source of Load Imbalance
    // ---------------------------------------------------------------------------------
    // This is the classic reduction tree. In the first iteration, 128 threads work. 
    // In the second, 64 threads work. Then 32, 16, 8, etc. 
    // Because threads are systematically retiring/dropping out, earlier blocks scheduled 
    // on the GPU might end up waiting longer or taking a different execution path than 
    // later blocks. This creates microscopic differences in execution time (Load Imbalance).
    // =================================================================================
    
    // Start at 1/2 block stride and divide by two each iteration
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Let thread 0 for this block write its final computed result to main memory
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
    
    // =================================================================================
    // KNOWLEDGE HINT 3: Recording the Finish Line
    // ---------------------------------------------------------------------------------
    // The work is completely done. Thread 0 presses the stopwatch again.
    // To prevent overwriting our start times, we store the END time in the second 
    // half of the 'time' array (offset by gridDim.x, which is the total number of blocks).
    // =================================================================================
    if(threadIdx.x == 0){
        time[blockIdx.x + gridDim.x] = clock();
    }
}

void initialize_vector(int *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = 1;
    }
}

int main() {
    // Vector size: 2^16 = 65,536 elements
    int n = 1 << 16;
    size_t bytes = n * sizeof(int);

    // Host and Device pointers
    int *h_v, *h_v_r;
    int *d_v, *d_v_r;

    // Allocate memory
    h_v = (int*)malloc(bytes);
    h_v_r = (int*)malloc(bytes);
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_v_r, bytes);

    // Initialize vector with 1s
    initialize_vector(h_v, n);

    // Copy to device
    cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

    // Thread Block Size
    int TB_SIZE = SIZE;

    // Grid Size: 65536 / 256 = 256 Blocks
    int GRID_SIZE = n / TB_SIZE;

    // =================================================================================
    // Allocating the Profiling Array
    // We need 2 slots for EVERY block (one for start time, one for end time).
    // So we allocate GRID_SIZE * 2.
    // =================================================================================
    clock_t *time = new clock_t[GRID_SIZE * 2];
    clock_t *d_time;
    cudaMalloc(&d_time, sizeof(clock_t) * GRID_SIZE * 2);

    // Launch the FIRST reduction pass (reduces 65,536 elements to 256 partial sums)
    sum_reduction <<<GRID_SIZE, TB_SIZE >>> (d_v, d_v_r, d_time);

    // Fetch the timing data back to the CPU immediately after the first pass
    // We only care about this first pass because it has 256 blocks to compare!
    cudaMemcpy(time, d_time, sizeof(clock_t) * GRID_SIZE * 2, cudaMemcpyDeviceToHost);

    // Launch the SECOND reduction pass (reduces the 256 partial sums into 1 final sum)
    // We launch exactly 1 Block here. Profiling 1 block doesn't show load imbalance.
    sum_reduction <<<1, TB_SIZE >>> (d_v_r, d_v_r, d_time);

    // Copy final result to host
    cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

    // =================================================================================
    // Outputting the Profiling Data (CSV Format for Excel/Gnuplot)
    // We subtract the start time (time[i]) from the end time (time[i + GRID_SIZE])
    // to get the total elapsed clock cycles for Block 'i'.
    // =================================================================================
    cout << "Block,Clocks" << endl; 
    for(int i = 0; i < GRID_SIZE; i++){
        cout << i << "," << (time[i+GRID_SIZE] - time[i]) << endl;
    }

    return 0;
}